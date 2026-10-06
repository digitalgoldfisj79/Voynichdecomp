#!/usr/bin/env python3
"""
VMS-NEUROCIPHER-GER2 cached runtime.
Scientific algorithm/schedule is identical to GER1 / validated NeuroCipher.
Performance-only changes:
- load all static VMS/German vocabularies from a frozen JSON cache
- cap RapidFuzz workers per replica via NDEC_RAPIDFUZZ_WORKERS
- suitable for multiple independent replicas sharing one GPU.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,itertools,json,os,random,urllib.request
import numpy as np, torch
import neurodecipher_acl2019_refactor as nd

DISC_N=735; TRAIN_K=4103; FULL_K=10000; DISC_DEMAND=221
RATIO=DISC_DEMAND/DISC_N

# Performance-only worker cap: same cdist/scorer, fewer CPU threads per replica.
_orig_cdist=nd.process.cdist
def _cdist_limited(*a,**kw):
    kw["workers"]=int(os.environ.get("NDEC_RAPIDFUZZ_WORKERS","3"))
    return _orig_cdist(*a,**kw)
nd.process.cdist=_cdist_limited

def load_json(src):
    if src.startswith("http://") or src.startswith("https://"):
        raw=urllib.request.urlopen(src,timeout=120).read()
    else:
        raw=open(src,"rb").read()
    if src.endswith(".gz.b64"):
        raw=gzip.decompress(base64.b64decode(raw.strip()))
    return json.loads(raw.decode())

def scrambled_unique_vocab(forms,seed):
    out=[];seen=set();unchanged=0;random_tries=0;enumerated=0
    for w in forms:
        if len(w)<2 or len(set(w))<2:
            cand=w
        else:
            h=int(hashlib.sha256((str(seed)+"|"+w).encode()).hexdigest()[:16],16)
            rng=random.Random(h);a=list(w);cand=None
            for _ in range(512):
                rng.shuffle(a);q="".join(a);random_tries+=1
                if q!=w and q not in seen:
                    cand=q;break
            if cand is None and len(w)<=8:
                perms=sorted(set("".join(p) for p in itertools.permutations(w)))
                off=h%max(1,len(perms))
                for j in range(len(perms)):
                    q=perms[(off+j)%len(perms)];enumerated+=1
                    if q!=w and q not in seen:
                        cand=q;break
            if cand is None:cand=w
        if cand in seen:raise RuntimeError(("scramble uniqueness impossible",w,cand))
        if cand==w:unchanged+=1
        seen.add(cand);out.append(cand)
    return out,{"unchanged":unchanged,"random_tries":random_tries,"enumerated":enumerated,
                "unique":len(seen),"n":len(forms)}

class CachedRunner:
    def __init__(self,args):
        self.a=args
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
        self.dev=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        c=load_json(args.cache)
        if c.get("schema")!="VMS_NEUROCIPHER_GER2_CACHE_V2":raise RuntimeError("cache schema")
        v=c["vms"];d=c["dialects"][args.dialect]
        self.disc=v["discovery"];self.val=v["validation"]
        self.fin_unseen=v["final_unseen_discovery"];self.fin_strict=v["final_strict_novel"]
        self.vchars=v["chars"];self.v_audit=v["audit"]
        base=d["real"]
        if args.mode=="real":
            full=base;ctrl={"mode":"real","n":len(full)}
        else:
            full,a=scrambled_unique_vocab(base,args.scramble_seed)
            ctrl={"mode":"scramble","seed":args.scramble_seed,**a}
        if len(full)!=FULL_K:raise RuntimeError(("target full",len(full)))
        self.train_known=full[:TRAIN_K];self.full_known=full
        self.lcs=nd.Charset(self.vchars);self.kcs=nd.Charset(sorted(set("".join(full))))
        self.lost=sorted(self.disc,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        self.known=list(self.train_known)
        self.lost_ids,self.lost_len=nd.pad_words(self.lost,self.lcs,self.dev)
        self.known_ids,self.known_len=nd.pad_words(self.known,self.kcs,self.dev)
        self.flow=np.zeros((len(self.lost),len(self.known)),np.float32)
        self.model=nd.NeuroCipher(len(self.lcs),len(self.kcs),dropout=.3).to(self.dev)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)
        self.audit={"dialect":args.dialect,"mode":args.mode,"seed":args.seed,
                    "scramble_seed":args.scramble_seed,"target_train":len(self.train_known),
                    "target_full":len(self.full_known),"target_chars":len(self.kcs)-4,
                    "ref_docs":d["docs"],"ref_types":d["ref_types"],"control":ctrl,
                    "vms":self.v_audit,"cache_schema":c["schema"],
                    "screen_commit":c["screen_commit"],"solver_commit":c["solver_commit"],
                    "rapidfuzz_workers":int(os.environ.get("NDEC_RAPIDFUZZ_WORKERS","3"))}
        print("VGER2_AUDIT="+json.dumps(self.audit,ensure_ascii=False,separators=(",",":")),flush=True)

    def reset_model(self):
        self.model.reinit_like_upstream();self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)

    def model_disc(self):
        self.model.eval()
        with torch.no_grad():return self.model(self.lost_ids,self.lost_len,self.known_ids,self.known_len)

    def e_step(self,demand,edit):
        lp,sc,_=self.model_disc()
        costs=nd.expected_edits(lp,sc,self.known,self.kcs,edit)
        nf,cost=nd.mincost(costs,demand,5,3)
        self.flow=.25*self.flow+.75*nf
        print("VGER2_ESTEP="+json.dumps({"demand":demand,"edit":edit,"mcf_cost":cost,
             "flow_nonzero":int((nf.sum(1)>0).sum())}),flush=True)

    def train_epoch(self):
        self.model.train();last=(0.,0.,0.)
        for ids in nd.batches(len(self.known),500,self.dev):
            ids=sorted(ids,key=lambda j:int(self.known_len[j]),reverse=True)
            tid=self.known_ids[ids];tl=self.known_len[ids]
            lp,sc,reg=self.model(self.lost_ids,self.lost_len,tid,tl)
            fs=torch.tensor(self.flow[:,ids],dtype=torch.float32,device=self.dev)
            fk=fs.sum(0);total=fk.sum()
            if float(total)<=0:continue
            nll=torch.logsumexp(sc+torch.log(fs+1e-8),dim=0)
            nll=-(nll*fk).sum()/total;rloss=reg/total;loss=nll+self.a.reg_hyper*rloss
            self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0);self.opt.step()
            last=(float(loss.detach()),float(nll.detach()),float(rloss.detach()))
        return last

    def eval_cost(self,lost_forms,known_forms,label,eval_seed):
        if not lost_forms:return {"n":0,"demand":0,"mean_edit_cost":None}
        lf=sorted(lost_forms,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        kid,klen=nd.pad_words(known_forms,self.kcs,self.dev)
        n=len(lf);k=len(known_forms);costs=np.empty((n,k),np.float32)
        cpu_state=torch.random.get_rng_state()
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(eval_seed)
        self.model.eval()
        try:
            for st in range(0,n,self.a.eval_batch):
                en=min(st+self.a.eval_batch,n);lid,llen=nd.pad_words(lf[st:en],self.lcs,self.dev)
                with torch.no_grad():lp,sc,_=self.model(lid,llen,kid,klen)
                costs[st:en]=nd.expected_edits(lp,sc,known_forms,self.kcs,True)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
        demand=max(1,min(n,int(round(RATIO*n))))
        flow,icost=nd.mincost(costs,demand,5,3)
        mean=float((flow*costs).sum()/max(flow.sum(),1))
        vals=costs[flow>0]
        out={"label":label,"n":n,"known":k,"demand":demand,"mean_edit_cost":mean,
             "median_selected_cost":float(np.median(vals)) if len(vals) else None,
             "mcf_integer_cost":int(icost)}
        print("VGER2_EVAL="+json.dumps(out,separators=(",",":")),flush=True)
        return out

    def run(self):
        if self.a.smoke:
            lp,sc,reg=self.model_disc()
            print("VGER2_SMOKE="+json.dumps({"pass":bool(torch.isfinite(sc).all()),"scores_shape":list(sc.shape),
                  "device":str(self.dev),"audit":self.audit},ensure_ascii=False,separators=(",",":")),flush=True)
            return
        best=None;best_state=None
        for rnd in range(1,self.a.rounds+1):
            if rnd==1:
                self.flow.fill(DISC_DEMAND/self.flow.size)
                print("VGER2_ESTEP="+json.dumps({"round":1,"warmup_uniform":True,
                     "flow_total":float(self.flow.sum())}),flush=True)
            else:
                self.e_step(min((rnd-1)*50,DISC_DEMAND),edit=(rnd>self.a.warm_up_steps));self.reset_model()
            for ep in range(1,self.a.epochs+1):
                loss,nll,reg=self.train_epoch();ge=(rnd-1)*self.a.epochs+ep
                if ep%self.a.log_every==0:
                    print("VGER2_TRAIN="+json.dumps({"round":rnd,"epoch":ep,"global_epoch":ge,
                          "loss":loss,"nll":nll,"reg":reg}),flush=True)
                if ep%self.a.eval_every==0:
                    vv=self.eval_cost(self.val,self.train_known,"validation",8081)
                    meta={"round":rnd,"epoch":ep,"global_epoch":ge,"validation":vv}
                    if best is None or vv["mean_edit_cost"]<best["validation"]["mean_edit_cost"]:
                        best=meta;best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                        print("VGER2_BEST_CHECKPOINT="+json.dumps(best,separators=(",",":")),flush=True)
        self.model.load_state_dict(best_state);self.model.to(self.dev)
        print("VGER2_SELECTED_CHECKPOINT="+json.dumps(best,separators=(",",":")),flush=True)
        fs=self.eval_cost(self.fin_strict,self.full_known,"final_strict_novel",9191)
        fu=self.eval_cost(self.fin_unseen,self.full_known,"final_unseen_discovery",9292)
        out={"phase":"VMS_NEUROCIPHER_GER2","status":"complete","audit":self.audit,
             "selected_checkpoint":best,"final_strict_novel":fs,"final_unseen_discovery":fu}
        print("VGER2_RESULT_JSON="+json.dumps(out,ensure_ascii=False,separators=(",",":")),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",required=True)
    p.add_argument("--dialect",choices=["BAV","ALEM"],required=True)
    p.add_argument("--mode",choices=["real","scramble"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--scramble-seed",type=int,default=9001)
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--rounds",type=int,default=10);p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--eval-every",type=int,default=10);p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--warm-up-steps",type=int,default=5);p.add_argument("--reg-hyper",type=float,default=.5)
    p.add_argument("--eval-batch",type=int,default=48)
    CachedRunner(p.parse_args()).run()
if __name__=="__main__":main()
