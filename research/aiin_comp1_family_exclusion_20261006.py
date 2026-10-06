#!/usr/bin/env python3
"""
AIIN-COMP1 — prospective family-exclusion test.

Hypothesis frozen before run:
  VMS terminal -aiin behaves compositionally: the suffix predicts an Alemannic
  -lich/-liche/-lichen/-licher/-liches/-lichem target class, while the VMS
  material before -aiin retains target-stem information.

No -aiin token is present in lost-side training. Exact fixed checkpoints are
imported from the already-adjudicated GER2 ALEM runs, so this experiment performs
no checkpoint selection or tuning.

Primary observables:
  1) unconstrained MCF assignments of held-out -aiin tokens are enriched in LICH
     vs 500 length-matched random target classes;
  2) mean minimum expected-edit cost to the LICH class beats length-matched random
     target classes;
  3) -aiin is unusually close to LICH versus matched VMS 4-char suffix families;
  4) VMS pre-aiin stem distance correlates with assigned German pre-lich stem
     distance, against permutation.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,itertools,json,os,random,re,urllib.request
import numpy as np, torch
from scipy.stats import rankdata
import neurodecipher_acl2019_refactor as nd

TRAIN_K=4103; FULL_K=10000; DISC_DEMAND=221
CHECKPOINTS={
  1234:(10,130),
  2026:(10,140),
  17:(10,130),
  73:(9,130),
}
LICH_RE=re.compile(r"lich(?:e|en|er|es|em)?$",re.I)

_orig_cdist=nd.process.cdist
def _cdist_limited(*a,**kw):
    kw["workers"]=int(os.environ.get("NDEC_RAPIDFUZZ_WORKERS","1"))
    return _orig_cdist(*a,**kw)
nd.process.cdist=_cdist_limited

def load_json(src):
    raw=urllib.request.urlopen(src,timeout=120).read() if src.startswith("http") else open(src,"rb").read()
    if src.endswith(".gz.b64"): raw=gzip.decompress(base64.b64decode(raw.strip()))
    return json.loads(raw.decode())

def nlev(a,b):
    if a==b:return 0.0
    m,n=len(a),len(b)
    if not m or not n:return 1.0
    prev=list(range(n+1))
    for i,ca in enumerate(a,1):
        cur=[i]+[0]*n
        for j,cb in enumerate(b,1):
            cur[j]=min(cur[j-1]+1,prev[j]+1,prev[j-1]+(ca!=cb))
        prev=cur
    return prev[n]/max(m,n)

def spearman(x,y):
    if len(x)<3:return float("nan")
    rx=rankdata(np.asarray(x,float)); ry=rankdata(np.asarray(y,float))
    if np.std(rx)==0 or np.std(ry)==0:return 0.0
    return float(np.corrcoef(rx,ry)[0,1])

class Run:
    def __init__(self,a):
        self.a=a
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vc=json.load(open(a.vms_cache))
        gc=load_json(a.german_cache)
        assert vc["schema"]=="VMS_AIIN_COMP1_V1"
        assert gc["schema"]=="VMS_NEUROCIPHER_GER2_CACHE_V2"
        self.train=list(vc["train"]); self.aiin=list(vc["aiin"])
        self.strict=list(vc["strict_final"]); self.controls=list(vc["suffix_controls"])
        self.full_known=list(gc["dialects"]["ALEM"]["real"]); self.known=self.full_known[:TRAIN_K]
        self.lcs=nd.Charset(sorted(set("".join(self.train+self.aiin+self.strict))))
        self.kcs=nd.Charset(sorted(set("".join(self.full_known))))
        self.lost=sorted(self.train,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        self.lost_ids,self.lost_len=nd.pad_words(self.lost,self.lcs,self.dev)
        self.known_ids,self.known_len=nd.pad_words(self.known,self.kcs,self.dev)
        self.flow=np.zeros((len(self.lost),len(self.known)),np.float32)
        self.model=nd.NeuroCipher(len(self.lcs),len(self.kcs),dropout=.3).to(self.dev)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)
        self.tgt_round,self.tgt_epoch=CHECKPOINTS[a.seed]
        self.lich_idx=np.array([i for i,w in enumerate(self.full_known) if LICH_RE.search(w)],dtype=int)
        if len(self.lich_idx)<22: raise RuntimeError(("LICH class too small",len(self.lich_idx)))
        self.audit={"seed":a.seed,"checkpoint":[self.tgt_round,self.tgt_epoch],
          "train_n":len(self.train),"train_aiin_n":sum(w.endswith("aiin") for w in self.train),
          "heldout_aiin_n":len(self.aiin),"strict_n":len(self.strict),
          "lich_target_n":len(self.lich_idx),"target_n":len(self.full_known),
          "suffix_controls_n":len(self.controls)}
        print("AIIN_COMP1_AUDIT="+json.dumps(self.audit,separators=(",",":")),flush=True)

    def reset_model(self):
        self.model.reinit_like_upstream(); self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)

    def e_step(self,demand,edit):
        self.model.eval()
        with torch.no_grad(): lp,sc,_=self.model(self.lost_ids,self.lost_len,self.known_ids,self.known_len)
        costs=nd.expected_edits(lp,sc,self.known,self.kcs,edit)
        nf,cost=nd.mincost(costs,demand,5,3)
        self.flow=.25*self.flow+.75*nf
        print("AIIN_COMP1_ESTEP="+json.dumps({"seed":self.a.seed,"demand":demand,"edit":edit,"cost":cost}),flush=True)

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
            nll=-(nll*fk).sum()/total; rloss=reg/total; loss=nll+.5*rloss
            self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0);self.opt.step()
            last=(float(loss.detach()),float(nll.detach()),float(rloss.detach()))
        return last

    def costs(self,lost_forms,known_forms,seed):
        lf=sorted(lost_forms,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        kid,klen=nd.pad_words(known_forms,self.kcs,self.dev)
        out=np.empty((len(lf),len(known_forms)),np.float32)
        cs=torch.random.get_rng_state(); gs=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        self.model.eval()
        try:
            for st in range(0,len(lf),48):
                en=min(st+48,len(lf)); lid,llen=nd.pad_words(lf[st:en],self.lcs,self.dev)
                with torch.no_grad():lp,sc,_=self.model(lid,llen,kid,klen)
                out[st:en]=nd.expected_edits(lp,sc,known_forms,self.kcs,True)
        finally:
            torch.random.set_rng_state(cs)
            if gs is not None:torch.cuda.set_rng_state_all(gs)
        return lf,out

    def run(self):
        for rnd in range(1,self.tgt_round+1):
            if rnd==1:self.flow.fill(DISC_DEMAND/self.flow.size)
            else:
                self.e_step(min((rnd-1)*50,DISC_DEMAND),edit=(rnd>5));self.reset_model()
            mx=self.tgt_epoch if rnd==self.tgt_round else 150
            for ep in range(1,mx+1):
                loss,nll,reg=self.train_epoch()
                if ep%50==0 or ep==mx:
                    print("AIIN_COMP1_TRAIN="+json.dumps({"seed":self.a.seed,"round":rnd,"epoch":ep,"loss":loss}),flush=True)

        aiin_sorted,cfull=self.costs(self.aiin,self.full_known,62026)
        lich=set(self.lich_idx.tolist())
        # Unconstrained assignment, then ask how often it independently lands in LICH.
        flow,_=nd.mincost(cfull,len(aiin_sorted),5,3)
        assigned=[]
        for i,w in enumerate(aiin_sorted):
            js=np.where(flow[i]>0)[0]
            if len(js)!=1: raise RuntimeError(("assignment",w,len(js)))
            j=int(js[0]);assigned.append((w,j,self.full_known[j],float(cfull[i,j])))
        hit=sum(j in lich for _,j,_,_ in assigned)

        # Length-matched random German classes, frozen RNG.
        rng=np.random.default_rng(77117+self.a.seed)
        nonlich=[i for i in range(len(self.full_known)) if i not in lich]
        bylen={}
        for i in nonlich:bylen.setdefault(len(self.full_known[i]),[]).append(i)
        lich_lens=[len(self.full_known[i]) for i in self.lich_idx]
        lens_count={}
        for L in lich_lens:lens_count[L]=lens_count.get(L,0)+1
        null_sets=[]
        for b in range(500):
            ss=[]
            for L,n in lens_count.items():
                pool=bylen.get(L,[])
                if len(pool)>=n: ss.extend(rng.choice(pool,n,replace=False).tolist())
                else: ss.extend(rng.choice(nonlich,n,replace=False).tolist())
            null_sets.append(np.array(ss,dtype=int))
        obs_cost=float(np.min(cfull[:,self.lich_idx],axis=1).mean())
        null_cost=np.array([float(np.min(cfull[:,idx],axis=1).mean()) for idx in null_sets])
        null_hit=np.array([sum(j in set(idx.tolist()) for _,j,_,_ in assigned) for idx in null_sets],float)
        cost_z=float((null_cost.mean()-obs_cost)/(null_cost.std(ddof=1) or 1))
        cost_p=float((1+np.sum(null_cost<=obs_cost))/(len(null_cost)+1))
        hit_z=float((hit-null_hit.mean())/(null_hit.std(ddof=1) or 1))
        hit_p=float((1+np.sum(null_hit>=hit))/(len(null_hit)+1))

        # VMS suffix-family null: score every matched 4-char family to the same LICH class.
        lich_words=[self.full_known[i] for i in self.lich_idx]
        all_ctrl_tokens=sorted(set(w for c in self.controls for w in c["tokens"]))
        toks,cl=self.costs(all_ctrl_tokens,lich_words,62027)
        row={w:i for i,w in enumerate(toks)}
        aiin_lens=np.mean([len(w) for w in self.aiin])
        famscores=[]
        for c in self.controls:
            ws=[w for w in c["tokens"] if w in row]
            if len(ws)<10:continue
            if abs(np.mean([len(w) for w in ws])-aiin_lens)>1.5:continue
            vals=[float(np.min(cl[row[w]])) for w in ws]
            famscores.append({"suffix":c["suffix"],"n":len(ws),"score":float(np.mean(vals))})
        af,acl=self.costs(self.aiin,lich_words,62027)
        aiin_class_score=float(np.min(acl,axis=1).mean())
        fvals=np.array([x["score"] for x in famscores],float)
        fam_z=float((fvals.mean()-aiin_class_score)/(fvals.std(ddof=1) or 1)) if len(fvals)>1 else float("nan")
        fam_p=float((1+np.sum(fvals<=aiin_class_score))/(len(fvals)+1)) if len(fvals) else 1.0

        # Force a capacity-limited assignment within LICH, then test stem-to-stem geometry.
        flich,_=nd.mincost(acl,len(af),5,3)
        pairs=[]
        for i,w in enumerate(af):
            js=np.where(flich[i]>0)[0]
            if len(js)!=1:continue
            tw=lich_words[int(js[0])]
            gst=LICH_RE.sub("",tw)
            if gst:pairs.append((w[:-4],gst,w,tw))
        vx=[];gy=[]
        for i in range(len(pairs)):
            for j in range(i+1,len(pairs)):
                vx.append(nlev(pairs[i][0],pairs[j][0]));gy.append(nlev(pairs[i][1],pairs[j][1]))
        rho=spearman(vx,gy)
        prng=np.random.default_rng(88119+self.a.seed);gst=[p[1] for p in pairs];nullrho=[]
        vst=[p[0] for p in pairs]
        for _ in range(1000):
            pg=list(prng.permutation(gst)); xx=[];yy=[]
            for i in range(len(vst)):
                for j in range(i+1,len(vst)):
                    xx.append(nlev(vst[i],vst[j]));yy.append(nlev(pg[i],pg[j]))
            nullrho.append(spearman(xx,yy))
        nr=np.array(nullrho,float)
        rho_z=float((rho-nr.mean())/(nr.std(ddof=1) or 1))
        rho_p=float((1+np.sum(nr>=rho))/(len(nr)+1))

        res={"seed":self.a.seed,"audit":self.audit,
          "unconstrained":{"lich_hits":hit,"n":len(assigned),"rate":hit/len(assigned),
            "null_hit_mean":float(null_hit.mean()),"null_hit_sd":float(null_hit.std(ddof=1)),
            "z":hit_z,"p":hit_p},
          "class_cost":{"obs":obs_cost,"null_mean":float(null_cost.mean()),"null_sd":float(null_cost.std(ddof=1)),
            "z":cost_z,"p":cost_p},
          "vms_suffix_null":{"aiin_score":aiin_class_score,"control_n":len(fvals),
            "control_mean":float(fvals.mean()) if len(fvals) else None,
            "control_sd":float(fvals.std(ddof=1)) if len(fvals)>1 else None,"z":fam_z,"p":fam_p,
            "best_controls":sorted(famscores,key=lambda x:x["score"])[:10]},
          "stem_geometry":{"pairs":len(pairs),"rho":rho,"null_mean":float(nr.mean()),
            "null_sd":float(nr.std(ddof=1)),"z":rho_z,"p":rho_p},
          "lich_assignments":[{"vms":w,"target":t,"cost":c} for w,j,t,c in assigned if j in lich][:30]}
        print("AIIN_COMP1_RESULT="+json.dumps(res,ensure_ascii=False,separators=(",",":")),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--vms-cache",required=True);p.add_argument("--german-cache",required=True)
    p.add_argument("--seed",type=int,choices=sorted(CHECKPOINTS),required=True)
    Run(p.parse_args()).run()
if __name__=="__main__":main()
