#!/usr/bin/env python3
"""
VMS-NEUROCIPHER-GER1 — preregistered direct Voynich -> 15c German screen.

Validated solver is imported unchanged from neurodecipher_acl2019_refactor.py
(commit e36031a9a6b8b67fcebb4d6f4af1c3753fad4287).

Question (narrow):
Does a monotonic character-level word decipherment trained on Voynich exact token
forms generalise to held-out physical folds better against real 15c German word
forms than against a length/character-matched within-word-scrambled German target?

This is NOT a test of every German-under-FORM hypothesis. A negative result rejects
only this direct surface word-transduction architecture.

Firewall:
- ZLZI, strict +P0.
- frozen physical bifolium folds from canonical fold assignment.
- first two tokens of every physical line excluded (LINE_ENTRY firewall).
- discovery = folds 2/3; validation = fold 4; final = folds 0/1.
- discovery lost vocab = top 735 types by folds2/3 frequency.
- target train vocab = top 4103 ReF15 manuscript forms in requested dialect.
- demand schedule and all solver hyperparameters copied from validated Ugaritic/German instrument.
- checkpoint selection = minimum fold4 expected-edit min-cost-flow cost, no final access.
- validation/final use only lost types absent from discovery; strict final also excludes
  every type seen anywhere on validation fold4.
- final target vocab expands to top 10000 forms.
- matched control trains an entirely separate model against deterministic within-word
  scrambles of the same ranked German vocabulary.
- no Voynich decoded-word inspection is emitted in this screen.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, itertools, json, math, random, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np, torch
import neurodecipher_acl2019_refactor as nd

REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
FOLD_SHA="e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888"
FOLD_BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/420d4363973174760a78000f1049339dbd26fa46/research/hf_emergent_occupancy_fold.py"
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112)]
BIF_BY_NUM={n:f'B{a:03d}_{b:03d}' for a,b in PAIRS for n in (a,b)}
DISC_N=735; TRAIN_K=4103; FULL_K=10000; MAX_HELDOUT=735; DISC_DEMAND=221
RATIO=DISC_DEMAND/DISC_N

def canon_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def parse_num(f):
    m=re.match(r"f(\d+)",str(f)); return int(m.group(1)) if m else None
def section(f):
    n=parse_num(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if 85<=n<=102:return "PHARMA"
    if 103<=n<=116:return "RECIPES"
    return "UNK"
def davis_hand(folio,line_no):
    s=str(folio);n=parse_num(s);side='r' if 'r' in s else ('v' if 'v' in s else '');ln=int(line_no or 0)
    if n is None:return 'UNKNOWN'
    if n==115 and side=='r':return 'S2' if ln<=12 else 'S3'
    if 1<=n<=24:return 'S1'
    maps=[{25:'S1',26:'S2',27:'S1',28:'S1',29:'S1',30:'S1',31:'S2',32:'S1'},
          {33:'S2',34:'S2',35:'S1',36:'S1',37:'S1',38:'S1',39:'S2',40:'S2'},
          {41:'S5',42:'S1',43:'S2',44:'S1',45:'S1',46:'S2',47:'S1',48:'S5'},
          {49:'S1',50:'S2',51:'S1',52:'S1',53:'S1',54:'S1',55:'S2',56:'S1'}]
    for mp in maps:
        if n in mp:return mp[n]
    if n==57:return 'S1' if side=='v' else 'S5'
    if n in (58,65):return 'S3'
    if n==66:return 'S5'
    if 67<=n<=73:return 'S4'
    if 75<=n<=84:return 'S2'
    if 85<=n<=86:return 'MIXED_ROSE'
    if 87<=n<=90:return 'S1'
    if n==93:return 'S1'
    if n in (94,95):return 'S3'
    if n==96:return 'S1'
    if 99<=n<=102:return 'S1'
    if 103<=n<=116:return 'S3'
    return 'UNKNOWN'

def build_all_rows(obj):
    rows=[]; eid=0
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            toks=[t.lower() for t in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",t.lower())]
            for pos,t in enumerate(toks):
                rows.append(dict(eid=eid,folio=fol,line=int(ls),pos=pos,line_len=len(toks),
                                 bifolium=BIF_BY_NUM[n],section=section(fol),hand=davis_hand(fol,int(ls)),token=t))
                eid+=1
    return rows
def fold_assignment(rows):
    by=collections.defaultdict(list)
    for r in rows:by[r["bifolium"]].append(r)
    secs=sorted({r["section"] for r in rows});hands=sorted({r["hand"] for r in rows});feats={}
    for bif,rs in by.items():
        cs=collections.Counter(r["section"] for r in rs);ch=collections.Counter(r["hand"] for r in rs)
        feats[bif]=np.array([len(rs)]+[cs[x] for x in secs]+[ch[x] for x in hands],float)
    total=sum(feats.values(),np.zeros(1+len(secs)+len(hands)));target=total/5;scale=np.maximum(target,1)
    loads=[np.zeros_like(target) for _ in range(5)];counts=[0]*5;assign={}
    for bif in sorted(feats,key=lambda z:(-feats[z][0],z)):
        cand=[]
        for f in range(5):
            trial=[x.copy() for x in loads];trial[f]+=feats[bif];cc=counts.copy();cc[f]+=1
            sc=float(sum(np.sum(((x-target)/scale)**2) for x in trial))+0.002*sum(x*x for x in cc)
            cand.append((sc,loads[f][0],counts[f],f))
        f=min(cand)[-1];assign[bif]=f;loads[f]+=feats[bif];counts[f]+=1
    return assign

def load_vms():
    raw=urllib.request.urlopen(V_URL,timeout=120).read()
    if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("V corpus SHA mismatch")
    obj=json.loads(raw)
    # Do not reconstruct the physical split here: import the exact frozen loader.
    ns={"__name__":"vger_frozen_fold"}
    src=urllib.request.urlopen(FOLD_BASE,timeout=120).read().decode()
    exec(compile(src,FOLD_BASE,"exec"),ns)
    _,folds=ns["load_rows"]()
    if ns["canon_sha"](folds)!=FOLD_SHA:raise RuntimeError(("fold sha",ns["canon_sha"](folds)))
    # strict +P0 and LINE_ENTRY firewall
    byfold={i:collections.Counter() for i in range(5)}
    bif_by_type={i:collections.defaultdict(collections.Counter) for i in range(5)}
    excluded_missing_fold=collections.Counter()
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        bif=BIF_BY_NUM[n]
        if bif not in folds:
            for ls,rec in ld.items():
                if str(rec.get("u",""))!="+P0":continue
                toks=[t.lower() for t in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",t.lower())]
                for pos,t in enumerate(toks):
                    if pos>=2:excluded_missing_fold[bif]+=1
            continue
        f=folds[bif]
        for ls,rec in ld.items():
            if str(rec.get("u",""))!="+P0":continue
            toks=[t.lower() for t in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",t.lower())]
            for pos,t in enumerate(toks):
                if pos<2:continue
                byfold[f][t]+=1;bif_by_type[f][t][bif]+=1
    dc=byfold[2]+byfold[3]
    discovery=[w for w,n in dc.most_common(DISC_N)]
    dset=set(discovery)
    val_seen=set(byfold[4])
    validation=[w for w,n in byfold[4].most_common() if w not in dset][:MAX_HELDOUT]
    final_unseen_disc=[w for w,n in (byfold[0]+byfold[1]).most_common() if w not in dset][:MAX_HELDOUT]
    final_strict=[w for w,n in (byfold[0]+byfold[1]).most_common() if w not in dset and w not in val_seen][:MAX_HELDOUT]
    allchars=sorted(set("".join(set().union(*[set(c) for c in byfold.values()]))))
    audit={
      "fold_sha":canon_sha(folds),
      "strict_tokens_after_lineentry":{str(f):int(sum(c.values())) for f,c in byfold.items()},
      "strict_types":{str(f):len(c) for f,c in byfold.items()},
      "discovery_types":len(discovery),"validation_novel_types":len(validation),
      "final_unseen_discovery_types":len(final_unseen_disc),"final_strict_novel_types":len(final_strict),
      "validation_min_count":int(byfold[4][validation[-1]]) if validation else None,
      "final_strict_min_count":int((byfold[0]+byfold[1])[final_strict[-1]]) if final_strict else None,
      "chars":allchars,
      "excluded_unmapped_bifolia":dict(excluded_missing_fold),
      "excluded_unmapped_tokens":int(sum(excluded_missing_fold.values()))}
    return discovery,validation,final_unseen_disc,final_strict,allchars,audit

def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def header(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None);d={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,v=line.split(":",1);d[k.strip().lower()]=v.strip()
    return d
def load_ref_ranked(dialect):
    rb=urllib.request.urlopen(REF_URL,timeout=300).read()
    tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz");freq=collections.Counter();docs=0
    for name in [n for n in tar.getnames() if n.endswith(".xml")]:
        try:root=ET.fromstring(tar.extractfile(name).read())
        except:continue
        md=header(root);med=md.get("medium","").lower();tm=md.get("time","").lower();area=md.get("language-area","").lower()
        if "handschrift" not in med or not tm.startswith("15,"):continue
        bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
        alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
        take=(dialect=="BAV" and bav) or (dialect=="ALEM" and alem)
        if not take:continue
        docs+=1
        for tok in root.iter():
            if tok.tag.split("}")[-1]!="token":continue
            for x in [z for z in tok if z.tag.split("}")[-1] in ("tok_anno","mod")]:
                form=(x.attrib.get("ascii") or x.attrib.get("utf") or x.attrib.get("trans") or "").strip().lower()
                pos=child(x,"pos")
                if not(form and pos and not pos.startswith("$") and all(ch.isalpha() for ch in form)):continue
                freq[form]+=1
    ranked=[w for w,n in freq.most_common()]
    return ranked,freq,docs

def scrambled_unique_vocab(forms,seed):
    """One-to-one within-word anagram control; preserves each word's exact char multiset and length."""
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
            if cand is None:
                # This can occur only for a word whose multiset has no unused alternative.
                # Retain the original rather than altering length or character inventory.
                cand=w
        if cand in seen:
            raise RuntimeError(("scramble uniqueness impossible",w,cand))
        if cand==w:unchanged+=1
        seen.add(cand);out.append(cand)
    return out,{"unchanged":unchanged,"random_tries":random_tries,"enumerated":enumerated,
                "unique":len(seen),"n":len(forms)}

def make_target(ranked,mode,sseed):
    base=ranked[:FULL_K]
    if len(base)<FULL_K:raise RuntimeError(("target vocab too small",len(base)))
    if mode=="real":
        return base[:TRAIN_K],base,{"unchanged":0,"unique":len(base),"n":len(base)}
    out,audit=scrambled_unique_vocab(base,sseed)
    return out[:TRAIN_K],out,audit

DISC,VAL,FIN_UNSEEN,FIN_STRICT,VCHARS,V_AUDIT=load_vms()

class VMSRunner:
    def __init__(self,args):
        self.a=args
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
        self.dev=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        ranked,freq,docs=load_ref_ranked(args.dialect)
        self.train_known,self.full_known,ctrl_audit=make_target(ranked,args.mode,args.scramble_seed)
        self.lcs=nd.Charset(VCHARS);self.kcs=nd.Charset(sorted(set("".join(self.full_known))))
        self.lost=sorted(DISC,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        self.known=list(self.train_known)
        self.lost_ids,self.lost_len=nd.pad_words(self.lost,self.lcs,self.dev)
        self.known_ids,self.known_len=nd.pad_words(self.known,self.kcs,self.dev)
        self.flow=np.zeros((len(self.lost),len(self.known)),np.float32)
        self.model=nd.NeuroCipher(len(self.lcs),len(self.kcs),dropout=.3).to(self.dev)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)
        self.audit={"dialect":args.dialect,"mode":args.mode,"seed":args.seed,"scramble_seed":args.scramble_seed,
                    "ref_docs":docs,"ref_types":len(ranked),"target_train":len(self.train_known),"target_full":len(self.full_known),
                    "target_chars":len(self.kcs)-4,"control":ctrl_audit,"vms":V_AUDIT,
                    "solver_commit":"e36031a9a6b8b67fcebb4d6f4af1c3753fad4287"}
        print("VGER_AUDIT="+json.dumps(self.audit,ensure_ascii=False,separators=(",",":")),flush=True)

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
        print("VGER_ESTEP="+json.dumps({"demand":demand,"edit":edit,"mcf_cost":cost,
              "flow_nonzero":int((nf.sum(1)>0).sum())}),flush=True)

    def train_epoch(self):
        self.model.train()
        last=(0.,0.,0.)
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

    def eval_cost(self,lost_forms,known_forms,label,eval_seed=8081):
        if not lost_forms:return {"n":0,"demand":0,"mean_edit_cost":None}
        lf=sorted(lost_forms,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        kid,klen=nd.pad_words(known_forms,self.kcs,self.dev)
        n=len(lf);k=len(known_forms);costs=np.empty((n,k),np.float32)
        cpu_state=torch.random.get_rng_state()
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(eval_seed)
        self.model.eval()
        bs=self.a.eval_batch
        try:
            for st in range(0,n,bs):
                en=min(st+bs,n);lid,llen=nd.pad_words(lf[st:en],self.lcs,self.dev)
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
        print("VGER_EVAL="+json.dumps(out,separators=(",",":")),flush=True)
        return out

    def run(self):
        if self.a.audit_only:
            print("VGER_AUDIT_ONLY="+json.dumps(self.audit,ensure_ascii=False,separators=(",",":")),flush=True)
            return
        best=None;best_state=None
        for rnd in range(1,self.a.rounds+1):
            if rnd==1:
                self.flow.fill(DISC_DEMAND/self.flow.size)
                print("VGER_ESTEP="+json.dumps({"round":1,"warmup_uniform":True,
                      "flow_total":float(self.flow.sum())}),flush=True)
            else:
                self.e_step(min((rnd-1)*50,DISC_DEMAND),edit=(rnd>self.a.warm_up_steps));self.reset_model()
            for ep in range(1,self.a.epochs+1):
                loss,nll,reg=self.train_epoch();ge=(rnd-1)*self.a.epochs+ep
                if ep%self.a.log_every==0:
                    print("VGER_TRAIN="+json.dumps({"round":rnd,"epoch":ep,"global_epoch":ge,
                          "loss":loss,"nll":nll,"reg":reg}),flush=True)
                if ep%self.a.eval_every==0:
                    v=self.eval_cost(VAL,self.train_known,"validation",eval_seed=8081)
                    meta={"round":rnd,"epoch":ep,"global_epoch":ge,"validation":v}
                    if best is None or v["mean_edit_cost"]<best["validation"]["mean_edit_cost"]:
                        best=meta;best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                        print("VGER_BEST_CHECKPOINT="+json.dumps(best,separators=(",",":")),flush=True)
        self.model.load_state_dict(best_state);self.model.to(self.dev)
        print("VGER_SELECTED_CHECKPOINT="+json.dumps(best,separators=(",",":")),flush=True)
        final_strict=self.eval_cost(FIN_STRICT,self.full_known,"final_strict_novel",eval_seed=9191)
        final_unseen=self.eval_cost(FIN_UNSEEN,self.full_known,"final_unseen_discovery",eval_seed=9292)
        out={"phase":"VMS_NEUROCIPHER_GER1","status":"complete","audit":self.audit,
             "selected_checkpoint":best,"final_strict_novel":final_strict,
             "final_unseen_discovery":final_unseen,
             "interpretation_firewall":"No language inference from this run alone. Compare real target only against independently trained matched scrambled target; screen escalates only on same-sign validation and sealed-final advantage."}
        print("VGER_RESULT_JSON="+json.dumps(out,ensure_ascii=False,separators=(",",":")),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dialect",choices=["BAV","ALEM"],required=True)
    p.add_argument("--mode",choices=["real","scramble"],required=True)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--scramble-seed",type=int,default=9001)
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--warm-up-steps",type=int,default=5)
    p.add_argument("--reg-hyper",type=float,default=.5)
    p.add_argument("--eval-batch",type=int,default=48)
    args=p.parse_args()
    if args.audit_only:
        print("VGER_VMS_AUDIT_ONLY="+json.dumps(V_AUDIT,ensure_ascii=False,separators=(",",":")),flush=True)
        return
    VMSRunner(args).run()
if __name__=="__main__":main()
