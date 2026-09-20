#!/usr/bin/env python3
import collections, hashlib, json, math, re, urllib.request
from pathlib import Path
import numpy as np

VERSION="supergrammar-v03-conditionals-20260920-v1"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
LAYERS=("ZLZI","ZLZB","TTLI","JSLI","VDRB","TTIA")
LAMBDAS=(1.,5.,20.,100.,1000.,1e9)
Q_TARGETS=("k","t")
I_TARGETS=("n","r","l","m","d","$","other")

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
def stable_int(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)
def load():
    p=Path("/tmp/voynich_transcriptions_slim.json");urllib.request.urlretrieve(CORPUS_URL,p)
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=CORPUS_SHA:raise RuntimeError(f"corpus sha mismatch {got}")
    return json.load(open(p))
def tokens(obj,layer):
    out=[]
    for fol,ld in obj["pages"].items():
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get(layer,"")
            for t in txt.split():
                x=t.lower()
                if re.fullmatch(r"[a-z]+",x):
                    out.append((fol,section(fol),x))
    return out
def balanced_folds(events,k,salt):
    # Deterministic greedy folio allocation balancing total event load and section load.
    by=collections.defaultdict(list)
    for e in events:by[e["folio"]].append(e)
    secs=sorted({e["section"] for e in events})
    feat={}
    for fol,es in by.items():
        c=collections.Counter(e["section"] for e in es)
        feat[fol]=np.array([len(es)]+[c[s] for s in secs],float)
    total=sum(feat.values(),np.zeros(1+len(secs)));target=total/k;scale=np.maximum(target,1)
    loads=[np.zeros_like(target) for _ in range(k)];counts=[0]*k;ass={}
    order=sorted(feat,key=lambda f:(-feat[f][0],stable_int(f+"|"+salt)))
    for fol in order:
        cand=[]
        for j in range(k):
            trial=[x.copy() for x in loads];trial[j]+=feat[fol];cc=counts.copy();cc[j]+=1
            score=float(sum(np.sum(((x-target)/scale)**2) for x in trial))+0.002*sum(x*x for x in cc)
            cand.append((score,loads[j][0],counts[j],j))
        j=min(cand)[-1];ass[fol]=j;loads[j]+=feat[fol];counts[j]+=1
    return ass
def q_events(tokrows):
    ev=[]
    for fol,sec,t in tokrows:
        if t.startswith("qok") or t.startswith("qot"):
            ev.append(dict(folio=fol,section=sec,q=1,target=t[2],tail=t[3:]))
        elif t.startswith("ok") or t.startswith("ot"):
            ev.append(dict(folio=fol,section=sec,q=0,target=t[1],tail=t[2:]))
    return ev
def i_events(tokrows):
    ev=[]
    for fol,sec,t in tokrows:
        for m in re.finditer(r"i+",t):
            a,b=m.span();run=b-a
            prev1=t[a-1:a] if a>0 else "^"
            prev2=t[max(0,a-2):a] if a>0 else "^"
            if a==1:prev2="^"+prev2
            nxt=t[b:b+1]
            term=nxt if nxt in ("n","r","l","m","d") else ("$" if b==len(t) else "other")
            ev.append(dict(folio=fol,section=sec,run=("1" if run==1 else ("2" if run==2 else "3+")),
                           prev1=prev1,prev2=prev2,target=term))
    return ev
class Base:
    def __init__(self,train,targets,basekey):
        self.targets=targets;self.basekey=basekey
        self.root=collections.Counter(e["target"] for e in train)
        self.sec=collections.defaultdict(collections.Counter)
        self.ctx=collections.defaultdict(collections.Counter)
        for e in train:
            self.sec[e["section"]][e["target"]]+=1
            self.ctx[basekey(e)][e["target"]]+=1
        self.V=len(targets)
    def rootp(self,y):
        n=sum(self.root.values());return (self.root.get(y,0)+.5)/(n+.5*self.V)
    def secp(self,e,y):
        c=self.sec[e["section"]];n=sum(c.values());return (c.get(y,0)+20*self.rootp(y))/(n+20)
    def prob(self,e,y,lam):
        c=self.ctx[self.basekey(e)];n=sum(c.values());p=self.secp(e,y)
        return (c.get(y,0)+lam*p)/(n+lam)
class Enhanced:
    def __init__(self,train,targets,enhkey,base,base_lam):
        self.enhkey=enhkey;self.base=base;self.base_lam=base_lam
        self.ctx=collections.defaultdict(collections.Counter)
        for e in train:self.ctx[enhkey(e)][e["target"]]+=1
    def prob(self,e,y,lam):
        c=self.ctx[self.enhkey(e)];n=sum(c.values());p=self.base.prob(e,y,self.base_lam)
        return (c.get(y,0)+lam*p)/(n+lam)
def loss(model,rows,lam):
    if not rows:return float("inf")
    return -sum(math.log2(max(model.prob(e,e["target"],lam),1e-300)) for e in rows)/len(rows)
def tune_base(train,targets,basekey,outer,salt):
    ass=balanced_folds(train,4,f"{salt}|inner|{outer}|base")
    best=None
    for lam in LAMBDAS:
        ls=[]
        for f in range(4):
            tr=[e for e in train if ass[e["folio"]]!=f];va=[e for e in train if ass[e["folio"]]==f]
            if not va:continue
            m=Base(tr,targets,basekey);ls.append(loss(m,va,lam))
        x=float(np.mean(ls))
        if best is None or (x,lam)<best:best=(x,lam)
    return best[1]
def tune_enh(train,targets,basekey,enhkey,outer,salt,base_lam):
    ass=balanced_folds(train,4,f"{salt}|inner|{outer}|enh")
    best=None
    for lam in LAMBDAS:
        ls=[]
        for f in range(4):
            tr=[e for e in train if ass[e["folio"]]!=f];va=[e for e in train if ass[e["folio"]]==f]
            if not va:continue
            b=Base(tr,targets,basekey);m=Enhanced(tr,targets,enhkey,b,base_lam);ls.append(loss(m,va,lam))
        x=float(np.mean(ls))
        if best is None or (x,lam)<best:best=(x,lam)
    return best[1]
def signflip_sums(blocks,total_n):
    vals=np.array(list(blocks.values()),float)
    eff=float(vals.sum()/total_n);sd=float(np.sqrt(np.sum(vals*vals))/total_n)
    return dict(effect=eff,null_sd=sd,effect_over_null_sd=(abs(eff)/sd if sd else None),
                n_blocks=len(vals),positive_blocks=int((vals>0).sum()),negative_blocks=int((vals<0).sum()))
def evaluate(events,targets,basekey,enhkey,salt):
    outer=balanced_folds(events,5,f"{salt}|outer")
    blocks=collections.Counter();folds=[]
    for f in range(5):
        tr=[e for e in events if outer[e["folio"]]!=f];te=[e for e in events if outer[e["folio"]]==f]
        lb=tune_base(tr,targets,basekey,f,salt);le=tune_enh(tr,targets,basekey,enhkey,f,salt,lb)
        b=Base(tr,targets,basekey);m=Enhanced(tr,targets,enhkey,b,lb)
        sg=0.0
        for e in te:
            pb=max(b.prob(e,e["target"],lb),1e-300);pe=max(m.prob(e,e["target"],le),1e-300)
            g=math.log2(pe/pb);blocks[e["folio"]]+=g;sg+=g
        folds.append(dict(fold=f,n_test=len(te),lambda_base=lb,lambda_enh=le,gain_bits_per_event=sg/len(te)))
    st=signflip_sums(blocks,len(events));st["folds"]=folds;st["n_events"]=len(events);st["n_folios"]=len({e["folio"] for e in events})
    return st
def main():
    obj=load();out=dict(version=VERSION,corpus_sha256=CORPUS_SHA,layers={})
    for layer in LAYERS:
        tr=tokens(obj,layer);q=q_events(tr);ie=i_events(tr)
        q1=evaluate(q,Q_TARGETS,lambda e:(e["section"],),lambda e:(e["section"],e["q"]),f"{layer}|q1")
        q2=evaluate(q,Q_TARGETS,lambda e:(e["section"],e["tail"]),lambda e:(e["section"],e["tail"],e["q"]),f"{layer}|q2")
        i1=evaluate(ie,I_TARGETS,lambda e:(e["section"],e["prev1"]),lambda e:(e["section"],e["prev1"],e["run"]),f"{layer}|i1")
        i2=evaluate(ie,I_TARGETS,lambda e:(e["section"],e["prev2"]),lambda e:(e["section"],e["prev2"],e["run"]),f"{layer}|i2")
        out["layers"][layer]=dict(q1=q1,q2=q2,i1=i1,i2=i2)
    def passed(x):return x["effect"]>0 and x["effect_over_null_sd"]>=2
    alt=[x for x in LAYERS if x!="ZLZI"]
    out["decisions"]=dict(
      Q_PRIMARY_PASS=passed(out["layers"]["ZLZI"]["q2"]),
      Q_ALT_RESOLVED=sum(passed(out["layers"][x]["q2"]) for x in alt),
      Q_ALT_NEGATIVE=sum(out["layers"][x]["q2"]["effect"]<0 and out["layers"][x]["q2"]["effect_over_null_sd"]>=2 for x in alt),
      I_PRIMARY_PASS=passed(out["layers"]["ZLZI"]["i1"]) and passed(out["layers"]["ZLZI"]["i2"]),
      I_ALT_BOTH_RESOLVED=sum(passed(out["layers"][x]["i1"]) and passed(out["layers"][x]["i2"]) for x in alt),
      I_ALT_NEGATIVE=sum((out["layers"][x]["i1"]["effect"]<0 and out["layers"][x]["i1"]["effect_over_null_sd"]>=2) or
                         (out["layers"][x]["i2"]["effect"]<0 and out["layers"][x]["i2"]["effect_over_null_sd"]>=2) for x in alt))
    payload=json.dumps(out,sort_keys=True,separators=(",",":"));out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("SGV03_CONDITIONALS="+json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
