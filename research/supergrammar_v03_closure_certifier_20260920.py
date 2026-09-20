#!/usr/bin/env python3
import argparse, collections, hashlib, json, math, os, re, urllib.request
from pathlib import Path
import numpy as np

VERSION="supergrammar-v03-closure-certifier-20260920-v1"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
ROWS_CANON_SHA="74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f"
FOLDS_CANON_SHA="e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888"
LAYERS=("ZLZI","ZLZB","TTLI","JSLI","VDRB","TTIA")
LAMBDAS=(16.,64.,256.,1024.,4096.,16384.,1e9)
PARENT_ALPHA=64.0
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
       (25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
       (43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
       (71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
       (94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF_BY_NUM={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def canon_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
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
    s=str(folio);n=parse_num(s);side="r" if "r" in s else ("v" if "v" in s else "");ln=int(line_no or 0)
    if n is None:return "UNKNOWN"
    if n==115 and side=="r":return "S2" if ln<=12 else "S3"
    if 1<=n<=24:return "S1"
    maps=[{25:"S1",26:"S2",27:"S1",28:"S1",29:"S1",30:"S1",31:"S2",32:"S1"},
          {33:"S2",34:"S2",35:"S1",36:"S1",37:"S1",38:"S1",39:"S2",40:"S2"},
          {41:"S5",42:"S1",43:"S2",44:"S1",45:"S1",46:"S2",47:"S1",48:"S5"},
          {49:"S1",50:"S2",51:"S1",52:"S1",53:"S1",54:"S1",55:"S2",56:"S1"}]
    for mp in maps:
        if n in mp:return mp[n]
    if n==57:return "S1" if side=="v" else "S5"
    if n in (58,65):return "S3"
    if n==66:return "S5"
    if 67<=n<=73:return "S4"
    if 75<=n<=84:return "S2"
    if 85<=n<=86:return "MIXED_ROSE"
    if 87<=n<=90:return "S1"
    if n==93:return "S1"
    if n in (94,95):return "S3"
    if n==96:return "S1"
    if 99<=n<=102:return "S1"
    if 103<=n<=116:return "S3"
    return "UNKNOWN"
def lenbin(t):
    n=len(t);return "12" if n<=2 else ("34" if n<=4 else ("56" if n<=6 else "7p"))
def family(t):return f"{t[0]}|{t[-1]}|{lenbin(t)}"
def build_events(obj):
    rows=[];eid=0
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get("ZLZI","")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            for pos,t in enumerate(toks):
                rows.append(dict(eid=eid,folio=fol,line=int(re.match(r"\d+",str(ls)).group()),pos=pos,line_len=len(toks),
                                 page=fol,bifolium=BIF_BY_NUM[n],section=section(fol),hand=davis_hand(fol,int(re.match(r"\d+",str(ls)).group())),
                                 token=t,family=family(t)));eid+=1
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
def load_obj_rows_folds():
    p=Path("/tmp/voynich_transcriptions_slim.json");urllib.request.urlretrieve(CORPUS_URL,p)
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=CORPUS_SHA:raise RuntimeError("corpus sha mismatch")
    obj=json.load(open(p));rows=build_events(obj);folds=fold_assignment(rows)
    if len(rows)!=34087 or canon_sha(rows)!=ROWS_CANON_SHA or canon_sha(folds)!=FOLDS_CANON_SHA:raise RuntimeError("canonical source mismatch")
    return obj,rows,folds
def build_transitions(obj,layer):
    bypage=collections.OrderedDict()
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        lines=[]
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get(layer,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if not toks:continue
            s=str(ls);m=re.match(r"(\d+)",s);ln=int(m.group(1)) if m else 0
            lines.append(((ln,s),toks))
        if lines:bypage[fol]=sorted(lines,key=lambda x:x[0])
    rows=[]
    for fol,lines in bypage.items():
        prev=None
        for li,(_,toks) in enumerate(lines):
            for j,t in enumerate(toks):
                if prev is None:boundary="PAGE_START"
                elif j==0:boundary="LINE_BREAK"
                else:boundary="SPACE"
                rows.append(dict(folio=fol,bifolium=BIF_BY_NUM[parse_num(fol)],section=section(fol),boundary=boundary,prev=prev,target=t))
                prev=t
    return rows
def morph(prev):
    if prev is None:return ("<START>","<START>",0)
    return (prev[:1],prev[-2:],len(prev))
def inner_fold(bif,outer):
    h=int(hashlib.sha256((bif+"|closure-v03|"+str(outer)).encode()).hexdigest()[:8],16)
    return h%4
class Parent:
    def __init__(self,train):
        self.globalc=collections.Counter(r["target"] for r in train)
        self.ctx=collections.defaultdict(collections.Counter)
        for r in train:self.ctx[(r["section"],r["boundary"])][r["target"]]+=1
        self.vocab=set(self.globalc);self.V=len(self.vocab)+1;self.N=sum(self.globalc.values())
    def gp(self,y):
        c=self.globalc.get(y,0);return (c+0.5)/(self.N+0.5*self.V)
    def prob(self,r,y):
        c=self.ctx[(r["section"],r["boundary"])];n=sum(c.values());return (c.get(y,0)+PARENT_ALPHA*self.gp(y))/(n+PARENT_ALPHA)
class Child:
    def __init__(self,train,keyfun,parent,lam):
        self.parent=parent;self.lam=lam;self.c=collections.defaultdict(collections.Counter)
        for r in train:self.c[(r["section"],r["boundary"],keyfun(r["prev"]))][r["target"]]+=1
        self.keyfun=keyfun
    def prob(self,r,y):
        pc=self.parent.prob(r,y);c=self.c[(r["section"],r["boundary"],self.keyfun(r["prev"]))];n=sum(c.values())
        return (c.get(y,0)+self.lam*pc)/(n+self.lam)
def mean_loss(model,rows):
    if not rows:return float("inf")
    s=0.0
    for r in rows:s-=math.log2(max(model.prob(r,r["target"]),1e-300))
    return s/len(rows)
def tune_child(train,keyfun,parent_builder,outer,base_mode):
    best=None
    for lam in LAMBDAS:
        losses=[]
        for inf in range(4):
            tr=[r for r in train if inner_fold(r["bifolium"],outer)!=inf]
            va=[r for r in train if inner_fold(r["bifolium"],outer)==inf]
            p=parent_builder(tr,base_mode,outer)
            c=Child(tr,keyfun,p,lam)
            losses.append(mean_loss(c,va))
        x=float(np.mean(losses))
        if best is None or (x,lam)<(best[0],best[1]):best=(x,lam)
    return best[1]
def parent_builder(train,mode,outer):
    p=Parent(train)
    if mode=="PARENT":return p
    if mode=="MORPH":
        lam=tune_child(train,morph,lambda tr,_,__:Parent(tr),outer,"PARENT")
        return Child(train,morph,p,lam)
    raise ValueError(mode)
def signflip(vals):
    a=np.asarray(vals,float)
    eff=float(a.mean());sd=float(np.sqrt(np.sum(a*a))/len(a));ratio=float(abs(eff)/sd) if sd else None
    return dict(n_blocks=len(a),effect=eff,null_sd=sd,effect_over_null_sd=ratio,positive=int((a>0).sum()),negative=int((a<0).sum()))
def evaluate_layer(rows,folds):
    morph_block=[];exact_block=[];exact_lams=[]
    fold_rows=[]
    for outer in range(5):
        tr=[r for r in rows if folds[r["bifolium"]]!=outer]
        te=[r for r in rows if folds[r["bifolium"]]==outer]
        p=Parent(tr)
        lm=tune_child(tr,morph,lambda x,_,__:Parent(x),outer,"PARENT")
        m=Child(tr,morph,p,lm)
        def exactkey(prev):return prev if prev is not None else "<START>"
        le=tune_child(tr,exactkey,lambda x,_,__:parent_builder(x,"MORPH",outer),outer,"MORPH")
        e=Child(tr,exactkey,m,le)
        by=collections.defaultdict(lambda:[0.0,0.0,0])
        for r in te:
            pp=max(p.prob(r,r["target"]),1e-300);pm=max(m.prob(r,r["target"]),1e-300);pe=max(e.prob(r,r["target"]),1e-300)
            z=by[r["bifolium"]];z[0]+=math.log2(pm/pp);z[1]+=math.log2(pe/pm);z[2]+=1
        for bif,(gm,ge,n) in by.items():
            morph_block.append(gm/n);exact_block.append(ge/n)
        exact_lams.append(le)
        fold_rows.append(dict(fold=outer,n_test=len(te),lambda_morph=lm,lambda_exact=le,morph_gain_bits=float(np.mean([v[0]/v[2] for v in by.values()])),exact_gain_bits=float(np.mean([v[1]/v[2] for v in by.values()]))))
    return dict(morph=signflip(morph_block),exact=signflip(exact_block),exact_lambdas=exact_lams,
                exact_off_folds=sum(x>=16384 for x in exact_lams),folds=fold_rows)
def main():
    obj,canon,folds=load_obj_rows_folds()
    out=dict(version=VERSION,corpus_sha256=CORPUS_SHA,rows_sha256=ROWS_CANON_SHA,folds_sha256=FOLDS_CANON_SHA,layers={})
    for layer in LAYERS:
        rows=build_transitions(obj,layer);out["layers"][layer]=dict(n=len(rows),**evaluate_layer(rows,folds))
    alts=[k for k in LAYERS if k!="ZLZI"]
    morph_ok=out["layers"]["ZLZI"]["morph"]["effect"]>0 and out["layers"]["ZLZI"]["morph"]["effect_over_null_sd"]>=2
    alt_pos=sum(out["layers"][x]["morph"]["effect"]>0 and out["layers"][x]["morph"]["effect_over_null_sd"]>=2 for x in alts)
    alt_neg=sum(out["layers"][x]["morph"]["effect"]<0 and out["layers"][x]["morph"]["effect_over_null_sd"]>=2 for x in alts)
    exact_layer_closed=[]
    for x in LAYERS:
        e=out["layers"][x]["exact"];g=e["effect"];r=e["effect_over_null_sd"];offs=out["layers"][x]["exact_off_folds"]
        exact_layer_closed.append((g<=0.001 or r<2) and offs>=4)
    out["decisions"]=dict(
      MORPH_REPRESENTATION_ROBUST=bool(morph_ok and alt_pos>=4 and alt_neg==0),
      morph_alt_resolved_positive=alt_pos,morph_alt_resolved_negative=alt_neg,
      EXACT_IDENTITY_CLOSED=bool(all(exact_layer_closed)),
      exact_layers_closed=sum(exact_layer_closed))
    payload=json.dumps(out,sort_keys=True,separators=(",",":"));out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("SGV03_CLOSURE="+json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
