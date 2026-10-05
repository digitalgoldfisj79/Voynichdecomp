#!/usr/bin/env python3
"""
NK3 — real-Voynich homophonic/form-family search.
Prospective design:
  discovery physical folds 2,3: learn FORM-only type embedding + centroids
  validation fold 4: choose K using external-context equivalence
  final untouched folds 0,1: primary test
Family construction never sees external context or exact-token identity as a feature.
External-context test is restricted/matched by coarse current-form, section, frequency, and ED bin.
Replicate ZLZI/ZLZB/TTLI. Stolfi/Mauro closure remains reserved for NK4.
"""
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED=20261005
RNG=np.random.default_rng(SEED)
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"

# Frozen FORM imported, including canonical physical folds.
m={"__name__":"nk3_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),m)
segment=m["segment"]; ST=m["ST"]; folds=m["folds"]
K8_GROUPS=[(0,),(1,),(2,10),(3,7,8),(4,11),(5,),(6,),(9,)]
G8={c:i for i,g in enumerate(K8_GROUPS) for c in g}

PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
       (25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
       (43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
       (71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
       (94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}
SECS=["HERBAL","ASTRO","BIO","PHARMA","RECIPES","UNK"]
SECIDX={s:i for i,s in enumerate(SECS)}

def fnum(f):
    z=re.match(r"f(\d+)",str(f)); return int(z.group(1)) if z else None
def section(f):
    n=fnum(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if 85<=n<=102:return "PHARMA"
    if 103<=n<=116:return "RECIPES"
    return "UNK"
def safe_form(t):
    try:
        ps=segment(t)
        cs=[G8[ST[p]] for p in ps]
        return ps,cs
    except Exception:
        return None
def rawlenbin(n):
    return min(max(n,1),10)-1
def plenbin(n):
    return min(max(n,1),6)-1
def freqbin(n):
    if n<=2:return 0
    if n<=4:return 1
    if n<=8:return 2
    if n<=16:return 3
    return 4
def edbin(d):
    if d<=1:return 1
    if d==2:return 2
    if d==3:return 3
    return 4
def levenshtein(a,b,cap=4):
    if abs(len(a)-len(b))>=cap:return cap
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        rowmin=i
        for j,cb in enumerate(b,1):
            v=min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb))
            cur.append(v); rowmin=min(rowmin,v)
        prev=cur
        if rowmin>=cap:return cap
    return min(prev[-1],cap)

def internal_feature(t):
    sf=safe_form(t)
    if sf is None:return None
    ps,cs=sf
    x=np.zeros(8+8+6+10+8+64+5,float)
    o=0
    x[o+cs[0]]=1;o+=8
    x[o+cs[-1]]=1;o+=8
    x[o+plenbin(len(ps))]=1;o+=6
    x[o+rawlenbin(len(t))]=1;o+=10
    cc=collections.Counter(cs)
    for c,n in cc.items():x[o+c]=n/len(cs)
    o+=8
    if len(cs)>1:
        for a,b in zip(cs[:-1],cs[1:]):x[o+a*8+b]+=1/(len(cs)-1)
    o+=64
    for i,g in enumerate("fkpt"):x[o+i]=float(g in t)
    x[o+4]=sum(t.count(g) for g in "fkpt")/max(1,len(t))
    return x

def load_obj():
    dat=urllib.request.urlopen(CORPUS_URL,timeout=120).read()
    got=hashlib.sha256(dat).hexdigest()
    if got!=CORPUS_SHA:raise RuntimeError(("corpus_sha",got))
    return json.loads(dat)

OBJ=load_obj()

def build_rows(tid):
    rows=[]; eid=0
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF:continue
        bif=BIF[n]
        if bif not in folds:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower()) and safe_form(t.lower()) is not None]
            if not toks:continue
            forms=[safe_form(t) for t in toks]
            for pos,t in enumerate(toks):
                ps,cs=forms[pos]
                pr=forms[pos-1][1] if pos>0 else None
                nx=forms[pos+1][1] if pos+1<len(toks) else None
                rows.append(dict(eid=eid,folio=fol,line=str(ls),pos=pos,line_len=len(toks),bif=bif,fold=int(folds[bif]),
                                 section=section(fol),token=t,ps=ps,cs=cs,
                                 prev_start=(pr[0] if pr else 8),prev_final=(pr[-1] if pr else 8),
                                 next_start=(nx[0] if nx else 8),next_final=(nx[-1] if nx else 8),
                                 prev_len=(min(len(toks[pos-1]),8)-1 if pos>0 else 8),
                                 next_len=(min(len(toks[pos+1]),8)-1 if pos+1<len(toks) else 8)))
                eid+=1
    return rows

def context_vec(r):
    # All coordinates are external to current token identity/form.
    # 4 neighbour FORM marginals (9 each), 2 neighbour length marginals (9 each),
    # line-position (3), local recurrence flags (2).
    x=np.zeros(36+18+3+2,float);o=0
    for v in (r["prev_start"],r["prev_final"],r["next_start"],r["next_final"]):
        x[o+v]=1;o+=9
    for v in (r["prev_len"],r["next_len"]):
        x[o+v]=1;o+=9
    pc=0 if r["pos"]==0 else (2 if r["pos"]==r["line_len"]-1 else 1)
    x[o+pc]=1;o+=3
    # recurrence is external: same surface token immediately before/after
    x[o]=float(r["pos"]>0 and r["token"]==r.get("_prev_token",""));o+=1
    x[o]=float(r["pos"]<r["line_len"]-1 and r["token"]==r.get("_next_token",""))
    return x

def annotate_recurrence(rows):
    by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line"])].append(r)
    for rs in by.values():
        rs.sort(key=lambda z:z["pos"])
        for i,r in enumerate(rs):
            r["_prev_token"]=rs[i-1]["token"] if i else ""
            r["_next_token"]=rs[i+1]["token"] if i+1<len(rs) else ""

def type_stats(rows):
    d={}
    by=collections.defaultdict(list)
    for r in rows:by[r["token"]].append(r)
    for t,rs in by.items():
        sf=safe_form(t)
        if sf is None:continue
        ps,cs=sf
        cv=np.mean(np.stack([context_vec(r) for r in rs]),axis=0)
        # group-normalize each external categorical block, then concatenate
        # means already sum to 1 per group; L2 normalize only for cosine.
        norm=np.linalg.norm(cv);cv=cv/norm if norm else cv
        sc=np.zeros(len(SECS),float)
        for r in rs:sc[SECIDX[r["section"]]]+=1
        sc/=sc.sum()
        dom=SECS[int(np.argmax(sc))]
        d[t]=dict(n=len(rs),cv=cv,sec=sc,dom=dom,first=cs[0],final=cs[-1],
                  plen=plenbin(len(ps)),rlen=rawlenbin(len(t)),feat=internal_feature(t))
    return d

def fit_cluster(train_stats,K):
    types=sorted(train_stats)
    X=np.stack([train_stats[t]["feat"] for t in types])
    w=np.array([train_stats[t]["n"] for t in types],float)
    sc=StandardScaler().fit(X)
    Z=sc.transform(X)
    km=KMeans(n_clusters=K,random_state=SEED,n_init=20,max_iter=500).fit(Z,sample_weight=w)
    return sc,km

def assign(stats,model):
    sc,km=model
    out={}
    for t,v in stats.items():
        out[t]=int(km.predict(sc.transform(v["feat"][None,:]))[0])
    return out

def cell_key(v):
    # Prospective matching: same entry group, piece-count bucket, dominant section, and frequency bin.
    # Final group intentionally not fixed: it is part of complete-form diversity.
    return (v["first"],v["plen"],v["dom"],freqbin(v["n"]))

def pair_panel(stats,fam,max_types_cell=80):
    cells=collections.defaultdict(list)
    for t,v in stats.items():
        if v["n"]>=2 and t in fam:cells[cell_key(v)].append(t)
    pos=[];neg=[];ed_pos=[];ed_neg=[]
    for key,ts0 in cells.items():
        ts=sorted(ts0,key=lambda t:(-stats[t]["n"],t))[:max_types_cell]
        if len(ts)<3:continue
        for i in range(len(ts)):
            a=ts[i]
            for j in range(i+1,len(ts)):
                b=ts[j]
                d=levenshtein(a,b,4); eb=edbin(d)
                sim=float(np.dot(stats[a]["cv"],stats[b]["cv"]))
                rec=(sim,eb,key,a,b)
                if fam[a]==fam[b]:pos.append(rec);ed_pos.append(d)
                else:neg.append(rec);ed_neg.append(d)
    # Exact matching by stratum + ED bin; balance positives/negatives deterministically.
    bp=collections.defaultdict(list);bn=collections.defaultdict(list)
    for z in pos:bp[(z[2],z[1])].append(z)
    for z in neg:bn[(z[2],z[1])].append(z)
    y=[];s=[];usedp=usedn=0
    rng=np.random.default_rng(SEED+len(stats)+len(fam))
    for k in sorted(set(bp)&set(bn),key=str):
        p=bp[k];n=bn[k];q=min(len(p),len(n),500)
        if q<2:continue
        ip=rng.choice(len(p),q,replace=False);inn=rng.choice(len(n),q,replace=False)
        for ii in ip:y.append(1);s.append(p[int(ii)][0]);usedp+=1
        for ii in inn:y.append(0);s.append(n[int(ii)][0]);usedn+=1
    auc=float(roc_auc_score(y,s)) if usedp>=20 and usedn>=20 else None
    same_sims=[z[0] for z in pos];diff_sims=[z[0] for z in neg]
    return dict(auc=auc,n_pos=len(pos),n_neg=len(neg),n_bal_pos=usedp,n_bal_neg=usedn,
                same_mean=float(np.mean(same_sims)) if same_sims else None,
                diff_mean=float(np.mean(diff_sims)) if diff_sims else None,
                delta=(float(np.mean(same_sims)-np.mean(diff_sims)) if same_sims and diff_sims else None),
                median_ed_same=(float(np.median(ed_pos)) if ed_pos else None),
                families=len(set(fam.values())),types=len(stats))

def perm_null(stats,fam,nperm=200):
    obs=pair_panel(stats,fam)
    vals=[]
    # Permute family labels only within the same coarse matching stratum:
    # preserves current-token entry/length/section/frequency structure.
    cells=collections.defaultdict(list)
    for t,v in stats.items():
        if t in fam:cells[cell_key(v)].append(t)
    base=dict(fam)
    rng=np.random.default_rng(SEED+991+len(stats))
    for _ in range(nperm):
        ff=dict(base)
        for ts in cells.values():
            labs=[base[t] for t in ts];rng.shuffle(labs)
            for t,z in zip(ts,labs):ff[t]=z
        p=pair_panel(stats,ff)
        if p["auc"] is not None:vals.append(p["auc"])
    mu=float(np.mean(vals)) if vals else None
    sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    z=((obs["auc"]-mu)/sd if obs["auc"] is not None and sd and sd>0 else None)
    return dict(obs=obs,null_mean=mu,null_sd=sd,z=z,nperm_valid=len(vals))

def exact_token_split_benchmark(rows):
    # Context stability upper/control benchmark: same exact type across two deterministic occurrence halves.
    by=collections.defaultdict(list)
    for r in rows:by[r["token"]].append(r)
    A={};B={}
    for t,rs in by.items():
        if len(rs)<4:continue
        a=rs[::2];b=rs[1::2]
        for dst,xx in ((A,a),(B,b)):
            v=np.mean(np.stack([context_vec(r) for r in xx]),axis=0);n=np.linalg.norm(v);dst[t]=v/n if n else v
    ts=sorted(set(A)&set(B))
    pos=[float(np.dot(A[t],B[t])) for t in ts]
    neg=[]
    for i,t in enumerate(ts):
        if len(ts)>1:neg.append(float(np.dot(A[t],B[ts[(i+137)%len(ts)]])))
    return dict(n=len(ts),same=float(np.mean(pos)) if pos else None,rotated_diff=float(np.mean(neg)) if neg else None,
                delta=(float(np.mean(pos)-np.mean(neg)) if pos and neg else None))

def run_tid(tid,K,final=False):
    rows=build_rows(tid);annotate_recurrence(rows)
    train=type_stats([r for r in rows if r["fold"] in (2,3)])
    val=type_stats([r for r in rows if r["fold"]==4])
    test0=type_stats([r for r in rows if r["fold"]==0])
    test1=type_stats([r for r in rows if r["fold"]==1])
    model=fit_cluster(train,K)
    av=assign(val,model)
    v=perm_null(val,av,100 if not final else 200)
    out=dict(tid=tid,K=K,n_rows=len(rows),n_train_types=len(train),n_val_types=len(val),
             validation=v,exact_val=exact_token_split_benchmark([r for r in rows if r["fold"]==4]))
    if final:
        finals={}
        for name,st,rr in [
            ("fold0",test0,[r for r in rows if r["fold"]==0]),
            ("fold1",test1,[r for r in rows if r["fold"]==1]),
            ("fold01",type_stats([r for r in rows if r["fold"] in (0,1)]),[r for r in rows if r["fold"] in (0,1)])]:
            aa=assign(st,model);finals[name]=perm_null(st,aa,500)
            finals[name]["exact_control"]=exact_token_split_benchmark(rr)
        out["final"]=finals
    return out

# Select K only on ZLZI validation external context.
grid=[]
for K in (8,12,16,24,32):
    r=run_tid("ZLZI",K,False)
    z=r["validation"]["z"]
    grid.append(dict(K=K,z=z,auc=r["validation"]["obs"]["auc"],n_bal=r["validation"]["obs"]["n_bal_pos"],
                     null_mean=r["validation"]["null_mean"],null_sd=r["validation"]["null_sd"]))
usable=[x for x in grid if x["z"] is not None and x["n_bal"]>=40]
if not usable:raise RuntimeError(("NK3 validation insufficient",grid))
best=max(usable,key=lambda x:x["z"])
KSEL=int(best["K"])

# Final ZLZI and transcription replications use same selected K; no further tuning.
results={}
for tid in ("ZLZI","ZLZB","TTLI"):
    results[tid]=run_tid(tid,KSEL,True)

# Primary prospective gate: BOTH untouched physical folds positive and combined >2 null SD,
# plus same direction on all three transcriptions. No claim is made here about Stolfi/Mauro closure.
z0=results["ZLZI"]["final"]["fold0"]["z"]
z1=results["ZLZI"]["final"]["fold1"]["z"]
zc=results["ZLZI"]["final"]["fold01"]["z"]
rep=[results[t]["final"]["fold01"]["z"] for t in ("ZLZI","ZLZB","TTLI")]
primary_pass=bool(zc is not None and zc>2 and z0 is not None and z1 is not None and z0>0 and z1>0 and all(x is not None and x>0 for x in rep))

out={
 "phase":"NK3","status":"complete","seed":SEED,
 "design":{
   "family_definition":"KMeans on frozen internal FORM features only; exact token identity excluded",
   "discovery_folds":[2,3],"validation_fold":4,"final_folds":[0,1],
   "K_grid":[8,12,16,24,32],
   "external_context":"neighbour start/final FORM groups, neighbour lengths, line position, immediate recurrence",
   "pair_matching":"same current entry group + piece-length bucket + dominant section + frequency bin + edit-distance bin",
   "null":"family labels permuted within the same coarse matching strata",
   "closure":"Stolfi/Mauro reserved for NK4"
 },
 "validation_grid":grid,"selected_K":KSEL,"results":results,
 "primary_gate_pass":primary_pass
}
print("NK3_JSON="+json.dumps(out,separators=(",",":")),flush=True)
