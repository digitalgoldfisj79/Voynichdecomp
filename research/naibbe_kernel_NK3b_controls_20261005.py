#!/usr/bin/env python3
"""
NK3b hostile-control completion.
Uses the already frozen NK3 K=12 family model and untouched folds 0/1.
No re-tuning. Adds preregistered controls omitted from NK3a:
- tighter current-shape match includes current FINAL K8 class and raw length
- exact ED-bin balance retained
- page/section/position-preserving context shuffle
- explicit pure-ED context-equivalence benchmark
"""
import urllib.request, json, collections, numpy as np
SRC="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/eb094abf9590499732fdd47b666df17289d0609e/research/naibbe_kernel_NK3_20261005.py"
src=urllib.request.urlopen(SRC,timeout=120).read().decode()
prefix=src.split("# Select K only on ZLZI validation external context.")[0]
ns={"__name__":"nk3_control_base"}
exec(compile(prefix,SRC,"exec"),ns)

# aliases
build_rows=ns["build_rows"]; annotate_recurrence=ns["annotate_recurrence"]; type_stats=ns["type_stats"]
fit_cluster=ns["fit_cluster"]; assign=ns["assign"]; perm_null=ns["perm_null"]; pair_panel=ns["pair_panel"]
safe_form=ns["safe_form"]; context_vec=ns["context_vec"]; freqbin=ns["freqbin"]; plenbin=ns["plenbin"]
rawlenbin=ns["rawlenbin"]; levenshtein=ns["levenshtein"]; edbin=ns["edbin"]; SECS=ns["SECS"]; SECIDX=ns["SECIDX"]
SEED=20261005; KSEL=12

def strict_cell(v):
    return (v["first"],v["final"],v["plen"],v["rlen"],v["dom"],freqbin(v["n"]))

# Override in the base function globals so perm_null/pair_panel use the strict cell.
ns["cell_key"]=strict_cell
perm_null.__globals__["cell_key"]=strict_cell
pair_panel.__globals__["cell_key"]=strict_cell

def posclass(r):
    return 0 if r["pos"]==0 else (2 if r["pos"]==r["line_len"]-1 else 1)

def shuffled_type_stats(rows,seed):
    rng=np.random.default_rng(seed)
    bygrp=collections.defaultdict(list)
    basevec={}
    for i,r in enumerate(rows):
        basevec[i]=context_vec(r)
        bygrp[(r["folio"],r["section"],posclass(r))].append(i)
    shuffled={}
    for g,idx in bygrp.items():
        src=list(idx); rng.shuffle(src)
        for i,j in zip(idx,src): shuffled[i]=basevec[j]
    bytok=collections.defaultdict(list)
    for i,r in enumerate(rows):bytok[r["token"]].append((r,shuffled[i]))
    out={}
    for t,rv in bytok.items():
        sf=safe_form(t)
        if sf is None:continue
        ps,cs=sf
        cv=np.mean(np.stack([v for _,v in rv]),axis=0)
        n=np.linalg.norm(cv);cv=cv/n if n else cv
        sc=np.zeros(len(SECS),float)
        for r,_ in rv:sc[SECIDX[r["section"]]]+=1
        sc/=sc.sum();dom=SECS[int(np.argmax(sc))]
        out[t]=dict(n=len(rv),cv=cv,sec=sc,dom=dom,first=cs[0],final=cs[-1],
                    plen=plenbin(len(ps)),rlen=rawlenbin(len(t)),feat=ns["internal_feature"](t))
    return out

def context_shuffle_null(rows,model,fam_ref,nrep=100):
    vals=[]
    for j in range(nrep):
        st=shuffled_type_stats(rows,SEED+7000+j)
        # family assignment itself does not use context. Recompute only for type overlap consistency.
        ff={t:fam_ref[t] for t in st if t in fam_ref}
        p=pair_panel(st,ff)
        if p["auc"] is not None:vals.append(p["auc"])
    return dict(mean=float(np.mean(vals)) if vals else None,
                sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                n=len(vals))

def ed_only_panel(stats,max_types_cell=80,nperm=300):
    # Explicit control: can coarse edit-distance proximity alone predict external-context similarity
    # after the same strict current-shape/section/frequency match?
    cells=collections.defaultdict(list)
    for t,v in stats.items():
        if v["n"]>=2:cells[strict_cell(v)].append(t)
    pos=[];neg=[]
    for key,ts0 in cells.items():
        ts=sorted(ts0,key=lambda t:(-stats[t]["n"],t))[:max_types_cell]
        for i,a in enumerate(ts):
            for b in ts[i+1:]:
                d=levenshtein(a,b,4)
                sim=float(np.dot(stats[a]["cv"],stats[b]["cv"]))
                if d<=2:pos.append((sim,key))
                elif d>=3:neg.append((sim,key))
    bp=collections.defaultdict(list);bn=collections.defaultdict(list)
    for z in pos:bp[z[1]].append(z[0])
    for z in neg:bn[z[1]].append(z[0])
    y=[];s=[];rng=np.random.default_rng(SEED+8123)
    for k in sorted(set(bp)&set(bn),key=str):
        q=min(len(bp[k]),len(bn[k]),400)
        if q<2:continue
        for ii in rng.choice(len(bp[k]),q,replace=False):y.append(1);s.append(bp[k][int(ii)])
        for ii in rng.choice(len(bn[k]),q,replace=False):y.append(0);s.append(bn[k][int(ii)])
    if len(set(y))<2 or len(y)<40:return dict(auc=None,n=len(y)//2)
    from sklearn.metrics import roc_auc_score
    obs=float(roc_auc_score(y,s))
    null=[]
    yy=np.array(y,int);ss=np.array(s,float)
    for _ in range(nperm):
        yp=rng.permutation(yy);null.append(float(roc_auc_score(yp,ss)))
    mu=float(np.mean(null));sd=float(np.std(null,ddof=1))
    return dict(auc=obs,n=len(y)//2,null_mean=mu,null_sd=sd,z=((obs-mu)/sd if sd else None))

def run_tid(tid):
    rows=build_rows(tid);annotate_recurrence(rows)
    train=type_stats([r for r in rows if r["fold"] in (2,3)])
    model=fit_cluster(train,KSEL)
    out={}
    for name,ffolds in (("fold0",(0,)),("fold1",(1,)),("fold01",(0,1))):
        rr=[r for r in rows if r["fold"] in ffolds]
        st=type_stats(rr);fam=assign(st,model)
        p=perm_null(st,fam,500)
        ed=ed_only_panel(st)
        rec=dict(strict_family=p,pure_ed=ed)
        if name=="fold01":
            sh=context_shuffle_null(rr,model,fam,100)
            rec["page_section_position_context_shuffle"]=sh
            if p["obs"]["auc"] is not None and sh["sd"]:
                rec["family_vs_context_shuffle_z"]=(p["obs"]["auc"]-sh["mean"])/sh["sd"]
        out[name]=rec
    return dict(tid=tid,K=KSEL,results=out)

res={t:run_tid(t) for t in ("ZLZI","ZLZB","TTLI")}
zs=[res[t]["results"]["fold01"]["strict_family"]["z"] for t in ("ZLZI","ZLZB","TTLI")]
z0=res["ZLZI"]["results"]["fold0"]["strict_family"]["z"]
z1=res["ZLZI"]["results"]["fold1"]["strict_family"]["z"]
zc=res["ZLZI"]["results"]["fold01"]["strict_family"]["z"]
gate=bool(zc is not None and zc>2 and z0 is not None and z1 is not None and z0>0 and z1>0 and all(z is not None and z>0 for z in zs))
out={
 "phase":"NK3b","status":"complete","selected_K_frozen":KSEL,
 "controls":{
   "strict_shape":"same current first+final K8, piece length, raw length, dominant section, frequency; exact ED bin balanced",
   "context_shuffle":"shuffle external context among occurrences within same page+section+line-position class",
   "pure_ed":"ED<=2 vs ED>=3 after strict shape/section/frequency matching"
 },
 "results":res,"strict_primary_gate_pass":gate
}
print("NK3B_JSON="+json.dumps(out,separators=(",",":")),flush=True)
