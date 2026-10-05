#!/usr/bin/env python3
"""NK3 rebuild v2 — text/internal family arms."""
import collections, hashlib, json, urllib.request, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
c={"__name__":"nk3rv2_core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
build_rows=c["build_rows"];internal_feature=c["internal_feature"];evaluate_family=c["evaluate_family"];compare_block_effect=c["compare_block_effect"]
safe_form=c["safe_form"];freqbin=c["freqbin"];SEED=c["SEED"]
KGRID=(4,8,12,16,24,32)

def all_types(rows):return sorted(set(r["token"] for r in rows))
def disc_counts(rows):return collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))
def lev(a,b,cap=3):
    if abs(len(a)-len(b))>=cap:return cap
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return min(prev[-1],cap)

def ed_seed_map(rows,max_seeds=24):
    cnt=disc_counts(rows);seeds=[]
    for t,_ in sorted(cnt.items(),key=lambda z:(-z[1],z[0])):
        if all(lev(t,s,3)>2 for s in seeds):
            seeds.append(t)
            if len(seeds)>=max_seeds:break
    other=len(seeds);f={}
    for t in all_types(rows):
        ds=[lev(t,s,3) for s in seeds]
        f[t]=int(np.argmin(ds)) if ds and min(ds)<=2 else other
    return f,{"seeds":seeds,"k":other+1}

def fit_form(rows,k=12,pca=False,ed=None):
    cnt=disc_counts(rows);tr=sorted(cnt);X=np.stack([internal_feature(t) for t in tr])
    eddim=None
    if ed is not None:
        eddim=max(ed.values())+1;E=np.zeros((len(tr),eddim),float)
        for i,t in enumerate(tr):E[i,int(ed[t])]=1.
        X=np.hstack([X,E])
    sc=StandardScaler().fit(X);Z=sc.transform(X);pc=None
    if pca:
        pc=PCA(n_components=min(16,Z.shape[1],max(2,len(tr)-1)),random_state=SEED).fit(Z);Z=pc.transform(Z)
    if k=="auto":
        scores=[]
        for kk in KGRID:
            if len(tr)<=kk*3:continue
            km=KMeans(n_clusters=kk,random_state=SEED,n_init=20,max_iter=500).fit(Z,sample_weight=np.array([cnt[t] for t in tr],float))
            try:s=float(silhouette_score(Z,km.labels_,sample_size=min(2000,len(tr)),random_state=SEED))
            except Exception:s=-9.
            sizes=np.bincount(km.labels_,minlength=kk);tiny=float(np.mean(sizes<5));scores.append((s-.05*tiny,kk,s,tiny))
        scores.sort(reverse=True);k=scores[0][1];sel={"selected":k,"grid":[{"K":q[1],"score":q[0],"silhouette":q[2],"tiny_share":q[3]} for q in scores]}
    else:sel={"selected":int(k)}
    km=KMeans(n_clusters=int(k),random_state=SEED,n_init=30,max_iter=500).fit(Z,sample_weight=np.array([cnt[t] for t in tr],float))
    out={}
    for t in all_types(rows):
        x=internal_feature(t)[None,:]
        if ed is not None:
            E=np.zeros((1,eddim),float);E[0,int(ed.get(t,eddim-1))]=1.;x=np.hstack([x,E])
        z=sc.transform(x);z=pc.transform(z) if pc is not None else z;out[t]=int(km.predict(z)[0])
    return out,sel

def section_only(rows):
    by=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in (2,3):by[r["token"]][r["section"]]+=1
    secs=sorted(set(r["section"] for r in rows if r["fold"] in (2,3)));sm={s:i for i,s in enumerate(secs)};other=len(secs)
    return {t:(sm[by[t].most_common(1)[0][0]] if by.get(t) else other) for t in all_types(rows)},{"sections":secs,"k":other+1}

def random_shape(rows,k=12):
    cnt=disc_counts(rows);out={}
    for t in all_types(rows):
        ps,cs=safe_form(t);shape=f"{cs[0]}|{cs[-1]}|{len(ps)}|{len(t)}|{freqbin(cnt.get(t,0))}"
        h=hashlib.sha256(("NK3RV2|"+shape+"|"+t).encode()).digest();out[t]=int.from_bytes(h[:8],"big")%k
    return out,{"k":k}

def exact_top(rows,n=24):
    cnt=disc_counts(rows);top=[t for t,_ in cnt.most_common(n)];mp={t:i for i,t in enumerate(top)};other=len(top)
    return {t:mp.get(t,other) for t in all_types(rows)},{"top":top,"k":other+1}

TID_ENV=os.getenv("NK3_TID")
TIDS=(TID_ENV,) if TID_ENV else ("ZLZI","ZLZB","TTLI")
OUT={}
for tid in TIDS:
    rows=build_rows(tid);print("NK3RV2_POP",tid,len(rows),collections.Counter(r["fold"] for r in rows),flush=True)
    ed,edi=ed_seed_map(rows);legacy,legi=fit_form(rows,12,False);learned,learni=fit_form(rows,"auto",True);hyb,hybi=fit_form(rows,"auto",True,ed)
    sec,seci=section_only(rows);rnd,rndi=random_shape(rows);exact,exi=exact_top(rows)
    arms={"FORM_LEGACY_K12":(legacy,legi,"candidate_legacy"),"FORM_LEARNED":(learned,learni,"candidate_primary"),
          "FORM_ED_HYBRID":(hyb,hybi,"candidate_primary"),"PURE_ED2":(ed,edi,"hostile_control"),
          "SECTION_ONLY":(sec,seci,"hostile_control"),"RANDOM_SHAPE_HASH12":(rnd,rndi,"hostile_control"),
          "EXACT_TOKEN_TOP24":(exact,exi,"diagnostic_control")}
    rr={}
    for j,(name,(fm,info,role)) in enumerate(arms.items()):
        ev=evaluate_family(rows,fm,nshuffle=250,seed=SEED+1000*j+sum(map(ord,tid)))
        rr[name]={"role":role,"construction":info,"evaluation":ev};print("NK3RV2_ARM",tid,name,json.dumps(ev,separators=(",",":")),flush=True)
    comps={}
    for name in ("FORM_LEGACY_K12","FORM_LEARNED","FORM_ED_HYBRID"):
        comps[name+"_VS_PURE_ED2"]=compare_block_effect(rr[name]["evaluation"],rr["PURE_ED2"]["evaluation"])
        comps[name+"_VS_SECTION_ONLY"]=compare_block_effect(rr[name]["evaluation"],rr["SECTION_ONLY"]["evaluation"])
        comps[name+"_VS_RANDOM"]=compare_block_effect(rr[name]["evaluation"],rr["RANDOM_SHAPE_HASH12"]["evaluation"])
    controls_ok=not rr["SECTION_ONLY"]["evaluation"].get("gate",False) and not rr["RANDOM_SHAPE_HASH12"]["evaluation"].get("gate",False)
    OUT[tid]={"n_rows":len(rows),"arms":rr,"comparisons":comps,"hostile_controls_ok":controls_ok}

summary={}
for name in ("FORM_LEGACY_K12","FORM_LEARNED","FORM_ED_HYBRID","PURE_ED2"):
    by={}
    for tid in OUT:
        e=OUT[tid]["arms"][name]["evaluation"];by[tid]={"gate":e.get("gate"),"gain":e.get("observed_context_gain_bits"),"block_z":e.get("physical_block_z0"),"folds":e.get("fold_gain_bits"),"shuffle_z":e.get("context_shuffle",{}).get("z")}
    vals=[by[t] for t in by]
    summary[name]={"by_transcription":by,
                   "same_direction_all_available":all(v["gain"] is not None and v["gain"]>0 for v in vals),
                   "primary_gate_ZLZI":(bool(by["ZLZI"]["gate"] and OUT["ZLZI"]["hostile_controls_ok"]) if "ZLZI" in by else None)}

out={"phase":"NK3_REBUILD_V2_TEXT","status":"complete","population":"strict +P0","discovery_folds":[2,3],"validation_fold":4,"final_folds":[0,1],
     "primary_context":"neighbor lags -2,-1,+1,+2 FORM/family + line position only",
     "nuisance":"current FORM/spelling + discovery frequency + current section + discovery-only token section-propensity",
     "primary_inference":"heldout gain >0, >2 physical-bifolium SE, positive in both final folds, >2 SD above within-page/line-position context shuffle",
     "results":OUT,"summary":summary}
print("NK3_REBUILD_V2_TEXT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
