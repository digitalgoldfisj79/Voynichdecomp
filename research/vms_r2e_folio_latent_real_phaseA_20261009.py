#!/usr/bin/env python3
# VMS-R2E real Recipes Phase A. Preregistered before real outcome on 2026-10-09.
import copy,json,math,sys,urllib.request
import numpy as np

INST_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d9cc8f34b208ea20fe290e73bbb0d40c3aae6709/research/vms_r2e_folio_latent_instrument_20261009.py"
CAL_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/36740cd684ed22dba02acf33f3bb1109c19bfc54/research/data/residrec4b_residual15_calibration_20261008.json"
PARENT_D2=161.34337398354887
EPS=1e-15

# Import the exact qualified R2E definitions but stop before synthetic execution.
src=urllib.request.urlopen(INST_URL,timeout=120).read().decode()
prefix=src.split("\nALL_SEEDS=")[0]
_old=list(sys.argv)
sys.argv=["r2e_defs","--mode","cal","--start","0","--count","1"]
ns={"__name__":"r2e_defs"}
exec(compile(prefix,INST_URL,"exec"),ns)
sys.argv=_old

lines=ns["lines"]
fit_mix=ns["fit_mix"]
tilt_probs=ns["tilt_probs"]
folios_from_lines=ns["folios_from_lines"]
S=ns["S"]
core=ns["core"]
r4=core["r1"]["r4"]

CAL=json.loads(urllib.request.urlopen(CAL_URL,timeout=120).read().decode())
MU=np.asarray(CAL["mean"],float)
IV=np.asarray(CAL["inv_cov"],float)
SM=float(CAL["scale_mean"])
SS=float(CAL["scale_sd"])

def bits_parent(ls):
    b=0.; n=0
    for s in ls:
        for e in s:
            y=int(e["y"]); p=np.asarray(e["p"],float)
            b += -math.log2(max(float(p[y]),EPS)); n+=1
    return b,n

def predict_lines(ls,md):
    by={}
    for s in ls:
        if not s: continue
        fol=str(s[0]["folio"])
        by.setdefault(fol,[]).append(s)
    out=[]; bits=0.; n=0
    for fol in sorted(by):
        post=np.asarray(md["pi"],float).copy()
        for s in by[fol]:
            zz=[]
            for e in s:
                p=np.asarray(e["p"],float); y=int(e["y"])
                Q=np.vstack([tilt_probs(p[None,:],md["b"][st])[0] for st in range(S)])
                pred=post@Q
                bits += -math.log2(max(float(pred[y]),EPS)); n+=1
                post=post*Q[:,y]
                post/=post.sum()
                x=dict(e); x["p"]=pred; zz.append(x)
            out.append(zz)
    return out,bits,n

def destroy_boundaries(ls,seed):
    # Within each physical outer fold, permute complete exact-entry blocks among
    # folio containers while preserving each folio's entry-count profile.
    rng=np.random.default_rng(seed)
    out=[]
    foldvals=sorted({int(s[0]["fold"]) for s in ls if s})
    for fv in foldvals:
        ss=[s for s in ls if s and int(s[0]["fold"])==fv]
        entries={}
        entry_folio={}
        entry_order={}
        for s in ss:
            eid=int(s[0]["eid"])
            entries.setdefault(eid,[]).append(s)
            entry_folio[eid]=str(s[0]["folio"])
            entry_order[eid]=min(entry_order.get(eid,10**9),int(s[0]["entry_line_index"]))
        for eid in entries:
            entries[eid]=sorted(entries[eid],key=lambda s:int(s[0]["entry_line_index"]))
        containers=sorted({str(s[0]["folio"]) for s in ss})
        counts={f:len({int(s[0]["eid"]) for s in ss if str(s[0]["folio"])==f}) for f in containers}
        eids=sorted(entries)
        perm=[eids[i] for i in rng.permutation(len(eids))]
        k=0
        for fol in containers:
            assigned=perm[k:k+counts[fol]]; k+=counts[fol]
            for slot,eid in enumerate(assigned):
                for s in entries[eid]:
                    zz=[]
                    for e in s:
                        x=dict(e)
                        x["_r2e_original_folio"]=str(e["folio"])
                        x["folio"]=fol
                        x["_r2e_container_slot"]=slot
                        zz.append(x)
                    out.append(zz)
        if k!=len(eids):
            raise RuntimeError(("DESTROY_ASSIGN",fv,k,len(eids)))
    # Construction is already fold -> folio -> entry-slot -> within-entry-line.
    return out

def d2(vec):
    z=np.asarray(vec,float)-MU
    return float(z@IV@z)

by={j:[s for s in lines if int(s[0]["fold"])==j] for j in range(5)}
pooled_parent=pooled_real=pooled_control=0.; nn=0
folds=[]; allreal=[]
nof115_parent=nof115_real=0.; nof115_n=0

for j in range(5):
    v=(j+1)%5
    trks=[k for k in range(5) if k not in (j,v)]
    tr=[s for k in trks for s in by[k]]
    te=by[j]

    md=fit_mix(tr,202610095400+j*1000)
    ore,br,nr=predict_lines(te,md)
    bp,np0=bits_parent(te)
    if nr!=np0: raise RuntimeError(("REAL_N",j,nr,np0))

    dtr=destroy_boundaries(tr,202610095500+j)
    dte=destroy_boundaries(te,202610095500+j)
    mdc=fit_mix(dtr,202610095600+j*1000)
    _,bc,nc=predict_lines(dte,mdc)
    if nc!=np0: raise RuntimeError(("CTRL_N",j,nc,np0))

    pooled_parent+=bp; pooled_real+=br; pooled_control+=bc; nn+=np0
    allreal.extend(ore)

    keep=[s for s in te if str(s[0]["folio"])!="f115r"]
    if keep:
        b0,n0=bits_parent(keep)
        _,b1,n1=predict_lines(keep,md)
        if n0!=n1: raise RuntimeError(("F115_N",j,n0,n1))
        nof115_parent+=b0; nof115_real+=b1; nof115_n+=n0

    folds.append({
      "fold":j,"n":np0,
      "parent_bpe":bp/np0,"real_bpe":br/np0,"control_bpe":bc/np0,
      "gain_vs_parent":(bp-br)/np0,
      "real_minus_control":(bc-br)/np0,
      "pi":np.asarray(md["pi"],float).tolist(),
      "iters":int(md["iters"])
    })

vec,names,nm=r4["metric_vector"](allreal)
if nm!=nn: raise RuntimeError(("METRIC_N",nm,nn))
D=d2(vec); ZD=(D-SM)/SS; red=1-D/PARENT_D2
pb=pooled_parent/nn; rb=pooled_real/nn; cb=pooled_control/nn
pos=sum(x["gain_vs_parent"]>0 for x in folds)
passed=bool(rb<pb and pos>=4 and rb<cb and red>=0.20)

out={
 "programme":"VMS-R2E-REAL-PHASEA",
 "status":"complete","n":nn,
 "parent_bpe":pb,"real_bpe":rb,"control_bpe":cb,
 "gain_vs_parent":pb-rb,
 "real_minus_control":cb-rb,
 "positive_folds":pos,
 "D2":D,"Z_D2":ZD,"D2_reduction_fraction":red,
 "parent_D2":PARENT_D2,
 "vector":np.asarray(vec,float).tolist(),"metric_names":names,
 "f115r_exclusion":{
   "n":nof115_n,
   "parent_bpe":nof115_parent/max(nof115_n,1),
   "real_bpe":nof115_real/max(nof115_n,1),
   "gain_vs_parent":(nof115_parent-nof115_real)/max(nof115_n,1)
 },
 "phaseA_pass":passed,
 "decision":"R2E_REAL_PHASEA_CANDIDATE" if passed else "R2E_REAL_PHASEA_FAIL",
 "folds":folds
}
print("R2E_REAL_RESULT="+json.dumps(out,separators=(",",":")),flush=True)
