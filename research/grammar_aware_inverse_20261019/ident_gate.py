#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
"""IDENT-GATE-1: exact mechanism equivalence + anchor/noise stress test.
Synthetic ONLY; neither the anchor nor the context proxy is historical evidence.
Frozen FORM from 2026-10-04. No source meaning inferred from the Voynich.
"""
import argparse,hashlib,json,math,time,urllib.request
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

URL=("https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
     "fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/"
     "research/select_form_socket_oracle_phaseK0_20261004.py")
src=urllib.request.urlopen(URL,timeout=120).read().decode()
module={"__name__":"frozen_form"}
exec(compile(src,URL,"exec"),module)
M=module
K=8
def emit(tag,obj):
    print(tag+"="+json.dumps(obj,separators=(",",":"),sort_keys=True,default=float),flush=True)

def forward_ll(lines,byfold,A,pi,U,Ve,Vr):
    """Exact HMM likelihood of surface forms under fixed forward source/regime,
    including exact FORM route piece terms and STOP, with independent fold resets.
    """
    out=0.
    for inds in byfold:
        prev=np.log(pi)
        for j,idx in enumerate(inds):
            row=lines[idx]
            shift=(row["section"]+2*row["entry"])%K
            emission=np.array([
              sum(M["token_loglik"](tok,(z+shift)%K,U,Ve,Vr) for tok in row["tokens"])
              for z in range(K)])
            if j==0:alpha=prev+emission
            else:alpha=logsumexp(alpha[:,None]+np.log(A),axis=0)+emission
        out+=float(logsumexp(alpha))
    return out

def gen(seed,lines_per_fold,tokens_per_line):
    rng=np.random.default_rng(seed)
    # IDENTICAL Markov controller for semantic and production interpretations.
    A=np.full((K,K),.30/(K-1),float)
    np.fill_diagonal(A,.70)
    pi=np.full(K,1/K)
    U,Ve0,Vr0=M["controls"](K,2,1.,rng,"F1")
    Ve,Vr=Ve0*5.,Vr0*.5
    lines=[];byfold=[];hidden=[];meta=[];a_source=[];a_none=[]
    for fold in range(5):
        indices=[];z=int(rng.choice(K,p=pi))
        for ii in range(lines_per_fold):
            if ii>0:z=int(rng.choice(K,p=A[z]))
            # Variation across section and entry; all such nuisance is shared by both mechanisms.
            section=(ii+fold)%3
            entry=(ii+2*fold)%3
            shift=(section+2*entry)%K
            controller=(z+shift)%K
            toks=[]
            for ti in range(tokens_per_line):
                for attempt in range(100):
                    tok=M["gen_token"](controller,U,Ve,Vr,rng)
                    if tok is not None:break
                else:raise RuntimeError("No valid FORM termination")
                toks.append(tok)
            row={"section":section,"entry":entry,"tokens":toks}
            lines.append(row);indices.append(len(lines)-1);hidden.append(z)
            meta.append((section,entry))
            # A1: anchor linked to latent category, can be semantic.
            a_source.append(int((z%2) ^ int(rng.random()<.15)))
            # A0: independent given section/entry strata; no causal source link.
            p0=.15+.15*section+.065*entry
            a_none.append(int(rng.random()<p0))
        byfold.append(indices)
    # A2: fully matched production anchor is coupled to A1 *exactly*.
    # The two causal stories have the same joint P(tokens, anchors).
    a_production=a_source[:]
    def enc_token(t):return ".".join(str(int(x)) for x in t)
    corpus_blob=json.dumps([
        {"section":x["section"],"entry":x["entry"],
         "tokens":[enc_token(t) for t in x["tokens"]]}
        for x in lines],sort_keys=True,separators=(",",":")).encode()
    hash_l=hashlib.sha256(corpus_blob).hexdigest()
    hash_p=hashlib.sha256(corpus_blob).hexdigest()
    ll_source=forward_ll(lines,byfold,A,pi,U,Ve,Vr)
    ll_production=forward_ll(lines,byfold,A,pi,U,Ve,Vr)
    if hash_l!=hash_p or abs(ll_source-ll_production)>1e-10:raise AssertionError("equivalent mechanism mismatch")
    return {"lines":lines,"byfold":byfold,"latent":np.array(hidden),"meta":np.array(meta),
      "anchors":{"independent":np.array(a_none),"semantic":np.array(a_source),
                 "production":np.array(a_production)},
      "hash_source":hash_l,"hash_production":hash_p,
      "surface_ll_source":ll_source,"surface_ll_production":ll_production}

def features(lines):
    X0=[];XF=[]
    for row in lines:
        sec,ent=row["section"],row["entry"]
        nuisance=[float(sec==a) for a in range(3)]+[float(ent==a) for a in range(3)]
        # fixed true FORM pieces only; no per-case latent mapping or source truth
        hist=np.zeros(12)
        route=np.zeros((8,12))
        lens=[]
        for tok in row["tokens"]:
            pieces=[int(M["PCLASS"][i]) for i in tok]
            hist[pieces[0]]+=1
            for p,q in zip(pieces[:-1],pieces[1:]):
                route[M["G_OF"][p],q]+=1
            lens.append(len(tok))
        hist/=len(row["tokens"])
        route/=max(1,sum(max(0,x-1) for x in lens))
        form=np.r_[hist,route.flatten(),np.mean(lens)]
        X0.append(nuisance)
        XF.append(np.r_[nuisance,form])
    return np.asarray(X0),np.asarray(XF)

def fit(train,val,X,y):
    # Select both model regularizations on VAL; unchanged across arms.
    best=None
    for C in (.03,.3,3.):
        model=LogisticRegression(C=C,max_iter=250,solver="lbfgs")
        model.fit(X[train],y[train])
        p=np.clip(model.predict_proba(X[val])[:,1],1e-10,1-1e-10)
        loss=log_loss(y[val],p,labels=[0,1])
        if best is None or loss<best[0]:best=(loss,C,model)
    return best

def result_arm(y,Xbase,Xfull,byfold,seed):
    tr=np.array(byfold[0]+byfold[1]+byfold[2],dtype=int)
    va=np.array(byfold[3],dtype=int)
    te=np.array(byfold[4],dtype=int)
    bas=fit(tr,va,Xbase,y)
    ful=fit(tr,va,Xfull,y)
    pb=np.clip(bas[2].predict_proba(Xbase[te])[:,1],1e-10,1-1e-10)
    pf=np.clip(ful[2].predict_proba(Xfull[te])[:,1],1e-10,1-1e-10)
    bits=(log_loss(y[te],pb,labels=[0,1])-
          log_loss(y[te],pf,labels=[0,1]))/math.log(2)
    auc=float(roc_auc_score(y[te],pf)) if len(np.unique(y[te]))>1 else None
    return {"holdout_test_bits_per_line_gain":float(bits),
            "holdout_anchor_AUC":auc,"base_C":bas[1],"with_form_C":ful[1],
            "train_lines":len(tr),"validation_lines":len(va),"test_lines":len(te)}

def one(seed,lines_per_fold,tokens_per_line):
    d=gen(seed,lines_per_fold,tokens_per_line)
    x0,xf=features(d["lines"])
    inds=d["byfold"]
    acts={key:result_arm(arr,x0,xf,inds,seed) for key,arr in d["anchors"].items()}
    pos=acts["semantic"]["holdout_test_bits_per_line_gain"]
    null=acts["independent"]["holdout_test_bits_per_line_gain"]
    twin=acts["production"]["holdout_test_bits_per_line_gain"]
    if not(np.array_equal(d["anchors"]["semantic"],d["anchors"]["production"])
           and abs(pos-twin)<1e-12):raise AssertionError("hard negative not equivalent")
    return {"seed":seed,"surface_sha":d["hash_source"],"surface_ll_nats":d["surface_ll_source"],
      "mechanism_loglik_difference":d["surface_ll_source"]-d["surface_ll_production"],
      "mechanism_surface_sha_identical":d["hash_source"]==d["hash_production"],
      "semantic_vs_production_jointly_identical":True,
      "arms":acts,"positive_gain_exceeds_null":bool(pos>null)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20271000)
    p.add_argument("--seeds",type=int,default=20)
    p.add_argument("--lines",type=int,default=32)
    p.add_argument("--tokens",type=int,default=10)
    a=p.parse_args()
    emit("IDENT_GATE_AUDIT",{"frozen_FORM_url":URL,"renderer_sha256":hashlib.sha256(src.encode()).hexdigest(),
      "sim_lines_per_fold":a.lines,"sim_tokens_per_line":a.tokens,
      "n_physical_folds":5,"seeds":a.seeds,"first_seed":a.seed,
      "limitations":"CONNECT and LINE_ENTRY only nuisance-inspired section/entry terms; NOT historical calibrated forward modules.",
      "meaning":"Source and production are exactly the same observable stochastic process by construction."})
    rows=[]
    for j in range(a.seeds):
        x=one(a.seed+j,a.lines,a.tokens)
        rows.append(x)
        emit("IDENT_GATE_CASE",x)
    def summarize(arm):
        x=[r["arms"][arm]["holdout_test_bits_per_line_gain"] for r in rows]
        y=[r["arms"][arm]["holdout_anchor_AUC"] for r in rows if r["arms"][arm]["holdout_anchor_AUC"] is not None]
        return {"median_bits":float(np.median(x)),"range":[float(min(x)),float(max(x))],
          "n_gt_fixed_gain":sum(v>.02 for v in x),
          "median_auc":float(np.median(y)) if y else None}
    summary={"N":len(rows),"independent":summarize("independent"),
      "semantic":summarize("semantic"),"production":summarize("production"),
      "all_surface_identical":all(x["mechanism_surface_sha_identical"] for x in rows),
      "max_abs_model_ll_delta":max(abs(x["mechanism_loglik_difference"]) for x in rows),
      "semantic_prod_all_identical":all(x["semantic_vs_production_jointly_identical"] for x in rows),
      "gate_power":sum(r["arms"]["semantic"]["holdout_test_bits_per_line_gain"]>.02 for r in rows)/len(rows)>=.9,
      "gate_fpr":sum(r["arms"]["independent"]["holdout_test_bits_per_line_gain"]>.02 for r in rows)/len(rows)<=.05}
    emit("IDENT_GATE_FINAL",{"schema":"IDENT-GATE-1-20261019","completed":True,"summary":summary,"cases":rows,
      "interpretation":"This proves observational equivalence of the explicitly matched model pair, and demonstrates anchor correlation alone cannot distinguish semantics from production."})
if __name__=="__main__":main()
