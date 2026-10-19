#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
"""GAI-Q0: SOURCE-TO-FROZEN-FORM ORACLE CEILING.
Not a real Voynich inversion, not a blind recovery experiment.
Source shape: LANG (structured bigram), NOTATION (motif chain),
TABLE (IID Zipf), NONE (no source coupling).
The exact GAI prereg is authoritative.
"""
import argparse,collections,hashlib,json,math,urllib.request
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

FORM_SOURCE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
src=urllib.request.urlopen(FORM_SOURCE,timeout=180).read().decode()
ns={"__name__":"frozen_form_module"}
exec(compile(src,FORM_SOURCE,"exec"),ns)
k0=ns

def emit(tag,x):
    print(tag+"="+json.dumps(x,separators=(",",":"),sort_keys=True,default=str),flush=True)

def build_source(n,family,rng):
    # Important: all three sources have the SAME strictly positive Zipf
    # stationary-like marginal target. Source family only changes dynamics.
    # pi itself is not necessarily exactly stationary for non-IID A.
    freq=np.arange(1,n+1,dtype=float)**-0.8
    freq/=freq.sum()
    pi=freq[rng.permutation(n)]
    if family=="TABLE":
        return pi, np.tile(pi,(n,1)), {}
    A=np.tile(.30*pi if family=="LANG" else .10*pi,(n,1))
    # A source-specific hidden topology, sampled only from synthetic truth.
    if family=="LANG":
        # Zipf-like source with local reusable recurring item alternatives
        for x in range(n):
            y=int(rng.integers(n));z=int(rng.integers(n))
            while y==x:y=int(rng.integers(n))
            while z in (x,y):z=int(rng.integers(n))
            A[x,x]+=.20
            A[x,y]+=.30
            A[x,z]+=.20
    elif family=="NOTATION":
        order=rng.permutation(n)
        nxt={int(order[i]):int(order[(i+1)%n]) for i in range(n)}
        for x in range(n):
            A[x,x]+=.12
            A[x,nxt[x]]+=.78
    else:raise ValueError(family)
    A/=A.sum(axis=1,keepdims=True)
    return pi,A,{}

def sample_sequence(A,pi,N,rng):
    # This is a structured source, not a sample of actual Latin prose.
    n=len(pi);x=np.empty(N,dtype=int)
    x[0]=int(rng.choice(n,p=pi))
    for t in range(1,N):x[t]=int(rng.choice(n,p=A[x[t-1]]))
    return x

def frozen_controller(rng):
    U,Ve,Vr=k0["controls"](8,2,1.,rng,"F1")
    return U,Ve*5.,Vr*.5

def render(x,gmap,U,Ve,Vr,rng,none=False):
    obs=[]
    zeroU=np.zeros_like(U) if none else U
    for tokenid in x:
        for j in range(100):
            y=k0["gen_token"](int(gmap[tokenid]),zeroU,Ve,Vr,rng)
            if y is not None:
                obs.append(y);break
        else:raise RuntimeError("pathological token nontermination")
    return obs

def emission_matrix(obs,gmap,U,Ve,Vr,none=False):
    # Exact normalized frozen FORM likelihood, including STOP.
    nctrl=8
    zeros=np.zeros_like(U) if none else U
    classll=np.empty((len(obs),nctrl),dtype=float)
    for i,t in enumerate(obs):
        for c in range(nctrl):
            classll[i,c]=k0["token_loglik"](t,c,zeros,Ve,Vr)
    return classll[:,gmap]

def fb_predict(E,A,pi):
    # All normalization handled by exact forward backward likelihood.
    n,k=E.shape
    logA=np.log(np.maximum(A,1e-300))
    alpha=np.empty((n,k))
    scale=np.empty(n)
    alpha[0]=np.log(np.maximum(pi,1e-300))+E[0]
    scale[0]=logsumexp(alpha[0]);alpha[0]-=scale[0]
    for t in range(1,n):
        alpha[t]=logsumexp(alpha[t-1,:,None]+logA,axis=0)+E[t]
        scale[t]=logsumexp(alpha[t]);alpha[t]-=scale[t]
    beta=np.zeros((n,k))
    for t in range(n-2,-1,-1):
        beta[t]=logsumexp(logA+E[t+1,None,:]+beta[t+1,None,:],axis=1)-scale[t+1]
    p=alpha+beta
    p-=logsumexp(p,axis=1,keepdims=True)
    return float(scale.sum()),np.exp(p)

def simple_iid_loglik(E,pi):
    return float(logsumexp(np.log(np.maximum(pi,1e-300))[None,:]+E,axis=1).sum())

def evaluate(seed,nsource,family,N,block=20):
    rng=np.random.default_rng(seed)
    generator="LANG" if family=="NONE" else family
    pi,A,_=build_source(nsource,generator,rng)
    gmap=(rng.permutation(nsource)%8).astype(int)
    U,Ve,Vr=frozen_controller(rng)
    true=sample_sequence(A,pi,N,rng)
    obs=render(true,gmap,U,Ve,Vr,rng,none=family=="NONE")
    E=emission_matrix(obs,gmap,U,Ve,Vr,none=family=="NONE")
    ll,P=fb_predict(E,A,pi)
    pred=P.argmax(1)
    lo=int(.75*N)
    acc=float(np.mean(pred[lo:]==true[lo:]))
    groupacc=float(np.mean(gmap[pred[lo:]]==gmap[true[lo:]]))
    source_nmi=float(normalized_mutual_info_score(true[lo:],pred[lo:]))
    source_ari=float(adjusted_rand_score(true[lo:],pred[lo:]))
    group_nmi=float(normalized_mutual_info_score(gmap[true[lo:]],gmap[pred[lo:]]))
    ll_iid=simple_iid_loglik(E,pi)
    # Control: same multiset of observations, shuffled in local blocks,
    # destroying sequential association but not marginal FORM distribution.
    order=np.arange(N)
    for a in range(0,N,block):
        ix=order[a:a+block].copy()
        rng.shuffle(ix)
        order[a:a+block]=ix
    Esh=E[order]
    llsh,_=fb_predict(Esh,A,pi)
    shuffled_gain=(llsh-simple_iid_loglik(Esh,pi))/N/math.log(2)
    evidence_gain=(ll-ll_iid)/N/math.log(2)
    lens=[len(x) for x in obs]
    # Pairwise posterior same-identity confidence is another future target.
    # This stage uses the upper-bound hard-label NMI and group confusion only.
    return {
        "seed":seed,"family":family,"n_source":nsource,"token_n":N,
        "source_nmi":source_nmi,"source_ari":source_ari,
        "source_exact_accuracy":acc,"select_group_accuracy":groupacc,
        "select_group_nmi":group_nmi,
        "sequential_over_iid_bits_per_token":evidence_gain,
        "shuffled_sequential_over_iid_bits_per_token":shuffled_gain,
        "seq_over_shuffled_gain_bits":evidence_gain-shuffled_gain,
        "mean_FORM_piece_len":float(np.mean(lens)),
        "q95_FORM_piece_len":float(np.quantile(lens,.95)),
        "unique_surface_forms":len({tuple(t) for t in obs}),
        "source_types_seen":len(set(true)),
        "emission_matrix_hash":hashlib.sha256(E.round(8).tobytes()).hexdigest(),
        "oracle_warning":"TRUE A, pi, SELECT controls and source-to-8-controller map given to decoder. No blind recovery."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=32)
    p.add_argument("--tokens",type=int,default=600)
    p.add_argument("--seeds",type=int,default=2)
    p.add_argument("--seed",type=int,default=20262000)
    p.add_argument("--families",nargs="+",choices=["LANG","NOTATION","TABLE","NONE"],default=["LANG","NOTATION","TABLE","NONE"])
    args=p.parse_args()
    emit("GAI_Q0_AUDIT",{
        "source_model":"structured abstract vocabulary, not authentic language text",
        "forward_form_source":FORM_SOURCE,"source_type_controller_count":8,
        "k12":int(k0["K12"]),"piece_types":int(k0["NP"]),
        "legal_edges":int(k0["LEGAL"].sum()),"mean_real_form_length":float(np.mean([len(x) for x in k0["routes"]])),
        "src_sha":hashlib.sha256(src.encode()).hexdigest(),
        "no_CONNECT_LINE_ENTRY_in_this_stage":True,
        "oracle_known_param":True,
        "n_source":args.n,"tokens":args.tokens,"seeds":args.seeds,"seed_start":args.seed,
        "families":args.families,
        "fixed_pilot_not_confirmatory":True
    })
    results=[]
    for family in args.families:
        for i in range(args.seeds):
            record=evaluate(args.seed+i+["LANG","NOTATION","TABLE","NONE"].index(family)*10000,args.n,family,args.tokens)
            results.append(record)
            emit("GAI_Q0_CASE",record)
    summary=[]
    for family in args.families:
        a=[x for x in results if x["family"]==family]
        summary.append({"family":family,"n":len(a),
        "oracle_nmi_median":float(np.median([x["source_nmi"] for x in a])),
        "oracle_nmi_min":float(min(x["source_nmi"] for x in a)),
        "oracle_group_nmi_median":float(np.median([x["select_group_nmi"] for x in a])),
        "seq_gain_mean_bits":float(np.mean([x["sequential_over_iid_bits_per_token"] for x in a])),
        "order_effect_mean_bits":float(np.mean([x["seq_over_shuffled_gain_bits"] for x in a]))})
    emit("GAI_Q0_FINAL",{"schema":"GAI-Q0-STRUCTURED-ORACLE-PILOT-20261019",
       "complete":True,"results":results,"summary":summary,
       "limits":"A few oracle-positive synthetic samples only. Does not qualify blind inversion or tell whether Voynich encodes language."})
if __name__=="__main__":main()
