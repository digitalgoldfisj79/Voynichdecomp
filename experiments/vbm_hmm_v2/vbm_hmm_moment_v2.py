# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import sys, numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import vbm_hmm_v2 as b
b.NS='VBMHMMV2MOMENT'

def surface_bigram(seqs):
    C=np.zeros((b.NOBS,b.NOBS),float);n=0
    for q in seqs:
        z=np.asarray(q,np.int32)
        for x,y in zip(z,z[1:]):C[int(x),int(y)]+=1;n+=1
    if n:C/=n
    return C

def softmax_rows(Z):
    E=np.zeros_like(Z)
    for s in b.CIDX:
        z=Z[int(s),:b.KCORE];z=z-z.max();e=np.exp(z);E[int(s),:b.KCORE]=e/e.sum()
    for s in b.VIDX:
        z=Z[int(s),b.KCORE:];z=z-z.max();e=np.exp(z);E[int(s),b.KCORE:]=e/e.sum()
    return E

def moment_fit(P,lm,tag,start,steps=1500,lr=0.05):
    rng=np.random.default_rng(b.seed(b.NS,'moment',tag,start));Z=np.zeros((b.A,b.NOBS),float)
    for s in b.CIDX:Z[int(s),:b.KCORE]=rng.normal(0,.35,b.KCORE)
    for s in b.VIDX:Z[int(s),b.KCORE:]=rng.normal(0,.35,b.KBR)
    m=np.zeros_like(Z);v=np.zeros_like(Z);beta1=.9;beta2=.999;eps=1e-8
    J=(lm.pi[:,None]*lm.T);J/=J.sum()
    best=(1e99,None)
    for t in range(1,steps+1):
        E=softmax_rows(Z);M=E.T@J@E;R=M-P;loss=float(np.sum(R*R))
        if loss<best[0]:best=(loss,E.copy())
        G=J@E@R.T + J.T@E@R
        GZ=np.zeros_like(Z)
        for s in b.CIDX:
            ss=int(s);e=E[ss,:b.KCORE];g=G[ss,:b.KCORE];GZ[ss,:b.KCORE]=e*(g-float(np.dot(g,e)))
        for s in b.VIDX:
            ss=int(s);e=E[ss,b.KCORE:];g=G[ss,b.KCORE:];GZ[ss,b.KCORE:]=e*(g-float(np.dot(g,e)))
        m=beta1*m+(1-beta1)*GZ;v=beta2*v+(1-beta2)*(GZ*GZ);mh=m/(1-beta1**t);vh=v/(1-beta2**t);Z-=lr*mh/(np.sqrt(vh)+eps)
        if t%100==0 and t>=400:
            # deterministic weak decay toward finite logits; avoids numerical one-hot lock before EM.
            Z*=.999
    return best[1],best[0]

def train_from_E(obs,offs,lm,E,maxit):
    prev=None;stable=0;conv=False;score=-1e99
    for it in range(maxit):
        ll,counts,_,n=b._fb_counts(obs,offs,lm.T,lm.pi,E,b.CIDX,b.VIDX,False);E=b._norm_emission(counts,0.05,b.CIDX,b.VIDX);score=ll/max(1,n)
        if prev is not None:
            rel=abs(score-prev)/max(1.,abs(prev));stable=stable+1 if rel<1e-7 else 0
            if stable>=3:conv=True;break
        prev=score
    return {'E':E,'train_score':score,'iterations':it+1,'converged':conv}

def paired_fit_moment(fit,hold,lm,tag,truth_hold=None,maxit=40):
    obs,offs=b.flatten(fit);P=surface_bigram(fit);runs=[]
    for st in range(4):
        E,ml=moment_fit(P,lm,tag,st);r=train_from_E(obs,offs,lm,E,maxit);r['moment_loss']=ml;r['start']=st;runs.append(r)
    ra=max(runs[:2],key=lambda z:z['train_score']);rb=max(runs[2:],key=lambda z:z['train_score']);ea=b.eval_E(hold,lm,ra['E'],truth_hold);eb=b.eval_E(hold,lm,rb['E'],truth_hold);agree=float(np.mean(ea['dec']==eb['dec'])) if len(ea['dec']) else 0.;gap=abs(ea['score']-eb['score']);rec=min(ea['recovery'],eb['recovery']) if truth_hold is not None else None
    return {'A':ra,'B':rb,'A_eval':ea,'B_eval':eb,'score':(ea['score']+eb['score'])/2,'score_gap':gap,'decode_agreement':agree,'recovery':rec,'converged':bool(ra['converged'] and rb['converged'])}

b.paired_fit=paired_fit_moment
if __name__=='__main__':b.main()
