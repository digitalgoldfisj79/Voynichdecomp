# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, json, math, statistics, sys
from dataclasses import dataclass
import numpy as np
from numba import njit
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
import amadi_residuals_v1 as ar
import vbm_structure_v1 as s0
import vbm_typed_v1 as vt

ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}
NS='VBMHMMV2'; PLAIN=s0.PLAIN; A=len(PLAIN); P2I={c:i for i,c in enumerate(PLAIN)}
VIDX=np.array([P2I[c] for c in 'aeiou'],np.int32); CIDX=np.array([i for i,c in enumerate(PLAIN) if c not in 'aeiou'],np.int32)
KCORE=21; KBR=123; NOBS=KCORE+KBR
ALLOC=['ANTI_SQRT','UNIFORM','SQRT_FREQ','FREQ_PROP','SUPER_FREQ','DIRICHLET_SKEW']
USES=['FLAT','SKEW']
H1=s0.H1; C1=s0.C1

def seed(*x): return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big') & 0x7fffffff

@dataclass
class LM:
    name:str; T:np.ndarray; pi:np.ndarray; tri:np.ndarray; freq:np.ndarray; control:list[str]; meta:dict

def build_lm(name,train,control):
    C=np.full((A,A),0.25,float); F=np.full(A,0.25,float); tri=np.full((A+1,A+1,A+1),0.25,float); B=A; n=0
    for raw in train:
        q=s0.norm(raw)
        if not q: continue
        z=[P2I[x] for x in q]; n+=len(z)
        for x in z:F[x]+=1
        for x,y in zip(z,z[1:]): C[x,y]+=1
        zz=[B,B]+z+[B,B]
        for x,y,w in zip(zz,zz[1:],zz[2:]): tri[x,y,w]+=1
    C/=C.sum(1,keepdims=True); F/=F.sum(); tri/=tri.sum(2,keepdims=True)
    ctrl=[s0.norm(x) for x in control if s0.norm(x)]
    return LM(name,C,F.copy(),np.log(tri),F,ctrl,{'train_chars':n,'control_chars':sum(map(len,ctrl))})

def load_lms():
    cs=s0.corpora(); return {la:build_lm(la,*cs[la]) for la in ['bavarian','german','italian']}

def plain_span(control,tag,fitn,holdn):
    st=seed(NS,'span',tag)%len(control); fit=[];hold=[];nf=nh=0;j=0
    while nh<holdn:
        q=control[(st+j)%len(control)];j+=1
        if not q:continue
        if nf<fitn:fit.append(q);nf+=len(q)
        else:hold.append(q);nh+=len(q)
        if j>len(control)*50: raise RuntimeError(('span exhausted',tag,nf,nh,len(control)))
    return fit,hold

def largest_remainder(total,weights):
    w=np.array(weights,float); w=np.maximum(w,1e-12); w/=w.sum(); raw=w*total; z=np.floor(raw).astype(int); rem=total-int(z.sum()); frac=raw-z
    order=np.argsort(-frac,kind='stable')
    for i in range(rem):z[order[i]]+=1
    return z

def allocation_counts(states,nsurf,freq,regime,rng):
    m=len(states); rem=nsurf-m
    f=np.array([freq[int(x)] for x in states],float);f=np.maximum(f,1e-8)
    if regime=='ANTI_SQRT':w=f**-0.5
    elif regime=='UNIFORM':w=np.ones(m)
    elif regime=='SQRT_FREQ':w=f**0.5
    elif regime=='FREQ_PROP':w=f
    elif regime=='SUPER_FREQ':w=f**1.5
    elif regime=='DIRICHLET_SKEW':w=rng.dirichlet(np.full(m,0.20))*f**0.5
    else:raise ValueError(regime)
    return np.ones(m,dtype=int)+largest_remainder(rem,w)

def hidden_homophones(lm,regime,use,tag):
    rng=np.random.default_rng(seed(NS,'map',tag,regime,use)); pools={}; probs={}
    for states,lo,ns in [(CIDX,0,KCORE),(VIDX,KCORE,KBR)]:
        counts=allocation_counts(states,ns,lm.freq,regime,rng); surfaces=np.arange(lo,lo+ns,dtype=int);rng.shuffle(surfaces);k=0
        for st,c in zip(states,counts):
            p=surfaces[k:k+int(c)].copy();k+=int(c);pools[int(st)]=p
            if use=='FLAT':u=np.full(len(p),1/len(p))
            else:u=rng.dirichlet(np.full(len(p),0.25))
            probs[int(st)]=u
    return pools,probs,{PLAIN[int(st)]:len(pools[int(st)]) for st in range(A)}

def encrypt(seqs,pools,probs,tag):
    out=[];truth=[]
    for si,s in enumerate(seqs):
        rng=np.random.default_rng(seed(NS,'emit',tag,si));q=[];z=[]
        for ch in s:
            st=P2I[ch];p=pools[st];u=probs[st];q.append(int(p[int(rng.choice(len(p),p=u))]));z.append(st)
        if q:out.append(np.array(q,np.int32));truth.append(np.array(z,np.int32))
    return out,truth

def flatten(seqs):
    off=[0];flat=[]
    for q in seqs:flat.extend(map(int,q));off.append(len(flat))
    return np.array(flat,np.int32),np.array(off,np.int64)

@njit(nogil=True,cache=False)
def _fb_counts(obs,offs,T,pi,E,cstates,vstates,want_gamma):
    S=T.shape[0]; O=E.shape[1]; counts=np.zeros((S,O),np.float64); ll=0.0; total=0
    gamma_out=np.empty((len(obs),S),np.float32) if want_gamma else np.empty((1,1),np.float32)
    for ss in range(len(offs)-1):
        lo=offs[ss];hi=offs[ss+1];L=hi-lo
        if L<=0:continue
        alpha=np.zeros((L,S),np.float64); beta=np.zeros((L,S),np.float64); scales=np.ones(L,np.float64)
        y=obs[lo]; cur=cstates if y<21 else vstates; sc=0.0
        for jj in range(len(cur)):
            j=cur[jj];v=pi[j]*E[j,y];alpha[0,j]=v;sc+=v
        if sc<=1e-300:sc=1e-300
        scales[0]=sc;ll+=math.log(sc);alpha[0]/=sc
        for t in range(1,L):
            yp=obs[lo+t-1]; y=obs[lo+t]; prev=cstates if yp<21 else vstates; cur=cstates if y<21 else vstates;sc=0.0
            for jj in range(len(cur)):
                j=cur[jj];v=0.0
                for ii in range(len(prev)):
                    i=prev[ii];v+=alpha[t-1,i]*T[i,j]
                v*=E[j,y];alpha[t,j]=v;sc+=v
            if sc<=1e-300:sc=1e-300
            scales[t]=sc;ll+=math.log(sc)
            for jj in range(len(cur)):alpha[t,cur[jj]]/=sc
        y=obs[hi-1];cur=cstates if y<21 else vstates
        for jj in range(len(cur)):beta[L-1,cur[jj]]=1.0
        for t in range(L-2,-1,-1):
            y=obs[lo+t];yn=obs[lo+t+1];cur=cstates if y<21 else vstates;nxt=cstates if yn<21 else vstates;den=scales[t+1]
            for ii in range(len(cur)):
                i=cur[ii];v=0.0
                for jj in range(len(nxt)):
                    j=nxt[jj];v+=T[i,j]*E[j,yn]*beta[t+1,j]
                beta[t,i]=v/den
        for t in range(L):
            y=obs[lo+t];cur=cstates if y<21 else vstates;den=0.0
            for jj in range(len(cur)):
                j=cur[jj];den+=alpha[t,j]*beta[t,j]
            if den<=1e-300:den=1e-300
            for jj in range(len(cur)):
                j=cur[jj];g=alpha[t,j]*beta[t,j]/den;counts[j,y]+=g
                if want_gamma:gamma_out[lo+t,j]=g
            total+=1
    return ll,counts,gamma_out,total

@njit(nogil=True,cache=False)
def _norm_emission(counts,pseudo,cstates,vstates):
    S,O=counts.shape;E=np.zeros((S,O),np.float64)
    for ii in range(len(cstates)):
        s=cstates[ii];den=0.0
        for y in range(21):den+=counts[s,y]+pseudo
        for y in range(21):E[s,y]=(counts[s,y]+pseudo)/den
    for ii in range(len(vstates)):
        s=vstates[ii];den=0.0
        for y in range(21,O):den+=counts[s,y]+pseudo
        for y in range(21,O):E[s,y]=(counts[s,y]+pseudo)/den
    return E

def init_E(tag,start):
    rng=np.random.default_rng(seed(NS,'init',tag,start));E=np.zeros((A,NOBS),float)
    for s in CIDX:E[int(s),:KCORE]=rng.dirichlet(np.full(KCORE,0.35))
    for s in VIDX:E[int(s),KCORE:]=rng.dirichlet(np.full(KBR,0.35))
    return E

def train_start(obs,offs,lm,tag,start,maxit=40):
    E=init_E(tag,start);prev=None;stable=0;conv=False;ll=-1e99
    for it in range(maxit):
        ll,counts,_,n=_fb_counts(obs,offs,lm.T,lm.pi,E,CIDX,VIDX,False);E=_norm_emission(counts,0.05,CIDX,VIDX);score=ll/max(1,n)
        if prev is not None:
            rel=abs(score-prev)/max(1.0,abs(prev))
            stable=stable+1 if rel<1e-7 else 0
            if stable>=3:conv=True;break
        prev=score
    return {'E':E,'train_score':score,'iterations':it+1,'converged':conv}

def eval_E(seqs,lm,E,truth=None):
    obs,offs=flatten(seqs);ll,_,g,n=_fb_counts(obs,offs,lm.T,lm.pi,E,CIDX,VIDX,True);dec=np.argmax(g[:len(obs)],axis=1).astype(np.int32);rec=None
    if truth is not None:
        tr,_=flatten(truth);rec=float(np.mean(dec==tr)) if len(tr) else 0.0
    return {'score':ll/max(1,n),'dec':dec,'recovery':rec,'events':n}

def paired_fit(fit,hold,lm,tag,truth_hold=None,maxit=40):
    fo,ff=flatten(fit);runs=[]
    for st in range(4):runs.append(train_start(fo,ff,lm,tag,st,maxit))
    Aset=runs[:2];Bset=runs[2:];ra=max(Aset,key=lambda z:z['train_score']);rb=max(Bset,key=lambda z:z['train_score']);ea=eval_E(hold,lm,ra['E'],truth_hold);eb=eval_E(hold,lm,rb['E'],truth_hold)
    agree=float(np.mean(ea['dec']==eb['dec'])) if len(ea['dec']) else 0.0;gap=abs(ea['score']-eb['score']);rec=min(ea['recovery'],eb['recovery']) if truth_hold is not None else None
    return {'A':ra,'B':rb,'A_eval':ea,'B_eval':eb,'score':(ea['score']+eb['score'])/2,'score_gap':gap,'decode_agreement':agree,'recovery':rec,'converged':bool(ra['converged'] and rb['converged'])}

def q0_one(lm,regime,use,fitn,holdn,maxit):
    tag=f'Q0HS:{lm.name}:{regime}:{use}';fw,hw=plain_span(lm.control,tag,fitn,holdn);p,u,census=hidden_homophones(lm,regime,use,tag);fc,ft=encrypt(fw,p,u,tag+':F');hc,ht=encrypt(hw,p,u,tag+':H');r=paired_fit(fc,hc,lm,tag,ht,maxit)
    passed=bool(r['recovery']>=.85 and r['decode_agreement']>=.95 and r['score_gap']<=.01 and r['converged'])
    return {'language':lm.name,'allocation':regime,'usage':use,'fit_events':sum(map(len,fc)),'hold_events':sum(map(len,hc)),'recovery':r['recovery'],'decode_agreement':r['decode_agreement'],'score_gap':r['score_gap'],'score':r['score'],'A_converged':r['A']['converged'],'B_converged':r['B']['converged'],'A_iterations':r['A']['iterations'],'B_iterations':r['B']['iterations'],'pass':passed,'homophone_counts':census}

def q0(lms,workers,smoke=False):
    if smoke:jobs=[('bavarian','UNIFORM','FLAT'),('bavarian','FREQ_PROP','SKEW'),('bavarian','DIRICHLET_SKEW','SKEW')];fitn,holdn,maxit=5000,2000,20
    else:jobs=[(la,r,u) for la in lms for r in ALLOC for u in USES];fitn,holdn,maxit=18000,7000,40
    def one(j):return q0_one(lms[j[0]],j[1],j[2],fitn,holdn,maxit)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('Q0HS',json.dumps(z,sort_keys=True),flush=True)
    if smoke:return {'rows':rows,'pass':all(x['pass'] for x in rows),'smoke':True}
    anchors={('UNIFORM','FLAT'),('FREQ_PROP','FLAT'),('FREQ_PROP','SKEW'),('DIRICHLET_SKEW','SKEW')};per={};anchor_ok=True
    for la in lms:
        z=[x for x in rows if x['language']==la];per[la]={'passed':sum(x['pass'] for x in z),'total':len(z),'min_recovery':min(x['recovery'] for x in z),'median_recovery':statistics.median(x['recovery'] for x in z)}
        anchor_ok &= all(x['pass'] for x in z if (x['allocation'],x['usage']) in anchors)
    rec=[x['recovery'] for x in rows];passed=bool(all(per[la]['passed']>=10 for la in per) and anchor_ok and statistics.median(rec)>=.95 and min(rec)>=.80)
    return {'rows':rows,'per_language':per,'anchor_pass_all':anchor_ok,'median_recovery':statistics.median(rec),'min_recovery':min(rec),'pass':passed,'smoke':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0'],required=True);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();lms=load_lms();print('LMS',json.dumps({k:v.meta for k,v in lms.items()},sort_keys=True),flush=True);res=q0(lms,a.workers,a.mode=='smoke');print('RESULT_JSON',json.dumps(res,sort_keys=True))
if __name__=='__main__':main()
