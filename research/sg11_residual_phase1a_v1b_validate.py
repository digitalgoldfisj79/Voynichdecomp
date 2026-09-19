#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, math, re
from collections import Counter, defaultdict
import numpy as np

BASE_PATH='/tmp/hf_emergent_occupancy_fold.py'
TAU=float(__import__('os').environ.get('SG11_TAU','4096'))
STRENGTHS=(0.10,0.20,0.35)
NONEMPTY_QUIRES=(1,2,3,4,5,6,7,8,9,10,13,14,15,17,19,20)

spec=importlib.util.spec_from_file_location('base',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def qnum(page):
    n=base.parse_num(page)
    if n is None:return None
    if n<=8:return 1
    if n<=16:return 2
    if n<=24:return 3
    if n<=32:return 4
    if n<=40:return 5
    if n<=48:return 6
    if n<=56:return 7
    if n<=66:return 8
    if n<=68:return 9
    if n<=70:return 10
    if n<=72:return 11
    if n<=74:return 12
    if n<=84:return 13
    if n<=86:return 14
    if n<=90:return 15
    if n<=92:return 16
    if n<=96:return 17
    if n<=98:return 18
    if n<=102:return 19
    if n<=116:return 20
    return None

def h32(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)

def semantic_label(r): return h32('SEM|'+r['page'])%6
def exemplar_label(r): return h32('EX|'+r['bifolium'])%8
def serial_label(r): return ((int(r['line'])-1)//3)%4
def continuous_label(r): return h32('CONT|'+r['page'])%4
def production_label(r): return str(r['hand'])
def mixed_label(r): return semantic_label(r)*16 + (h32('HAND|'+str(r['hand']))%16)

def carrier_label(r,kind):
    if kind=='SEMANTIC':return semantic_label(r)
    if kind=='PRODUCTION':return production_label(r)
    if kind=='EXEMPLAR':return exemplar_label(r)
    if kind=='SERIAL':return serial_label(r)
    if kind=='CONTINUOUS':return continuous_label(r)
    if kind=='MIXED':return mixed_label(r)
    if kind=='NULL_SEM':return semantic_label(r)
    raise KeyError(kind)

def axis(label,fam,kind):
    if kind=='CONTINUOUS':
        a=1.0 if (h32('AX|CONT|'+fam)&1) else -1.0
        center=(-.75,-.25,.25,.75)[int(label)]
        return a*center
    if kind=='MIXED':
        sem=int(label)//16; hcode=int(label)%16
        a1=1.0 if (h32(f'AX|SEM|{sem}|{fam}')&1) else -1.0
        a2=1.0 if (h32(f'AX|HANDCODE|{hcode}|{fam}')&1) else -1.0
        return 0.5*(a1+a2)
    return 1.0 if (h32(f'AX|{kind}|{label}|{fam}')&1) else -1.0

def precompute_p0(rows,model,fams,fi):
    F=len(fams); gp=np.array([model.fam_gp.get(f,0.0) for f in fams],float)
    P=np.zeros((len(rows),F),np.float32); st=base.State()
    for i,r in enumerate(rows):
        st.boundary(r)
        c=model._fctx(r['section'],r['hand'],base.posclass(r),st.prevfam); nc=sum(c.values())
        ctx=np.fromiter((c.get(f,0) for f in fams),dtype=float,count=F)
        b=(ctx + base.BETA*gp)/(nc+base.BETA)
        npg=sum(st.famc.values())
        if npg:
            pc=np.fromiter((st.famc.get(f,0) for f in fams),dtype=float,count=F)
            p=(pc + model.alpha_f*b)/(npg+model.alpha_f)
        else:p=b
        p=np.maximum(p,1e-15);p/=p.sum();P[i]=p
        st.update_family(r['family'])
    return P

def gen_prob(P,rows,kind,strength,fams):
    if strength==0:return P.astype(np.float64,copy=True)
    G=np.empty_like(P,dtype=np.float64)
    labels=np.array([carrier_label(r,kind) for r in rows],dtype=object)
    for lab in sorted(set(labels.tolist()),key=str):
        ix=np.where(labels==lab)[0]
        ax=np.array([axis(lab,f,kind) for f in fams],float)
        w=np.exp2(strength*ax)
        X=P[ix].astype(np.float64)*w
        X/=X.sum(axis=1,keepdims=True);G[ix]=X
    return G

def sample_matrix(G,reps,rng,chunk=128):
    n,F=G.shape;Y=np.empty((n,reps),np.int16)
    for a in range(0,n,chunk):
        b=min(n,a+chunk);cdf=np.cumsum(G[a:b],axis=1);cdf[:,-1]=1.0
        u=rng.random((b-a,reps))
        # modest chunks keep (events x reps x families) memory bounded
        Y[a:b]=(u[:,:,None] > cdf[:,None,:]).sum(axis=2).astype(np.int16)
    return Y

def learn_ratios(P,Y,labels,reps,F):
    labs=sorted(set(labels.tolist()),key=str);li={x:i for i,x in enumerate(labs)}
    L=len(labs);E=np.zeros((L,F),float);O=np.zeros((reps,L,F),float)
    for lab in labs:
        ix=np.where(labels==lab)[0];j=li[lab];E[j]=P[ix].sum(axis=0)
        yy=Y[ix]
        for rr in range(reps):
            O[rr,j]=np.bincount(yy[:,rr],minlength=F)
    ratio=np.ones((reps,L,F),float)
    for j in range(L):
        et=E[j].sum();share=E[j]/max(et,1e-15)
        ratio[:,j,:]=(O[:,j,:] + TAU*share[None,:])/(E[j][None,:] + TAU*share[None,:] + 1e-15)
    return ratio,li

def test_effects(P,G,Y,labels,ratio,li):
    n,F=P.shape;reps=Y.shape[1];base_loss=np.zeros(reps);aug_loss=np.zeros(reps)
    for lab in sorted(set(labels.tolist()),key=str):
        ix=np.where(labels==lab)[0];j=li[lab];Psub=P[ix].astype(float);Ysub=Y[ix]
        # Z[event,rep] under multiplicative residual correction
        Z=Psub @ ratio[:,j,:].T
        rows=np.arange(len(ix))[:,None];rr=np.arange(reps)[None,:]
        py=Psub[rows,Ysub]
        ry=ratio[rr,j,Ysub]
        pay=np.maximum(py*ry/np.maximum(Z,1e-300),1e-300)
        base_loss+=(-np.log2(np.maximum(py,1e-300))).sum(axis=0)
        aug_loss+=(-np.log2(pay)).sum(axis=0)
    return (base_loss-aug_loss)/n

def oracle_gain(P,G):
    return float(np.mean(np.sum(G*np.log2(np.maximum(G,1e-300)/np.maximum(P,1e-300)),axis=1)))

def run(q,reps):
    rows,_=base.load_rows();qs=np.array([qnum(r['page']) for r in rows])
    trix=np.where(qs!=q)[0];teix=np.where(qs==q)[0]
    if len(teix)==0:raise RuntimeError('empty quire')
    train=[rows[i] for i in trix];model=base.FamilyModel(train);fams=sorted(model.fam_global);F=len(fams)
    P=precompute_p0(rows,model,fams,{f:i for i,f in enumerate(fams)})
    Ptr=P[trix];Pte=P[teix];Rtr=[rows[i] for i in trix];Rte=[rows[i] for i in teix]
    conditions=[('NULL_SEM',0.0)]
    for k in ('SEMANTIC','PRODUCTION','EXEMPLAR','SERIAL','CONTINUOUS'):
        for s in STRENGTHS:conditions.append((k,s))
    for s in (0.20,0.35):conditions.append(('MIXED',s))
    out={'quire':q,'ntrain':len(trix),'ntest':len(teix),'families':F,'reps':reps,'tau':TAU,'conditions':{}}
    for ci,(kind,strength) in enumerate(conditions):
        seed=202609290000 + q*10000 + ci*100
        rng=np.random.default_rng(seed)
        Gtr=gen_prob(Ptr,Rtr,kind,strength,fams);Gte=gen_prob(Pte,Rte,kind,strength,fams)
        Ytr=sample_matrix(Gtr,reps,rng);Yte=sample_matrix(Gte,reps,rng)
        ltr=np.array([carrier_label(r,kind) for r in Rtr],dtype=object)
        lte=np.array([carrier_label(r,kind) for r in Rte],dtype=object)
        ratio,li=learn_ratios(Ptr,Ytr,ltr,reps,F)
        # unseen categorical labels are a calibration failure for this quire/condition
        unseen=sorted(set(lte.tolist())-set(li),key=str)
        if unseen:
            eff=[float('nan')]*reps
        else:
            eff=test_effects(Pte,Gte,Yte,lte,ratio,li).tolist()
        key=f'{kind}@{strength:.2f}'
        out['conditions'][key]={'seed':seed,'oracle_gain_bits_per_event':oracle_gain(Pte,Gte),'unseen_labels':[str(x) for x in unseen],'effects':eff}
        print('DONE',q,key,'oracle',out['conditions'][key]['oracle_gain_bits_per_event'],flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quire',type=int,required=True);ap.add_argument('--reps',type=int,default=100)
    a=ap.parse_args()
    if a.quire not in NONEMPTY_QUIRES:raise SystemExit('quire not eligible')
    o=run(a.quire,a.reps)
    primary={'NULL_SEM@0.00','SEMANTIC@0.20','PRODUCTION@0.20','EXEMPLAR@0.20','SERIAL@0.20','CONTINUOUS@0.20','MIXED@0.20'}
    for k,v in o['conditions'].items():
        if k not in primary:
            x=np.asarray(v.pop('effects'),float)
            v['effect_mean']=float(np.mean(x)); v['effect_sd']=float(np.std(x,ddof=1)) if len(x)>1 else 0.0
    print('PHASE1A_VALID_Q_RESULT='+json.dumps(o,separators=(',',':'),sort_keys=True),flush=True)
