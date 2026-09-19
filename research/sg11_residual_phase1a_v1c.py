#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, math
import numpy as np

BASE_PATH='/tmp/hf_emergent_occupancy_fold.py'
TAU=1_000_000_000.0
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

def component_axis(component,level,fam):
    if component=='CONT':
        a=1.0 if (h32('AX|CONT|'+fam)&1) else -1.0
        center=(-.75,-.25,.25,.75)[int(level)]
        return a*center
    return 1.0 if (h32(f'AX|{component}|{level}|{fam}')&1) else -1.0

def carrier_components(r,kind):
    if kind in ('NULL_SEM','SEMANTIC'):
        return [('SEM',semantic_label(r),1.0)]
    if kind=='PRODUCTION':
        return [('HAND',production_label(r),1.0)]
    if kind=='EXEMPLAR':
        return [('EX',exemplar_label(r),1.0)]
    if kind=='SERIAL':
        return [('SER',serial_label(r),1.0)]
    if kind=='CONTINUOUS':
        return [('CONT',continuous_label(r),1.0)]
    if kind=='MIXED':
        return [('SEM',semantic_label(r),0.5),('HAND',production_label(r),0.5)]
    raise KeyError(kind)

def event_axis(r,kind,fam):
    return sum(w*component_axis(c,l,fam) for c,l,w in carrier_components(r,kind))

def precompute_p0(rows,model,fams):
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
        else:
            p=b
        p=np.maximum(p,1e-15); p/=p.sum(); P[i]=p
        st.update_family(r['family'])
    return P

def gen_prob(P,rows,kind,strength,fams):
    if strength==0:return P.astype(np.float64,copy=True)
    G=np.empty_like(P,dtype=np.float64)
    sigs={}
    for i,r in enumerate(rows):
        sig=tuple((c,str(l),float(w)) for c,l,w in carrier_components(r,kind))
        sigs.setdefault(sig,[]).append(i)
    for sig,idxs in sigs.items():
        ix=np.asarray(idxs,int)
        rr=rows[ix[0]]
        ax=np.array([event_axis(rr,kind,f) for f in fams],float)
        w=np.exp2(strength*ax)
        X=P[ix].astype(np.float64)*w
        X/=X.sum(axis=1,keepdims=True); G[ix]=X
    return G

def sample_matrix(G,reps,rng,chunk=128):
    n,F=G.shape; Y=np.empty((n,reps),np.int16)
    for a in range(0,n,chunk):
        b=min(n,a+chunk); cdf=np.cumsum(G[a:b],axis=1); cdf[:,-1]=1.0
        u=rng.random((b-a,reps))
        Y[a:b]=(u[:,:,None] > cdf[:,None,:]).sum(axis=2).astype(np.int16)
    return Y

def learn_component_ratios(P,Y,rows,kind,reps,F):
    specs={}
    for i,r in enumerate(rows):
        for comp,level,_w in carrier_components(r,kind):
            specs.setdefault((comp,str(level)),[]).append(i)
    ratios={}
    for key,idxs in specs.items():
        ix=np.asarray(idxs,int)
        E=P[ix].sum(axis=0)
        O=np.zeros((reps,F),float)
        yy=Y[ix]
        for rr in range(reps):
            O[rr]=np.bincount(yy[:,rr],minlength=F)
        et=E.sum(); share=E/max(et,1e-15)
        ratios[key]=(O + TAU*share[None,:])/(E[None,:] + TAU*share[None,:] + 1e-15)
    return ratios

def test_effects(P,Y,rows,kind,ratios):
    n,F=P.shape; reps=Y.shape[1]; base_loss=np.zeros(reps); aug_loss=np.zeros(reps)
    groups={}
    for i,r in enumerate(rows):
        sig=tuple((c,str(l),float(w)) for c,l,w in carrier_components(r,kind))
        groups.setdefault(sig,[]).append(i)
    unseen_event_count=0; unseen_components=set()
    for sig,idxs in groups.items():
        ix=np.asarray(idxs,int); Psub=P[ix].astype(float); Ysub=Y[ix]
        R=np.ones((reps,F),float)
        for comp,level,w in sig:
            key=(comp,level)
            if key in ratios:
                R *= np.power(np.maximum(ratios[key],1e-300),w)
            else:
                unseen_components.add(key)
                unseen_event_count += len(ix)
                # explicit hierarchical backoff: unseen component contributes neutral ratio 1
        Z=Psub @ R.T
        rows_ix=np.arange(len(ix))[:,None]; rr_ix=np.arange(reps)[None,:]
        py=Psub[rows_ix,Ysub]; ry=R[rr_ix,Ysub]
        pay=np.maximum(py*ry/np.maximum(Z,1e-300),1e-300)
        base_loss+=(-np.log2(np.maximum(py,1e-300))).sum(axis=0)
        aug_loss+=(-np.log2(pay)).sum(axis=0)
    return (base_loss-aug_loss)/n, sorted([f'{a}:{b}' for a,b in unseen_components]), unseen_event_count

def oracle_gain(P,G):
    return float(np.mean(np.sum(G*np.log2(np.maximum(G,1e-300)/np.maximum(P,1e-300)),axis=1)))

def conditions():
    out=[('NULL_SEM',0.0)]
    for k in ('SEMANTIC','PRODUCTION','EXEMPLAR','SERIAL','CONTINUOUS'):
        for s in STRENGTHS:out.append((k,s))
    for s in (0.20,0.35):out.append(('MIXED',s))
    return out

def run_q(q,reps):
    rows,_=base.load_rows(); qs=np.array([qnum(r['page']) for r in rows])
    trix=np.where(qs!=q)[0]; teix=np.where(qs==q)[0]
    if len(teix)==0:raise RuntimeError('empty quire')
    train=[rows[i] for i in trix]; model=base.FamilyModel(train); fams=sorted(model.fam_global); F=len(fams)
    P=precompute_p0(rows,model,fams)
    Ptr=P[trix]; Pte=P[teix]; Rtr=[rows[i] for i in trix]; Rte=[rows[i] for i in teix]
    out={'quire':q,'ntrain':len(trix),'ntest':len(teix),'families':F,'reps':reps,'tau':TAU,'conditions':{}}
    for ci,(kind,strength) in enumerate(conditions()):
        seed=202610050000 + q*10000 + ci*100
        rng=np.random.default_rng(seed)
        Gtr=gen_prob(Ptr,Rtr,kind,strength,fams); Gte=gen_prob(Pte,Rte,kind,strength,fams)
        Ytr=sample_matrix(Gtr,reps,rng); Yte=sample_matrix(Gte,reps,rng)
        ratios=learn_component_ratios(Ptr,Ytr,Rtr,kind,reps,F)
        eff,unseen,unseen_events=test_effects(Pte,Yte,Rte,kind,ratios)
        key=f'{kind}@{strength:.2f}'
        out['conditions'][key]={
            'seed':seed,'oracle_gain_bits_per_event':oracle_gain(Pte,Gte),
            'unseen_components':unseen,'unseen_event_component_count':unseen_events,
            'effects':eff.tolist()
        }
        print('DONE',q,key,'oracle',out['conditions'][key]['oracle_gain_bits_per_event'],'unseen',len(unseen),flush=True)
    return out

def exact_sign_matrix(n=16):
    x=np.arange(1<<n,dtype=np.uint32)[:,None]
    bits=((x >> np.arange(n,dtype=np.uint32)) & 1).astype(np.float64)
    return bits*2.0-1.0

def summarize_all(results):
    primary=['NULL_SEM@0.00','SEMANTIC@0.20','PRODUCTION@0.20','EXEMPLAR@0.20','SERIAL@0.20','CONTINUOUS@0.20','MIXED@0.20']
    n=len(results); reps=results[0]['reps']
    if n!=16:raise RuntimeError(f'need 16 quires, got {n}')
    S=exact_sign_matrix(n)
    summary={}
    for key in primary:
        E=np.array([r['conditions'][key]['effects'] for r in results],float) # 16 x reps
        obs=E.mean(axis=0)
        null_sd=np.sqrt((E*E).sum(axis=0))/n
        z=np.divide(obs,null_sd,out=np.zeros_like(obs),where=null_sd>0)
        flips=(S@E)/n
        p=(np.sum(flips>=obs[None,:],axis=0)+1.0)/(len(S)+1.0)
        wins=(E>0).sum(axis=0)
        det=(z>=2.0)&(p<=.05)&(wins>=11)
        summary[key]={
            'detections':int(det.sum()),'rate':float(det.mean()),
            'mean_gain_bits_per_event':float(obs.mean()),
            'mean_null_sd':float(null_sd.mean()),
            'mean_z':float(z.mean()),
            'median_exact_p':float(np.median(p)),
            'mean_positive_quires':float(wins.mean()),
            'unseen_component_folds':int(sum(bool(r['conditions'][key]['unseen_components']) for r in results))
        }
    passed=(summary['NULL_SEM@0.00']['detections']<=5 and
            all(summary[k]['detections']>=80 for k in primary[1:]))
    return {'screening_pass':bool(passed),'primary':summary}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quire',type=int)
    ap.add_argument('--reps',type=int,default=100)
    ap.add_argument('--all',action='store_true')
    a=ap.parse_args()
    if a.all:
        rr=[run_q(q,a.reps) for q in NONEMPTY_QUIRES]
        out={'protocol':'SG11_PHASE1A_V1C','quires':list(NONEMPTY_QUIRES),'reps':a.reps,
             'summary':summarize_all(rr),
             'unseen':{str(r['quire']):{k:v['unseen_components'] for k,v in r['conditions'].items() if v['unseen_components']} for r in rr}}
        print('PHASE1A_V1C_ALL_RESULT='+json.dumps(out,separators=(',',':'),sort_keys=True),flush=True)
    else:
        if a.quire not in NONEMPTY_QUIRES:raise SystemExit('quire not eligible')
        out=run_q(a.quire,a.reps)
        primary={'NULL_SEM@0.00','SEMANTIC@0.20','PRODUCTION@0.20','EXEMPLAR@0.20','SERIAL@0.20','CONTINUOUS@0.20','MIXED@0.20'}
        for k,v in out['conditions'].items():
            if k not in primary:
                x=np.asarray(v.pop('effects'),float)
                v['effect_mean']=float(np.mean(x));v['effect_sd']=float(np.std(x,ddof=1)) if len(x)>1 else 0.0
        print('PHASE1A_V1C_Q_RESULT='+json.dumps(out,separators=(',',':'),sort_keys=True),flush=True)
