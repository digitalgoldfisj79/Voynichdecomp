# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_discriminative_v4')
import vbm_hmm_moment_v2 as m
import vbm_amadi_q0_v3 as q3
import vbm_discriminative_v4 as v4
b=m.b
NS='VBMBGCONDV5Q1'
b.NS=NS;q3.NS=NS;q3.b.NS=NS;v4.NS=NS
CORE_REGIMES=q3.CORE_REGIMES
K=b.KCORE;A=b.NOBS

def seed(*x):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big')&0x7fffffff

def positive_surface(lms,truth_name,regime,tag,fitn=12000,holdn=5000):
    lm=lms[truth_name]
    fw,hw=b.plain_span(lm.control,tag,fitn,holdn)
    p,u,ph,census=q3.make_key(lm,regime,'CYCLE',tag)
    fc,ft=q3.encrypt_v3(fw,p,u,ph,'CYCLE',tag+':F')
    hc,ht=q3.encrypt_v3(hw,p,u,ph,'CYCLE',tag+':H')
    return fc,hc,{'truth':truth_name,'regime':regime,'census':census}

def _marginals(fit):
    c=np.full(K,0.5,float);v=np.full(A-K,0.5,float);tc=tv=0
    for q in fit:
        z=np.asarray(q,int)
        zc=z[z<K];zv=z[z>=K]-K
        if len(zc):c+=np.bincount(zc,minlength=K);tc+=len(zc)
        if len(zv):v+=np.bincount(zv,minlength=A-K);tv+=len(zv)
    c/=c.sum();v/=v.sum();return c,v,tc/max(1,tc+tv)

def _lengths(template):return [len(q) for q in template]

def _gen_iid(fit,template,tag):
    pc,pv,p_core=_marginals(fit);rng=np.random.default_rng(seed(NS,'iid',tag));out=[]
    for L in _lengths(template):
        q=[]
        for _ in range(L):
            if rng.random()<p_core:q.append(int(rng.choice(K,p=pc)))
            else:q.append(K+int(rng.choice(A-K,p=pv)))
        out.append(np.asarray(q,np.int32))
    return out

def _fit_type_markov(fit):
    init=np.full(2,0.5,float);T=np.full((2,2),0.5,float)
    for q in fit:
        if len(q)==0:continue
        typ=[0 if int(x)<K else 1 for x in q];init[typ[0]]+=1
        for x,y in zip(typ,typ[1:]):T[x,y]+=1
    init/=init.sum();T/=T.sum(1,keepdims=True);return init,T

def _gen_markov1(fit,template,tag):
    pc,pv,_=_marginals(fit);init,T=_fit_type_markov(fit);rng=np.random.default_rng(seed(NS,'m1',tag));out=[]
    for L in _lengths(template):
        if L<=0:continue
        t=int(rng.choice(2,p=init));q=[]
        for i in range(L):
            q.append(int(rng.choice(K,p=pc)) if t==0 else K+int(rng.choice(A-K,p=pv)))
            if i+1<L:t=int(rng.choice(2,p=T[t]))
        out.append(np.asarray(q,np.int32))
    return out

def _fit_slots(fit,p):
    TC=np.full((p,2),0.5,float);C=np.full((p,K),0.5,float);V=np.full((p,A-K),0.5,float)
    for q in fit:
        for i,yy in enumerate(q):
            y=int(yy);r=i%p
            if y<K:TC[r,0]+=1;C[r,y]+=1
            else:TC[r,1]+=1;V[r,y-K]+=1
    TC/=TC.sum(1,keepdims=True);C/=C.sum(1,keepdims=True);V/=V.sum(1,keepdims=True);return TC,C,V

def _gen_slots(fit,template,tag,p):
    TC,C,V=_fit_slots(fit,p);rng=np.random.default_rng(seed(NS,'slot',tag,p));out=[]
    for L in _lengths(template):
        q=[]
        for i in range(L):
            r=i%p;t=int(rng.choice(2,p=TC[r]))
            q.append(int(rng.choice(K,p=C[r])) if t==0 else K+int(rng.choice(A-K,p=V[r])))
        out.append(np.asarray(q,np.int32))
    return out

def _block_shuffle(seqs,tag,bs=3):
    rng=np.random.default_rng(seed(NS,'block',tag));out=[]
    for q in seqs:
        z=list(map(int,q));blocks=[z[i:i+bs] for i in range(0,len(z),bs)];rng.shuffle(blocks);flat=[x for bl in blocks for x in bl];out.append(np.asarray(flat,np.int32))
    return out

def destroy(fit,hold,family,tag,seed_idx):
    if family=='typed_iid':return _gen_iid(fit,fit,tag+':F'),_gen_iid(fit,hold,tag+':H')
    if family=='typed_markov1':return _gen_markov1(fit,fit,tag+':F'),_gen_markov1(fit,hold,tag+':H')
    if family=='typed_slot':
        p=2+(seed(NS,'period',tag,seed_idx)%7);return _gen_slots(fit,fit,tag+':F',p),_gen_slots(fit,hold,tag+':H',p)
    if family=='block_shuffle3':return _block_shuffle(fit,tag+':F',3),_block_shuffle(hold,tag+':H',3)
    raise ValueError(family)

def score_surface(lms,fit,hold,tag):
    null=v4.best_null(fit,hold);cand=[]
    for la in ['bavarian','german']:
        r=m.paired_fit_moment(fit,hold,lms[la],f'{tag}:{la}',None,40)
        cand.append({'language':la,'score':r['score'],'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_gap':r['score_gap'],'decode_agreement':r['decode_agreement']})
    winner=max(cand,key=lambda x:x['score'])
    delta=float(winner['score']-null['score'])
    return {'delta':delta,'winner':winner['language'],'winner_score':winner['score'],'null_score':null['score'],'null_model':null['model'],'candidates':cand}

def one_positive(lms,phase,truth,regime):
    tag=f'{phase}:POS:{truth}:{regime}:CYCLE';fit,hold,meta=positive_surface(lms,truth,regime,tag);sc=score_surface(lms,fit,hold,tag)
    return {'kind':'positive','phase':phase,'truth':truth,'regime':regime,'family':None,**sc,'fit_events':sum(map(len,fit)),'hold_events':sum(map(len,hold))}

def one_negative(lms,phase,family,i):
    truth='bavarian' if i%2==0 else 'german';regime=CORE_REGIMES[i%len(CORE_REGIMES)];tag=f'{phase}:NEG:{family}:{i}:{truth}:{regime}';fit0,hold0,_=positive_surface(lms,truth,regime,tag+':BASE');fit,hold=destroy(fit0,hold0,family,tag,i);sc=score_surface(lms,fit,hold,tag)
    return {'kind':'negative','phase':phase,'truth':None,'source_truth':truth,'regime':regime,'family':family,**sc,'fit_events':sum(map(len,fit)),'hold_events':sum(map(len,hold))}

def run_phase(lms,phase,workers):
    jobs=[]
    for truth in ['bavarian','german']:
        for regime in CORE_REGIMES:jobs.append(('p',truth,regime))
    for fam in ['typed_iid','typed_markov1','typed_slot','block_shuffle3']:
        for i in range(6):jobs.append(('n',fam,i))
    def one(j):
        return one_positive(lms,phase,j[1],j[2]) if j[0]=='p' else one_negative(lms,phase,j[1],j[2])
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for z in ex.map(one,jobs):
            rows.append(z);print('Q1V5',json.dumps({k:z.get(k) for k in ['phase','kind','truth','source_truth','regime','family','delta','winner','null_model']},sort_keys=True),flush=True)
    return rows

def rates(rows,tau):
    pos=[x for x in rows if x['kind']=='positive'];neg=[x for x in rows if x['kind']=='negative'];tp=sum(x['delta']>=tau for x in pos);fp=sum(x['delta']>=tau for x in neg);tpr=tp/max(1,len(pos));fpr=fp/max(1,len(neg));out={'tau':tau,'positive_n':len(pos),'negative_n':len(neg),'tp':tp,'fp':fp,'tpr':tpr,'fpr':fpr,'balanced_accuracy':(tpr+(1-fpr))/2}
    for la in ['bavarian','german']:
        q=[x for x in pos if x['truth']==la];out[la+'_recall']=sum(x['delta']>=tau for x in q)/max(1,len(q))
    return out

def choose_tau(rows):
    vals=sorted(set(float(x['delta']) for x in rows),reverse=True);best=None
    for t in vals:
        r=rates(rows,t)
        if r['fpr']<=.05:
            key=(r['tpr'],t)
            if best is None or key>best[0]:best=(key,r)
    if best is None:raise RuntimeError('no feasible threshold')
    return best[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms()
    cal=run_phase(lms,'CAL',a.workers);thr=choose_tau(cal);print('CAL_THRESHOLD',json.dumps(thr,sort_keys=True),flush=True)
    val=run_phase(lms,'VAL',a.workers);vr=rates(val,thr['tau']);passed=bool(vr['tpr']>=.85 and vr['fpr']<=.05 and vr['balanced_accuracy']>=.90 and vr['bavarian_recall']>=.80 and vr['german_recall']>=.80)
    fam={}
    for f in ['typed_iid','typed_markov1','typed_slot','block_shuffle3']:
        q=[x for x in val if x.get('family')==f];fam[f]={'n':len(q),'false_positives':sum(x['delta']>=thr['tau'] for x in q),'median_delta':float(np.median([x['delta'] for x in q]))}
    out={'namespace':NS,'TAU_LANG':thr['tau'],'calibration':thr,'validation':vr,'validation_negative_families':fam,'pass':passed,'cal_rows':cal,'val_rows':val}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
