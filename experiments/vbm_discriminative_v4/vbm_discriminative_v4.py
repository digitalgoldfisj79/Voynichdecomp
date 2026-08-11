# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,json,math,statistics,sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
import vbm_hmm_moment_v2 as m
import vbm_amadi_q0_v3 as q3
b=m.b
NS='VBMDISCV4'; b.NS=NS; q3.NS=NS; q3.b.NS=NS
CORE_REGIMES=q3.CORE_REGIMES
BRIDGE_SCHEDULES=q3.BRIDGE_SCHEDULES
A=b.NOBS; K=b.KCORE

def seed(*x):return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big')&0x7fffffff

def _fit_hier(seqs,max_order,alphabet=A,typed=False):
    base=np.full(alphabet,0.5,float); levels=[{} for _ in range(max_order+1)]
    # level0 kept as one global counter
    c0=np.zeros(alphabet,np.int64);n0=0
    type_levels=[{} for _ in range(max_order+1)] if typed else None
    t0=np.full(2,0.5,float) if typed else None
    for q in seqs:
        z=list(map(int,q));tz=[0 if x<K else 1 for x in z]
        for i,y in enumerate(z):
            c0[y]+=1;n0+=1
            if typed:t0[tz[i]]+=1
            for o in range(1,max_order+1):
                if i<o:break
                ctx=tuple(z[i-o:i]);d=levels[o].get(ctx)
                if d is None:d=np.zeros(alphabet,np.int32);levels[o][ctx]=d
                d[y]+=1
                if typed:
                    tc=tuple(tz[i-o:i]);dd=type_levels[o].get(tc)
                    if dd is None:dd=np.zeros(2,np.int32);type_levels[o][tc]=dd
                    dd[tz[i]]+=1
    return {'c0':c0,'n0':n0,'levels':levels,'type_levels':type_levels,'t0':t0,'max_order':max_order}

def _score_hier(model,seqs,order,typed=False,alpha=5.0):
    c0=model['c0'];n0=model['n0'];levels=model['levels'];ll=0.;n=0
    # ordinary hierarchical surface model
    if not typed:
        p0=(c0+0.5)/(n0+0.5*A)
        for q in seqs:
            z=list(map(int,q))
            for i,y in enumerate(z):
                p=float(p0[y])
                for o in range(1,min(order,i)+1):
                    d=levels[o].get(tuple(z[i-o:i]))
                    if d is not None:
                        den=float(d.sum());p=(float(d[y])+alpha*p)/(den+alpha)
                ll+=math.log(max(p,1e-300));n+=1
        return ll/max(1,n)
    # typed factorisation: P(type | type context) * P(symbol | surface context, type)
    t0=model['t0'];pt0=t0/t0.sum();tl=model['type_levels']
    # within-type base distribution
    pc=c0[:K]+0.5;pv=c0[K:]+0.5;pc=pc/pc.sum();pv=pv/pv.sum()
    for q in seqs:
        z=list(map(int,q));tz=[0 if x<K else 1 for x in z]
        for i,y in enumerate(z):
            ty=tz[i];pt=float(pt0[ty]);ps=float(pc[y] if ty==0 else pv[y-K])
            for o in range(1,min(order,i)+1):
                td=tl[o].get(tuple(tz[i-o:i]))
                if td is not None:
                    den=float(td.sum());pt=(float(td[ty])+alpha*pt)/(den+alpha)
                d=levels[o].get(tuple(z[i-o:i]))
                if d is not None:
                    if ty==0:
                        den=float(d[:K].sum());cy=float(d[y])
                    else:
                        den=float(d[K:].sum());cy=float(d[y])
                    ps=(cy+alpha*ps)/(den+alpha)
            ll+=math.log(max(pt*ps,1e-300));n+=1
    return ll/max(1,n)

def _iid_score(fit,hold):
    c=np.full(A,0.5,float);n=0
    for q in fit:
        z=np.asarray(q,int);c+=np.bincount(z,minlength=A);n+=len(z)
    p=c/c.sum();ll=0.;m0=0
    for q in hold:
        z=np.asarray(q,int);ll+=float(np.log(p[z]).sum());m0+=len(z)
    return ll/max(1,m0)

def _periodic_score(fit,hold,period,typed=False):
    if not typed:
        C=np.full((period,A),0.5,float)
        for q in fit:
            for i,y in enumerate(q):C[i%period,int(y)]+=1
        P=C/C.sum(1,keepdims=True);ll=0.;n=0
        for q in hold:
            for i,y in enumerate(q):ll+=math.log(float(P[i%period,int(y)]));n+=1
        return ll/max(1,n)
    TC=np.full((period,2),0.5,float);C0=np.full((period,K),0.5,float);C1=np.full((period,A-K),0.5,float)
    for q in fit:
        for i,yy in enumerate(q):
            y=int(yy);r=i%period;t=0 if y<K else 1;TC[r,t]+=1
            if t==0:C0[r,y]+=1
            else:C1[r,y-K]+=1
    PT=TC/TC.sum(1,keepdims=True);P0=C0/C0.sum(1,keepdims=True);P1=C1/C1.sum(1,keepdims=True);ll=0.;n=0
    for q in hold:
        for i,yy in enumerate(q):
            y=int(yy);r=i%period;t=0 if y<K else 1;p=float(PT[r,t])*(float(P0[r,y]) if t==0 else float(P1[r,y-K]));ll+=math.log(max(p,1e-300));n+=1
    return ll/max(1,n)

def best_null(fit,hold):
    out={'iid':_iid_score(fit,hold)}
    h=_fit_hier(fit,6,typed=False);ht=_fit_hier(fit,5,typed=True)
    for o in range(1,7):out[f'markov_hier_o{o}']=_score_hier(h,hold,o,False)
    for o in range(1,6):out[f'typed_hier_o{o}']=_score_hier(ht,hold,o,True)
    for p in range(2,9):
        out[f'slot_p{p}']=_periodic_score(fit,hold,p,False)
        out[f'typed_slot_p{p}']=_periodic_score(fit,hold,p,True)
    name=max(out,key=out.get);return {'score':out[name],'model':name,'scores':out}

def one_control(lms,truth_name,core_regime,bridge_schedule,fitn=18000,holdn=7000,maxit=40):
    truth=lms[truth_name];tag=f'Q0V4:{truth_name}:{core_regime}:{bridge_schedule}'
    fw,hw=b.plain_span(truth.control,tag,fitn,holdn);p,u,ph,census=q3.make_key(truth,core_regime,bridge_schedule,tag);fc,ft=q3.encrypt_v3(fw,p,u,ph,bridge_schedule,tag+':F');hc,ht=q3.encrypt_v3(hw,p,u,ph,bridge_schedule,tag+':H')
    null=best_null(fc,hc);cand=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fc,hc,lm,f'{tag}:{la}',ht if la==truth_name else None,maxit);cand.append({'language':la,'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_mean':r['score'],'score_gap':r['score_gap'],'decode_agreement':r['decode_agreement'],'recovery_diag':r['recovery'],'delta':r['score']-null['score']})
    meanrank=sorted(cand,key=lambda x:(-x['score_mean'],x['language']));Arank=sorted(cand,key=lambda x:(-x['score_A'],x['language']));Brank=sorted(cand,key=lambda x:(-x['score_B'],x['language']));tm=next(x for x in cand if x['language']==truth_name);margin=meanrank[0]['score_mean']-meanrank[1]['score_mean'];qual=bool(meanrank[0]['language']==truth_name and Arank[0]['language']==truth_name and Brank[0]['language']==truth_name and margin>=.02 and tm['score_gap']<=.10 and tm['delta']>0)
    return {'truth':truth_name,'core_regime':core_regime,'bridge_schedule':bridge_schedule,'winner_A':Arank[0]['language'],'winner_B':Brank[0]['language'],'winner_mean':meanrank[0]['language'],'margin_mean':margin,'truth_delta':tm['delta'],'truth_score_gap':tm['score_gap'],'truth_recovery_diag':tm['recovery_diag'],'qualified':qual,'null_score':null['score'],'null_model':null['model'],'homophone_counts':census,'fit_events':sum(map(len,fc)),'hold_events':sum(map(len,hc)),'candidates':cand,'null_scores':null['scores']}

def q0(lms,workers,smoke=False):
    jobs=[('bavarian','UNIFORM','FLAT'),('bavarian','FREQ_PROP','CYCLE'),('german','FREQ_PROP','CYCLE'),('italian','FREQ_PROP','CYCLE')] if smoke else [(la,r,s) for la in lms for r in CORE_REGIMES for s in BRIDGE_SCHEDULES]
    def one(j):return one_control(lms,*j,5000,2500,24) if smoke else one_control(lms,*j)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('Q0V4',json.dumps({k:z[k] for k in ['truth','core_regime','bridge_schedule','winner_A','winner_B','winner_mean','margin_mean','truth_delta','truth_score_gap','qualified','null_model']},sort_keys=True),flush=True)
    if smoke:return {'namespace':NS,'smoke':True,'pass':all(x['qualified'] for x in rows),'rows':rows}
    per={};cycle_ok=True
    for la in lms:
        z=[x for x in rows if x['truth']==la];cy=[x for x in z if x['bridge_schedule']=='CYCLE'];per[la]={'qualified':sum(x['qualified'] for x in z),'cycle_qualified':sum(x['qualified'] for x in cy),'total':12,'median_delta':statistics.median(x['truth_delta'] for x in z),'min_delta':min(x['truth_delta'] for x in z)};cycle_ok &= per[la]['cycle_qualified']>=5
    fp=all(x['qualified'] for x in rows if x['truth']=='bavarian' and x['core_regime']=='FREQ_PROP');overall=sum(x['qualified'] for x in rows);passed=bool(overall>=34 and all(per[la]['qualified']>=10 for la in per) and cycle_ok and fp)
    delta_floors={la:float(np.quantile([x['truth_delta'] for x in rows if x['truth']==la and x['truth_delta']>0],.05,method='linear')) for la in lms}
    margin_floors={la:float(np.quantile([x['margin_mean'] for x in rows if x['truth']==la and x['qualified']],.05,method='linear')) for la in lms}
    return {'namespace':NS,'smoke':False,'overall_qualified':overall,'total':36,'per_language':per,'cycle_gate':cycle_ok,'bavarian_freqprop_both':fp,'delta_floors':delta_floors,'margin_floors':margin_floors,'pass':passed,'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','q0'],required=True);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms();z=q0(lms,a.workers,a.mode=='smoke');print('RESULT_JSON',json.dumps(z,sort_keys=True))
if __name__=='__main__':main()
