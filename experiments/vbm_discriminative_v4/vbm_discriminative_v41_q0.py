# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse,concurrent.futures,json,statistics,sys
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
NS='VBMDISCV41'
b.NS=NS;q3.NS=NS;q3.b.NS=NS;v4.NS=NS;v4.b.NS=NS;v4.q3.NS=NS;v4.q3.b.NS=NS
CORE=q3.CORE_REGIMES;SCHED=q3.BRIDGE_SCHEDULES

def one_control(lms,truth_name,core_regime,bridge_schedule,rep):
    truth=lms[truth_name];tag=f'Q041:{truth_name}:{core_regime}:{bridge_schedule}:R{rep}'
    fw,hw=b.plain_span(truth.control,tag,18000,7000)
    p,u,ph,census=q3.make_key(truth,core_regime,bridge_schedule,tag)
    fc,ft=q3.encrypt_v3(fw,p,u,ph,bridge_schedule,tag+':F')
    hc,ht=q3.encrypt_v3(hw,p,u,ph,bridge_schedule,tag+':H')
    null=v4.best_null(fc,hc);cand=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fc,hc,lm,f'{tag}:{la}',ht if la==truth_name else None,40)
        cand.append({'language':la,'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_mean':r['score'],'score_gap':r['score_gap'],'decode_agreement':r['decode_agreement'],'recovery_diag':r['recovery'],'delta':r['score']-null['score']})
    meanrank=sorted(cand,key=lambda x:(-x['score_mean'],x['language']))
    Arank=sorted(cand,key=lambda x:(-x['score_A'],x['language']))
    Brank=sorted(cand,key=lambda x:(-x['score_B'],x['language']))
    tm=next(x for x in cand if x['language']==truth_name)
    margin=meanrank[0]['score_mean']-meanrank[1]['score_mean']
    qual=bool(meanrank[0]['language']==truth_name and Arank[0]['language']==truth_name and Brank[0]['language']==truth_name and margin>=.02 and tm['score_gap']<=.10 and tm['delta']>0)
    return {'truth':truth_name,'core_regime':core_regime,'bridge_schedule':bridge_schedule,'rep':rep,'winner_A':Arank[0]['language'],'winner_B':Brank[0]['language'],'winner_mean':meanrank[0]['language'],'margin_mean':margin,'truth_delta':tm['delta'],'truth_score_gap':tm['score_gap'],'truth_recovery_diag':tm['recovery_diag'],'qualified':qual,'null_score':null['score'],'null_model':null['model'],'homophone_counts':census,'fit_events':sum(map(len,fc)),'hold_events':sum(map(len,hc)),'candidates':cand}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms();jobs=[(la,r,s,rep) for la in lms for r in CORE for s in SCHED for rep in range(2)];rows=[]
    def one(j):return one_control(lms,*j)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for z in ex.map(one,jobs):
            rows.append(z);print('Q041',json.dumps({k:z[k] for k in ['truth','core_regime','bridge_schedule','rep','winner_A','winner_B','winner_mean','margin_mean','truth_delta','truth_score_gap','qualified','null_model']},sort_keys=True),flush=True)
    per={};cycle_ok=True
    for la in lms:
        z=[x for x in rows if x['truth']==la];cy=[x for x in z if x['bridge_schedule']=='CYCLE']
        per[la]={'qualified':sum(x['qualified'] for x in z),'total':24,'cycle_qualified':sum(x['qualified'] for x in cy),'cycle_total':12,'median_delta':statistics.median(x['truth_delta'] for x in z),'min_delta':min(x['truth_delta'] for x in z)}
        cycle_ok &= per[la]['cycle_qualified']>=10
    bavfp=all(x['qualified'] for x in rows if x['truth']=='bavarian' and x['core_regime']=='FREQ_PROP')
    passed=bool(all(per[la]['qualified']>=20 for la in per) and cycle_ok and bavfp)
    delta_floors={la:float(np.quantile([x['truth_delta'] for x in rows if x['truth']==la and x['qualified'] and x['truth_delta']>0],.05,method='linear')) for la in lms}
    margin_floors={la:float(np.quantile([x['margin_mean'] for x in rows if x['truth']==la and x['qualified']],.05,method='linear')) for la in lms}
    out={'namespace':NS,'total':72,'overall_qualified':sum(x['qualified'] for x in rows),'per_language':per,'cycle_gate':cycle_ok,'bavarian_freqprop_all4':bavfp,'delta_floors':delta_floors,'margin_floors':margin_floors,'pass':passed,'rows':rows}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
