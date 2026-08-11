# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures,json,statistics,sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import vbm_hmm_moment_v2 as m
b=m.b
b.NS='VBMHMMV2LIDQ0'
ANCHORS={('UNIFORM','FLAT'),('DIRICHLET_SKEW','SKEW')}

def one_control(lms,truth_name,regime,use):
    truth=lms[truth_name];tag=f'Q0LID:{truth_name}:{regime}:{use}';fw,hw=b.plain_span(truth.control,tag,18000,7000);p,u,census=b.hidden_homophones(truth,regime,use,tag);fc,ft=b.encrypt(fw,p,u,tag+':F');hc,ht=b.encrypt(hw,p,u,tag+':H');cand=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fc,hc,lm,f'{tag}:{la}',ht if la==truth_name else None,40)
        cand.append({'language':la,'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_mean':r['score'],'score_gap':r['score_gap'],'decode_agreement':r['decode_agreement'],'truth_recovery':r['recovery'],'A_iter':r['A']['iterations'],'B_iter':r['B']['iterations'],'A_em_stop':r['A']['converged'],'B_em_stop':r['B']['converged']})
    meanrank=sorted(cand,key=lambda x:(-x['score_mean'],x['language']));Arank=sorted(cand,key=lambda x:(-x['score_A'],x['language']));Brank=sorted(cand,key=lambda x:(-x['score_B'],x['language']));tm=next(x for x in cand if x['language']==truth_name);margin=meanrank[0]['score_mean']-meanrank[1]['score_mean'];qualified=bool(meanrank[0]['language']==truth_name and Arank[0]['language']==truth_name and Brank[0]['language']==truth_name and margin>=.02 and tm['score_gap']<=.10)
    return {'truth':truth_name,'allocation':regime,'usage':use,'winner_mean':meanrank[0]['language'],'winner_A':Arank[0]['language'],'winner_B':Brank[0]['language'],'margin_mean':margin,'truth_score_gap':tm['score_gap'],'truth_recovery_diag':tm['truth_recovery'],'qualified':qualified,'candidates':cand,'homophone_counts':census,'fit_events':sum(map(len,fc)),'hold_events':sum(map(len,hc))}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms();jobs=[(la,r,u) for la in lms for r in b.ALLOC for u in b.USES];rows=[]
    def one(j):return one_control(lms,*j)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('Q0LID',json.dumps({k:z[k] for k in ['truth','allocation','usage','winner_mean','winner_A','winner_B','margin_mean','truth_score_gap','truth_recovery_diag','qualified']},sort_keys=True),flush=True)
    per={};floors={}
    for la in lms:
        z=[x for x in rows if x['truth']==la];per[la]={'qualified':sum(x['qualified'] for x in z),'total':len(z),'min_margin':min(x['margin_mean'] for x in z),'median_margin':statistics.median(x['margin_mean'] for x in z),'anchor_pass':all(x['qualified'] for x in z if (x['allocation'],x['usage']) in ANCHORS),'freqprop_pass':all(x['qualified'] for x in z if x['allocation']=='FREQ_PROP')};floors[la]=float(np.quantile(np.array([x['margin_mean'] for x in z],float),.05,method='linear'))
    overall=sum(x['qualified'] for x in rows);freqall=all(x['qualified'] for x in rows if x['allocation']=='FREQ_PROP');anchorsall=all(per[la]['anchor_pass'] for la in per);passed=bool(overall>=32 and all(per[la]['qualified']>=10 for la in per) and freqall and anchorsall)
    out={'namespace':b.NS,'rows':rows,'overall_qualified':overall,'total':len(rows),'per_language':per,'margin_floors':floors,'freqprop_all6_pass':freqall,'anchor_extremes_all_pass':anchorsall,'pass':passed}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
