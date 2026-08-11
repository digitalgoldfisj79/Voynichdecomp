# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures,json,sys
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import vbm_hmm_moment_v2 as m
b=m.b

def one_regime(lms,regime,use):
    truth=lms['bavarian'];tag=f'LIDSMOKE:{regime}:{use}';fw,hw=b.plain_span(truth.control,tag,18000,7000);p,u,census=b.hidden_homophones(truth,regime,use,tag);fc,ft=b.encrypt(fw,p,u,tag+':F');hc,ht=b.encrypt(hw,p,u,tag+':H');rows=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fc,hc,lm,f'{tag}:{la}',ht if la=='bavarian' else None,40);rows.append({'language':la,'score':r['score'],'score_gap':r['score_gap'],'decode_agreement':r['decode_agreement'],'converged':r['converged'],'truth_recovery':r['recovery']})
    rows.sort(key=lambda x:(-x['score'],x['language']));return {'allocation':regime,'usage':use,'ranking':rows,'winner':rows[0]['language'],'margin':rows[0]['score']-rows[1]['score'],'truth_homophones':census}

def main():
    lms=b.load_lms();jobs=[('UNIFORM','FLAT'),('FREQ_PROP','SKEW'),('DIRICHLET_SKEW','SKEW')];out=[]
    # each regime internally runs 3 candidates; keep regimes serial to cap Numba memory/CPU contention.
    for j in jobs:
        z=one_regime(lms,*j);out.append(z);print('LID_SMOKE',json.dumps(z,sort_keys=True),flush=True)
    print('RESULT_JSON',json.dumps({'rows':out,'bavarian_wins':sum(x['winner']=='bavarian' for x in out),'all_margin_002':all(x['winner']=='bavarian' and x['margin']>=.02 for x in out)},sort_keys=True))
if __name__=='__main__':main()
