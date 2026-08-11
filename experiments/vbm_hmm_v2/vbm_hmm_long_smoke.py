# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures,json,sys
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import vbm_hmm_moment_v2 as m
b=m.b

def main():
    lms=b.load_lms();lm=lms['bavarian'];jobs=[('UNIFORM','FLAT'),('FREQ_PROP','SKEW'),('DIRICHLET_SKEW','SKEW')]
    def one(j):return b.q0_one(lm,j[0],j[1],18000,7000,40)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('LONG_SMOKE',json.dumps(z,sort_keys=True),flush=True)
    print('RESULT_JSON',json.dumps({'rows':rows,'all_pass':all(x['pass'] for x in rows)},sort_keys=True))
if __name__=='__main__':main()
