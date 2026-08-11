# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json, statistics, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_key_transfer_v6')
import vbm_key_transfer_v6 as v6

NS='VBMKEYTRANSFERV61_POSTHOC_PAIR_NULL';v6.NS=NS;A=v6.A

def fit_pair(seqs):
    C=np.full((A,A),0.25,float)
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):C[int(x),int(y)]+=1
    C/=C.sum();return C

def main():
    folios,labs,meta=v6.target_folios();folios,labs=v6.balanced_hash_order(folios,labs,6);rows=[]
    latent_ites=[6.615209842270804,6.217175608621934,6.428126336135995,6.375233029430261,6.497390590334435,5.83763885644876]
    for k in range(6):
        tr=[f for i,f in enumerate(folios) if i%6!=k];ho=[f for i,f in enumerate(folios) if i%6==k]
        M=fit_pair(v6.flatten_folios(tr));hs=v6.flatten_folios(ho);obs=v6.score_M(hs,M);ps=[]
        for r in range(24):ps.append(v6.score_M(v6.flatten_folios(v6.permuted(ho,f'PAIR:F{k}',r)),M))
        ite=float(obs-statistics.median(ps));row={'fold':k,'pair_obs':obs,'pair_perm_median':statistics.median(ps),'pair_perm_max':max(ps),'pair_ITE':ite,'latent_ITE':latent_ites[k],'latent_minus_pair_ITE':latent_ites[k]-ite};rows.append(row);print('PAIR_AUDIT',json.dumps(row,sort_keys=True),flush=True)
    out={'stage':'POSTHOC_FIT_ONLY_MATCHED_PAIR','rows':rows,'median_pair_ITE':statistics.median(x['pair_ITE'] for x in rows),'median_latent_ITE':statistics.median(latent_ites),'median_latent_minus_pair_ITE':statistics.median(x['latent_minus_pair_ITE'] for x in rows)}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
