# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json, math, statistics, sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1')
sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
sys.path.insert(0,'experiments/vbm_hmm_v2')
sys.path.insert(0,'experiments/vbm_amadi_homophone_v3')
sys.path.insert(0,'experiments/vbm_key_transfer_v6')
import vbm_key_transfer_v6 as v6

NS='VBMKEYTRANSFERV61_POSTHOC_SURFACE_NULL'
v6.NS=NS
A=v6.A


def fit_uni(seqs):
    c=np.full(A,0.5,float)
    for q in seqs:c+=np.bincount(np.asarray(q,int),minlength=A)
    return c/c.sum()

def score_uni(seqs,p):
    ll=0.;n=0
    for q in seqs:
        z=np.asarray(q,int);ll+=float(np.log(p[z]).sum());n+=len(z)
    return ll/max(1,n)

def fit_mk1(seqs):
    C=np.full((A,A),0.5,float)
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):C[int(x),int(y)]+=1
    return C/C.sum(1,keepdims=True)

def score_mk1(seqs,P):
    ll=0.;n=0
    for q in seqs:
        z=np.asarray(q,int)
        for x,y in zip(z,z[1:]):ll+=math.log(float(P[int(x),int(y)]));n+=1
    return ll/max(1,n)

def main():
    folios,labs,meta=v6.target_folios();folios,labs=v6.balanced_hash_order(folios,labs,6);rows=[]
    for k in range(6):
        tr=[f for i,f in enumerate(folios) if i%6!=k];ho=[f for i,f in enumerate(folios) if i%6==k]
        trseq=v6.flatten_folios(tr);hoseq=v6.flatten_folios(ho)
        pu=fit_uni(trseq);pm=fit_mk1(trseq)
        ou=score_uni(hoseq,pu);om=score_mk1(hoseq,pm);ups=[];mps=[]
        for r in range(24):
            hp=v6.permuted(ho,f'AUDIT:F{k}',r);hs=v6.flatten_folios(hp)
            ups.append(score_uni(hs,pu));mps.append(score_mk1(hs,pm))
        row={'fold':k,'unigram_obs':ou,'unigram_perm_median':statistics.median(ups),'unigram_ITE':ou-statistics.median(ups),
             'markov1_obs':om,'markov1_perm_median':statistics.median(mps),'markov1_ITE':om-statistics.median(mps)}
        rows.append(row);print('AUDIT',json.dumps(row,sort_keys=True),flush=True)
    out={'stage':'POSTHOC_FIT_ONLY','rows':rows,
         'median_unigram_ITE':statistics.median(x['unigram_ITE'] for x in rows),
         'median_markov1_ITE':statistics.median(x['markov1_ITE'] for x in rows)}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
