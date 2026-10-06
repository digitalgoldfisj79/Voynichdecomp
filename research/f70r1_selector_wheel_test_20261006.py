#!/usr/bin/env python3
"""Test f70r1 against regular selector-wheel / Losbuch slot architectures.

Inputs are the persisted f70r1 outer-register component table and the
gap-segmentation sensitivity table. The test is deliberately independent
of historical target counts until the final comparison.

Tests:
1. harmonic coherence for candidate slot counts 22/28/30/32;
2. local-density-preserving nulls (randomize positions within 4/6/8/10/12° bins);
3. cross-validated slot phase RMS;
4. max-stat scan over every k=8..64;
5. representation sensitivity and natural gap-unit count stability.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import pandas as pd

SEED=20261006

def coherence(a,k):
    z=np.exp(1j*k*np.deg2rad(np.asarray(a))).mean()
    return float(abs(z)), float((np.rad2deg(np.angle(z))/k)%(360/k))

def phase_cv_rms(a,k):
    a=np.sort(np.asarray(a)); tr=a[::2]; te=a[1::2]
    p=360/k
    z=np.exp(1j*2*np.pi*(tr%p)/p).mean()
    ph=(np.angle(z)%(2*np.pi))*p/(2*np.pi)
    d=((te-ph+p/2)%p)-p/2
    return float(np.sqrt(np.mean(d*d)))

def bin_null(a,k,bw,reps,rng,metric='coh'):
    a=np.asarray(a); bins=np.floor(a/bw)
    if metric=='coh': obs=coherence(a,k)[0]
    else: obs=phase_cv_rms(a,k)
    vals=np.empty(reps)
    for i in range(reps):
        x=(bins+rng.random(len(a)))*bw
        vals[i]=coherence(x,k)[0] if metric=='coh' else phase_cv_rms(x,k)
    sd=float(vals.std(ddof=1))
    if metric=='coh':
        p=float((np.sum(vals>=obs)+1)/(reps+1))
    else:
        p=float((np.sum(vals<=obs)+1)/(reps+1))
    return dict(obs=float(obs),null_mean=float(vals.mean()),null_sd=sd,
                z=float((obs-vals.mean())/sd),p=p)

def maxstat(a,bw,reps,rng,kmin=8,kmax=64):
    vals={k:coherence(a,k)[0] for k in range(kmin,kmax+1)}
    bk=max(vals,key=vals.get); obs=vals[bk]
    bins=np.floor(np.asarray(a)/bw); null=np.empty(reps)
    for i in range(reps):
        x=(bins+rng.random(len(a)))*bw
        null[i]=max(coherence(x,k)[0] for k in range(kmin,kmax+1))
    sd=float(null.std(ddof=1))
    return dict(best_k=int(bk),obs=float(obs),null_mean=float(null.mean()),
                null_sd=sd,z=float((obs-null.mean())/sd),
                p_upper=float((np.sum(null>=obs)+1)/(reps+1)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--components',required=True)
    ap.add_argument('--gap-sensitivity',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--reps',type=int,default=20000)
    args=ap.parse_args()
    cc=pd.read_csv(args.components)
    a=cc[(cc.angle_deg>=228)&(cc.angle_deg<339)].angle_deg.to_numpy()
    gap=pd.read_csv(args.gap_sensitivity)
    rng=np.random.default_rng(SEED)
    Ks=[22,28,30,32]; out={'n':len(a),'arc':[228,339],'candidate_k':Ks,'specific':{}}
    for k in Ks:
        out['specific'][str(k)]={'coherence':{},'crossval_rms':{}}
        for bw in [4,6,8,10,12]:
            out['specific'][str(k)]['coherence'][str(bw)]=bin_null(a,k,bw,args.reps,rng,'coh')
            out['specific'][str(k)]['crossval_rms'][str(bw)]=bin_null(a,k,bw,args.reps//2,rng,'rms')
    out['maxstat_8_64']={}
    for bw in [4,6,8,10,12]:
        out['maxstat_8_64'][str(bw)]=maxstat(a,bw,args.reps,rng)
    # depth was redundant in the preceding extraction, so count unique centre/threshold settings
    uniq=gap.drop_duplicates(['cx','cy','thr'])
    out['natural_units']={
      'median':float(gap.nunits.median()),'range':[int(gap.nunits.min()),int(gap.nunits.max())],
      'target_hits_81':{str(k):int((gap.nunits==k).sum()) for k in Ks},
      'target_hits_27_unique':{str(k):int((uniq.nunits==k).sum()) for k in Ks}
    }
    out['decision']={
      'regular_22_28_30_32_selector':'reject if no candidate survives null/sensitivity',
      'any_regular_lattice_8_64':'evaluate max-stat null',
      'broader_nonmetric_routing':'not tested by equal-slot geometry alone'
    }
    with open(args.out,'w') as f: json.dump(out,f,indent=2)

if __name__=='__main__': main()
