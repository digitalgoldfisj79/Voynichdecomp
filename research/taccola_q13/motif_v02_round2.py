#!/usr/bin/env python3
"""Taccola v0.2 development round 2: one principled Hu+topology local-shape composite.

Uses only exposed v0.1b feature artifacts. No holdout images/features and no Q13 access.
This is the final development round before either freezing v0.2 or abandoning the visual instrument.
"""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
import taccola_calibration_v01b as core
from motif_v02_develop import motif_masks,descs,host_uplift

VERSION='taccola-q13-v0.2-local-shape-round2-20260830'
KS=[12,20,30]
BSB_NULL_IDS=[k for k,v in core.MANUSCRIPTS.items() if v['role']=='control' and 'digitale-sammlungen.de' in v['manifest']]


def sim_matrix(A,B,scales):
    ah=np.vstack([x['hu'] for x in A]); bh=np.vstack([x['hu'] for x in B]); at=np.vstack([x['topology'] for x in A]); bt=np.vstack([x['topology'] for x in B])
    mh,sh=scales['hu']; mt,st=scales['topology']
    ah=(ah-mh)/sh; bh=(bh-mh)/sh; at=(at-mt)/st; bt=(bt-mt)/st
    dh=((ah[:,None,:]-bh[None,:,:])**2).mean(2)
    dt=((at[:,None,:]-bt[None,:,:])**2).mean(2)
    return np.exp(-np.sqrt(.5*dh+.5*dt)).astype(np.float32)


def set_score(M,k):
    a=M.max(0); b=M.max(1); ka=min(k,len(a)); kb=min(k,len(b))
    return .5*(float(np.mean(np.sort(a)[-ka:]))+float(np.mean(np.sort(b)[-kb:])))


def stats(pos,scores,ids):
    v=np.array([scores[k] for k in ids],float); mu=v.mean(); sd=v.std(ddof=1)
    return {'positive':float(pos),'null_mean':float(mu),'null_sd':float(sd),'effect':float(pos-mu),'z':float((pos-mu)/(sd+1e-9)),'p':float((1+np.sum(v>=pos))/(1+len(v))),'rank':int(1+np.sum(v>pos))}


def uplift(scores,ids):
    b=[scores[k] for k in ids if k in BSB_NULL_IDS]; o=[scores[k] for k in ids if k not in BSB_NULL_IDS]; sd=np.std([scores[k] for k in ids],ddof=1)
    return float((np.mean(b)-np.mean(o))/(sd+1e-9))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='dev_inputs'); args=ap.parse_args()
    bundles={}
    for p in Path(args.input).rglob('one_*.pkl'):
        with open(p,'rb') as f:b=pickle.load(f)
        if b['panel_sha256']!=core.EXPECTED_PANEL_SHA256:raise RuntimeError('panel hash mismatch')
        bundles[b['id']]=b
    if sorted(bundles)!=sorted(core.MANUSCRIPTS):raise RuntimeError('coverage mismatch')
    motifs={}
    for mid,b in bundles.items():
        mm=[]
        for pg in b['features']:
            for mask in motif_masks(pg['shape']):mm.append(descs(mask))
        motifs[mid]=mm
    scales={}
    for key in ['hu','topology']:
        X=np.vstack([m[key] for mid in motifs for m in motifs[mid]]); mu=X.mean(0); sd=X.std(0); sd=np.where(sd<1e-6,1,sd); scales[key]=(mu,sd)
    nullids=list(core.NULL_IDS); dirs=[('clm197','pal766'),('pal766','clm197')]
    out={'version':VERSION,'q13_accessed':False,'holdout_visual_access':False,'representation':'equal-block standardized Hu moments + skeleton/topology geometry','k_sensitivity':KS,'directions':{}}
    for a,b in dirs:
        dk=f'{a}_to_{b}'; out['directions'][dk]={}
        mats={cid:sim_matrix(motifs[a],motifs[cid],scales) for cid in [b]+nullids+['dresden_mixed']}
        for k in KS:
            scores={cid:set_score(M,k) for cid,M in mats.items()}
            st=stats(scores[b],scores,nullids); tech=stats(scores[b],scores,core.TECH_IDS)
            deriv_rank=1+sum(scores[x]>scores['dresden_mixed'] for x in nullids)
            out['directions'][dk][str(k)]={'stats':st,'technical':tech,'bsb_uplift_sd':uplift(scores,nullids),'derivative_rank_vs_null':int(deriv_rank),'derivative_score':float(scores['dresden_mixed']),'scores':scores}
    # Freeze-eligibility development rule, declared before this run:
    # k=20 z>=1.5 both dirs, rank<=3, technical rank<=2, host uplift<=0.5;
    # all k=12/20/30 remain positive with z>=1.25 and rank<=5.
    primary=[]; sens=[]
    for a,b in dirs:
        d=out['directions'][f'{a}_to_{b}']; p=d['20']
        primary.append(p['stats']['z']>=1.5 and p['stats']['rank']<=3 and p['technical']['rank']<=2 and p['bsb_uplift_sd']<=.5)
        sens.append(all(d[str(k)]['stats']['z']>=1.25 and d[str(k)]['stats']['rank']<=5 for k in KS))
    eligible=all(primary) and all(sens)
    out['freeze_eligibility']={'eligible':bool(eligible),'primary_rule':'k20 z>=1.5 both directions; overall rank<=3; technical rank<=2; BSB uplift<=0.5 SD','sensitivity_rule':'k12/20/30 z>=1.25 and rank<=5 both directions','meaning':'Eligibility permits a frozen held-out validation protocol only; it does not validate Taccola-Q13 and does not permit Q13 access.'}
    od=Path('taccola_v02_round2');od.mkdir(exist_ok=True);(od/'round2.json').write_text(json.dumps(out,indent=2))
    lines=['# Taccola v0.2 local-shape development round 2','','## RETRACTED FINDINGS','','- **RETRACTED:** the family-collapsed Siena concentration is unresolved (Fisher p=0.318).','','No Q13 or holdout visual material was accessed.','',f"Freeze eligible: `{eligible}`",'']
    for a,b in dirs:
        dk=f'{a}_to_{b}'; lines.append(f'## {dk}')
        for k in KS:
            x=out['directions'][dk][str(k)]; s=x['stats'];t=x['technical'];lines.append(f"- k={k}: z={s['z']:.3f}, p={s['p']:.4f}, rank={s['rank']}/35, technical rank={t['rank']}/7, BSB uplift={x['bsb_uplift_sd']:.2f} SD, Dresden derivative rank={x['derivative_rank_vs_null']}/35")
    (od/'ROUND2.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event':'v02_round2_done','freeze_eligible':eligible,'primary_z':{f'{a}_to_{b}':out['directions'][f'{a}_to_{b}']['20']['stats']['z'] for a,b in dirs}},sort_keys=True),flush=True)
if __name__=='__main__':main()
