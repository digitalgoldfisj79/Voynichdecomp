#!/usr/bin/env python3
"""Fixed-token rarefaction control for VMS bifolium unit effect.

Follow-up frozen after strict quire+L+H+I confound control, before viewing this
assay's target outcomes. Purpose: eliminate combined-pair token-count/sample-size
as an explanation for lower H1/TTR or higher cosine on true conjoint bifolia.

Each eligible folio is independently subsampled without replacement to K=100
word tokens. Within each of 64 rarefaction replicates, the true bifolium score
is compared with 512 random re-pairings preserving current quire + IVTFF $L
Currier + $H hand + $I broad section. Report the hierarchical pooled effect and
the distribution of per-replicate z scores.
"""
from __future__ import annotations
import json, math, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from vms_topology_marginalization import CORPORA, SEED, fetch_text, parse_ivtff, unit_maps, entropy, cosine

OUT=Path('artifacts/vms_topology_unit_rarefaction_v01'); OUT.mkdir(parents=True,exist_ok=True)
K=100; R=64; P=512

def key(f,fm):
    m=fm.get(f,{})
    return (m.get('L'),m.get('H'),m.get('I'))

def ttr(x): return len(set(x))/len(x) if x else float('nan')
def metrics(pairs,toks):
    h=[]; tt=[]; co=[]
    for a,b in pairs:
        z=toks[a]+toks[b]
        h.append(entropy(z)); tt.append(ttr(z)); co.append(cosine(toks[a],toks[b]))
    return {'H1':float(np.mean(h)),'TTR':float(np.mean(tt)),'pair_cosine':float(np.mean(co))}

def summarize(obs_by_rep,null_by_rep):
    # Pooled matched hierarchy: compare each null value to the observed value from
    # its own rarefaction replicate by storing differences. Effect = mean(obs-null).
    diffs=[]; obs=[]; nul=[]; zs=[]
    for o,ns in zip(obs_by_rep,null_by_rep):
        a=np.asarray(ns,float); m=float(a.mean()); sd=float(a.std(ddof=1));
        obs.append(o); nul.extend(ns); diffs.extend([o-x for x in ns]);
        zs.append((o-m)/sd if sd>0 else float('nan'))
    d=np.asarray(diffs,float); # d >0 means observed larger than matched null
    # Null SD for reporting is the RMS/pooled within-replicate null SD, not SD of differences.
    within_sds=[float(np.std(ns,ddof=1)) for ns in null_by_rep]
    null_sd=float(np.sqrt(np.mean(np.square(within_sds))))
    effect=float(np.mean(d)); z=effect/null_sd if null_sd else float('nan')
    # Empirical two-sided paired permutation position through centered within-rep nulls.
    centered=[]
    for o,ns in zip(obs_by_rep,null_by_rep):
        a=np.asarray(ns,float); centered.extend((a-a.mean()).tolist())
    centered=np.asarray(centered,float)
    p=float((1+np.sum(np.abs(centered)>=abs(effect)))/(1+len(centered)))
    return {'observed_mean':float(np.mean(obs)),'null_mean':float(np.mean(nul)),'effect':effect,'null_sd':null_sd,'effect_over_null_sd':z,'p_empirical':p,
            'per_rep_z_median':float(np.nanmedian(zs)),'per_rep_z_q10':float(np.nanquantile(zs,.1)),'per_rep_z_q90':float(np.nanquantile(zs,.9)),
            'n_rep_z_abs_ge2':int(np.sum(np.abs(zs)>=2)),'n_rarefaction':R,'n_pairings_each':P}

def run(code,url,sha,seed):
    ft,fm,_,_=parse_ivtff(fetch_text(url,sha)); _,byq,_=unit_maps(ft)
    strata=defaultdict(list); actuals=defaultdict(list)
    for q,us in byq.items():
        for u in us:
            if not u['complete']: continue
            a,b=u['folios']; ka,kb=key(a,fm),key(b,fm)
            if ka==kb and all(x is not None for x in ka) and len(ft[a])>=K and len(ft[b])>=K:
                sk=(q,)+ka; actuals[sk].append((a,b)); strata[sk].extend((a,b))
    eligible={sk:xs for sk,xs in strata.items() if len(xs)>=4}
    actual=[p for sk,ps in actuals.items() if sk in eligible for p in ps]
    folios=sorted({f for p in actual for f in p})
    rng=random.Random(seed)
    obs={k:[] for k in ('H1','TTR','pair_cosine')}; null={k:[] for k in obs}
    for rr in range(R):
        toks={}
        for f in folios:
            idx=rng.sample(range(len(ft[f])),K); idx.sort(); toks[f]=[ft[f][i] for i in idx]
        o=metrics(actual,toks)
        for k in obs: obs[k].append(o[k]); null[k].append([])
        for _ in range(P):
            pp=[]
            for sk,xs0 in eligible.items():
                xs=xs0[:]; rng.shuffle(xs); pp.extend((xs[i],xs[i+1]) for i in range(0,len(xs),2))
            m=metrics(pp,toks)
            for k in null: null[k][-1].append(m[k])
    return {'code':code,'K':K,'n_actual_pairs':len(actual),'n_folios':len(folios),'n_strata':len(eligible),'metrics':{k:summarize(obs[k],null[k]) for k in obs}}

def main():
    rs=[]
    for i,(code,(url,sha)) in enumerate(CORPORA.items()):
        r=run(code,url,sha,SEED+12000+i); rs.append(r); print(code,r['metrics'],flush=True)
    classes={}
    for m in ('H1','TTR','pair_cosine'):
        z={r['code']:r['metrics'][m]['effect_over_null_sd'] for r in rs}; pos=sum(v>=2 for v in z.values()); neg=sum(v<=-2 for v in z.values())
        status='TOPOLOGY_DEPENDENT_POSITIVE' if pos>=3 else ('TOPOLOGY_DEPENDENT_NEGATIVE' if neg>=3 else ('REPRESENTATION_DEPENDENT' if any(abs(v)>=2 for v in z.values()) else 'UNRESOLVED'))
        classes[m]={'status':status,'z_by_code':z,'n_pos_ge2':pos,'n_neg_le_minus2':neg}
    out={'protocol':'vms_topology_marginal_20260907_v01','control':'fixed_100_tokens_per_folio_plus_same_quire_L_H_I_repairing','results':rs,'classes':classes,
         'decision_rule':'<2 matched-null SD => metric does not resolve; cross-representation promotion >=3/4 same direction'}
    (OUT/'unit_rarefaction_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Fixed-token bifolium rarefaction closeout','',f'K={K} tokens per folio; {R} rarefactions; {P} strict re-pairings per rarefaction.','']
    for m,c in classes.items(): lines.append(f"- {m}: **{c['status']}**; z={c['z_by_code']}")
    (OUT/'UNIT_RAREFACTION_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
