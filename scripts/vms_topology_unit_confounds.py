#!/usr/bin/env python3
"""Strict bifolium unit confound control.

Re-tests unit-level H1/TTR/hapax/cosine using only re-pairings that preserve,
within the same current quire, the exact IVTFF Currier-language ($L), hand ($H)
and broad illustration/section ($I) stratum of each folio.

This is a follow-up forced by the first topology-marginal run: actual conjoint
pairs were 100% Currier-coherent, so a quire-only null is not sufficient for
claiming independent bifolium effects.
"""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np

from vms_topology_marginalization import (
    CORPORA, UNITS, SEED, fetch_text, parse_ivtff, unit_maps, lex, cosine,
    summarize_null
)

OUT=Path("artifacts/vms_topology_unit_confounds_v01"); OUT.mkdir(parents=True,exist_ok=True)
N=10000

def key(f,fm):
    m=fm.get(f,{})
    return (m.get('L'),m.get('H'),m.get('I'))

def pair_metrics(pairs,ft):
    rows=[]
    for a,b in pairs:
        z=lex(ft[a]+ft[b]); z['pair_cosine']=cosine(ft[a],ft[b]); rows.append(z)
    out={}
    for k in rows[0]: out[k]=float(np.mean([r[k] for r in rows]))
    return out

def run(code,url,sha,seed):
    ft,fm,_,_=parse_ivtff(fetch_text(url,sha)); _,byq,_=unit_maps(ft)
    # Build strata only from actual complete, same-stratum pairs. A stratum must
    # contain >=2 actual pairs (>=4 folios) to contribute a nontrivial re-pair null.
    strata=defaultdict(list); actual_by_stratum=defaultdict(list)
    for q,us in byq.items():
        for u in us:
            if not u['complete']: continue
            a,b=u['folios']; ka,kb=key(a,fm),key(b,fm)
            if ka==kb and all(x is not None for x in ka):
                sk=(q,)+ka; actual_by_stratum[sk].append((a,b)); strata[sk].extend((a,b))
    eligible={sk:xs for sk,xs in strata.items() if len(xs)>=4}
    actual=[p for sk,ps in actual_by_stratum.items() if sk in eligible for p in ps]
    if not actual: raise RuntimeError('no eligible strict strata')
    obs=pair_metrics(actual,ft); rng=random.Random(seed); null={k:[] for k in obs}
    for _ in range(N):
        pp=[]
        for sk,xs0 in eligible.items():
            xs=xs0[:]; rng.shuffle(xs); pp.extend((xs[i],xs[i+1]) for i in range(0,len(xs),2))
        s=pair_metrics(pp,ft)
        for k,v in s.items(): null[k].append(v)
    return {
      'code':code,'n_actual_pairs':len(actual),'n_strata':len(eligible),
      'strata':[{'quire':sk[0],'L':sk[1],'H':sk[2],'I':sk[3],'n_folios':len(xs)} for sk,xs in sorted(eligible.items())],
      'metrics':{k:summarize_null(obs[k],null[k]) for k in obs}
    }

def main():
    results=[]
    for i,(code,(url,sha)) in enumerate(CORPORA.items()):
        r=run(code,url,sha,SEED+9000+i); results.append(r); print(code,r['n_actual_pairs'],r['n_strata'],r['metrics'],flush=True)
    classes={}
    for metric in results[0]['metrics']:
        z={r['code']:r['metrics'][metric]['z'] for r in results}
        pos=sum(v>=2 for v in z.values()); neg=sum(v<=-2 for v in z.values())
        status='TOPOLOGY_DEPENDENT_POSITIVE' if pos>=3 else ('TOPOLOGY_DEPENDENT_NEGATIVE' if neg>=3 else ('REPRESENTATION_DEPENDENT' if any(abs(v)>=2 for v in z.values()) else 'UNRESOLVED'))
        classes[metric]={'status':status,'z_by_code':z,'n_pos_ge2':pos,'n_neg_le_minus2':neg}
    out={'protocol':'vms_topology_marginal_20260907_v01','control':'same_quire_same_L_H_I_repairing','n_null':N,'results':results,'classes':classes,
         'rule':'<2 null SD => metric does not resolve; promotion requires >=3/4 same direction'}
    (OUT/'unit_confound_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Strict bifolium unit confound closeout','', 'Null preserves current quire + IVTFF Currier language + hand + broad section.', '']
    for m,c in classes.items(): lines.append(f"- {m}: **{c['status']}**; z={c['z_by_code']}")
    (OUT/'UNIT_CONFOUND_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
