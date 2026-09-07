#!/usr/bin/env python3
"""Implementation-equivalent memoized C1 for vms_textual_linkage_20260907_v03.
No scientific or threshold changes; cross-bifolium target remains sealed.
"""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import vms_textual_linkage_v03_c1 as b
from vms_topology_marginalization import CORPORA, UNITS, fetch_text, parse_ivtff

OUT=Path('artifacts/vms_textual_linkage_v03_c1_fast');OUT.mkdir(parents=True,exist_ok=True)

def panel(body,code):
    ft,fm,_,_=parse_ivtff(body);lines=b.parse_lines(body);weights=b.motif_defs(lines)
    strata=defaultdict(list);actual_by=defaultdict(list)
    for uid,q,a,c in UNITS:
        if a not in lines or c not in lines:continue
        ka,kc=b.key(a,fm),b.key(c,fm)
        if ka==kc and all(v is not None for v in ka):
            sk=(q,)+ka;strata[sk].extend((a,c));actual_by[sk].append((a,c))
    eligible={sk:fs for sk,fs in strata.items() if len(fs)>=4}
    actual=[p for sk,ps in actual_by.items() if sk in eligible for p in ps]
    folios=sorted({f for fs in eligible.values() for f in fs})
    # Exact observed pair score matrix; identical score() function.
    M={}
    for i,a in enumerate(folios):
        for c in folios[i+1:]:M[(a,c)]=b.score(lines[a],lines[c],weights)[0]
    def sc(a,c):return M[tuple(sorted((a,c)))]
    obsvals=[];perm=[]
    for i,(a,c) in enumerate(actual):
        s,d=b.score(lines[a],lines[c],weights);obsvals.append(s)
        pz=b.perm_z(lines[a],lines[c],weights,b.SEED+1000*i+sum(map(ord,code)))
        perm.append({'a':a,'b':c,'score':s,'shared_bigram':d[2],'shared_trigram':d[3],**pz})
    obs=float(np.mean(obsvals));rng=random.Random(b.SEED+sum(map(ord,code)));null=[]
    for _ in range(b.N_REPAIR):
        vals=[]
        for sk,fs0 in eligible.items():
            fs=fs0[:];rng.shuffle(fs)
            vals.extend(sc(fs[i],fs[i+1]) for i in range(0,len(fs),2))
        null.append(float(np.mean(vals)))
    nm=float(np.mean(null));ns=float(np.std(null,ddof=1));z=(obs-nm)/ns if ns else float('nan');p=(1+sum(v>=obs for v in null))/(len(null)+1)
    finite=[x['z'] for x in perm if math.isfinite(x['z'])];meanpz=float(np.mean(finite)) if finite else float('nan');pos=sum(math.isfinite(x['z']) and x['z']>0 for x in perm)/len(perm)
    qualified=bool(z>=2 and p<=.01 and meanpz>0 and pos>=.75)
    return {'code':code,'n_actual_pairs':len(actual),'n_strata':len(eligible),'observed_mean_score':obs,'repair_null_mean':nm,'repair_null_sd':ns,'effect':obs-nm,'effect_over_null_sd':z,'repair_p':p,'permutation_mean_z':meanpz,'permutation_positive_fraction':pos,'qualified':qualified,'pair_details':perm,'rare_motif_counts':{'bigrams':len(weights[2]),'trigrams':len(weights[3])}}

def main():
    rs=[]
    for code,(url,sha) in CORPORA.items():
        r=panel(fetch_text(url,sha),code);rs.append(r);print(code,{k:v for k,v in r.items() if k!='pair_details'},flush=True)
    n=sum(r['qualified'] for r in rs);passed=n>=3
    out={'protocol':b.PROTOCOL,'stage':'C1','implementation':'memoized_equivalent','target_opened':False,'n_repairings':b.N_REPAIR,'n_within_line_permutations':b.N_DISC,'results':rs,'n_qualified_transcriptions':n,'stage_pass':passed}
    (OUT/'c1_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# v0.3 C1 rare-sequence calibration — memoized equivalent','', 'Cross-bifolium target graph opened: **NO**','']
    for r in rs:lines.append(f"- {r['code']}: actual {r['observed_mean_score']:.5f} vs re-pair null {r['repair_null_mean']:.5f} ± {r['repair_null_sd']:.5f}; effect {r['effect']:+.5f} = **{r['effect_over_null_sd']:.3f} SD**, p={r['repair_p']:.5g}; permutation mean z={r['permutation_mean_z']:.3f}, positive={r['permutation_positive_fraction']:.3f}; **{'PASS' if r['qualified'] else 'FAIL'}**")
    lines+=['',f"Qualified {n}/4. **C1 {'PASS' if passed else 'FAIL'}**"]
    (OUT/'C1_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
