#!/usr/bin/env python3
"""C1 known-answer calibration for vms_textual_linkage_20260907_v03.

No cross-bifolium target graph is computed.  Tests exact within-line word
bigrams/trigrams on real conjoint folio halves against same-quire + same L/H/I
re-pairings, with within-line token-order permutation enrichment.
"""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from vms_topology_marginalization import CORPORA, UNITS, fetch_text, clean_tokens, parse_ivtff

PROTOCOL='vms_textual_linkage_20260907_v03'
OUT=Path('artifacts/vms_textual_linkage_v03_c1'); OUT.mkdir(parents=True,exist_ok=True)
SEED=2026090722
N_DISC=128
N_REPAIR=10000
LINE_RE=re.compile(r'^<(f\d+[rv]\d*)\.\d+,[^>]*>\s*(.*)$')

def folio(page):
    m=re.match(r'f(\d+)',page); return int(m.group(1))

def parse_lines(body):
    lines=defaultdict(list)
    for raw in body.splitlines():
        m=LINE_RE.match(raw)
        if not m: continue
        p,txt=m.groups(); toks=clean_tokens(txt)
        if toks: lines[folio(p)].append(toks)
    return dict(lines)

def ngrams(lines,n):
    c=Counter()
    for ln in lines:
        for i in range(len(ln)-n+1): c[tuple(ln[i:i+n])]+=1
    return c

def motif_defs(lines_by_folio):
    dfs={2:Counter(),3:Counter()}
    for f,ls in lines_by_folio.items():
        for n in (2,3):
            dfs[n].update(set(ngrams(ls,n)))
    N=len(lines_by_folio); weights={}
    for n in (2,3):
        weights[n]={m:math.log((N+1)/(df+.5)) for m,df in dfs[n].items() if 2<=df<=5}
    return weights

def score(a,b,weights):
    out=0.0; detail={}
    for n,mult in ((2,1.0),(3,2.0)):
        A=set(ngrams(a,n));B=set(ngrams(b,n));shared=A&B&weights[n].keys()
        s=sum(weights[n][m] for m in shared)*mult;out+=s;detail[n]=len(shared)
    return out,detail

def shuffled_lines(ls,rng):
    out=[]
    for ln in ls:
        x=ln[:];rng.shuffle(x);out.append(x)
    return out

def perm_z(a,b,weights,seed,nperm=N_DISC):
    obs,_=score(a,b,weights);rng=random.Random(seed);vals=[]
    for _ in range(nperm): vals.append(score(shuffled_lines(a,rng),shuffled_lines(b,rng),weights)[0])
    m=float(np.mean(vals));sd=float(np.std(vals,ddof=1));z=(obs-m)/sd if sd>0 else (float('inf') if obs>m else 0.0)
    p=(1+sum(v>=obs for v in vals))/(len(vals)+1)
    return {'observed':obs,'null_mean':m,'null_sd':sd,'effect':obs-m,'z':z,'p':p}

def key(f,fm):
    x=fm.get(f,{})
    return (x.get('L'),x.get('H'),x.get('I'))

def panel(body,code):
    ft,fm,_,_=parse_ivtff(body);lines=parse_lines(body);weights=motif_defs(lines)
    strata=defaultdict(list); actual_by=defaultdict(list)
    for uid,q,a,b in UNITS:
        if a not in lines or b not in lines: continue
        ka,kb=key(a,fm),key(b,fm)
        if ka==kb and all(v is not None for v in ka):
            sk=(q,)+ka;strata[sk].extend((a,b));actual_by[sk].append((a,b))
    eligible={sk:fs for sk,fs in strata.items() if len(fs)>=4}
    actual=[p for sk,ps in actual_by.items() if sk in eligible for p in ps]
    pair_obs=[];perm=[]
    for i,(a,b) in enumerate(actual):
        s,d=score(lines[a],lines[b],weights);pair_obs.append(s)
        z=perm_z(lines[a],lines[b],weights,SEED+1000*i+sum(map(ord,code)))
        perm.append({'a':a,'b':b,'score':s,'shared_bigram':d[2],'shared_trigram':d[3],**z})
    obs=float(np.mean(pair_obs));rng=random.Random(SEED+sum(map(ord,code)));null=[]
    for _ in range(N_REPAIR):
        vals=[]
        for sk,fs0 in eligible.items():
            fs=fs0[:];rng.shuffle(fs)
            for i in range(0,len(fs),2): vals.append(score(lines[fs[i]],lines[fs[i+1]],weights)[0])
        null.append(float(np.mean(vals)))
    nm=float(np.mean(null));ns=float(np.std(null,ddof=1));z=(obs-nm)/ns if ns else float('nan');p=(1+sum(v>=obs for v in null))/(len(null)+1)
    pos=sum(math.isfinite(x['z']) and x['z']>0 for x in perm)/len(perm)
    mean_pz=float(np.mean([x['z'] for x in perm if math.isfinite(x['z'])]))
    qualified=bool(z>=2 and p<=.01 and mean_pz>0 and pos>=.75)
    return {'code':code,'n_actual_pairs':len(actual),'n_strata':len(eligible),'observed_mean_score':obs,'repair_null_mean':nm,'repair_null_sd':ns,'effect':obs-nm,'effect_over_null_sd':z,'repair_p':p,'permutation_mean_z':mean_pz,'permutation_positive_fraction':pos,'qualified':qualified,'pair_details':perm,
            'rare_motif_counts':{'bigrams':len(weights[2]),'trigrams':len(weights[3])}}

def main():
    rs=[]
    for code,(url,sha) in CORPORA.items():
        r=panel(fetch_text(url,sha),code);rs.append(r);print(code,{k:v for k,v in r.items() if k!='pair_details'},flush=True)
    n=sum(r['qualified'] for r in rs);passed=n>=3
    out={'protocol':PROTOCOL,'stage':'C1','target_opened':False,'n_repairings':N_REPAIR,'n_within_line_permutations':N_DISC,'results':rs,'n_qualified_transcriptions':n,'stage_pass':passed,'gate':'>=3/4: conjoint panel >=2 re-pair null SD, p<=.01, mean pair permutation z>0, >=75% pair z positive'}
    (OUT/'c1_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# v0.3 C1 rare-sequence functional-unit calibration','', 'Cross-bifolium target graph opened: **NO**','']
    for r in rs:
        lines.append(f"- {r['code']}: actual {r['observed_mean_score']:.5f} vs re-pair null {r['repair_null_mean']:.5f} ± {r['repair_null_sd']:.5f}; effect {r['effect']:+.5f} = **{r['effect_over_null_sd']:.3f} SD**, p={r['repair_p']:.5g}; within-line permutation mean z={r['permutation_mean_z']:.3f}, positive={r['permutation_positive_fraction']:.3f}; **{'PASS' if r['qualified'] else 'FAIL'}**")
    lines+=['',f"Qualified {n}/4. **C1 {'PASS' if passed else 'FAIL'}**",'', 'C1 FAIL => v0.3 target remains sealed.']
    (OUT/'C1_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
