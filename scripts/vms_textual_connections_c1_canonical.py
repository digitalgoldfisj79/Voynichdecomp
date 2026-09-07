#!/usr/bin/env python3
"""Canonical rerun of C1 for vms_textual_connections_20260907_v01.

This does NOT change the frozen protocol or thresholds. It fixes two audit issues
in the first implementation only:
1) Gutenberg byte packaging drift: compare the normalized whole-text token
   stream against the SHA-256 frozen from the Supabase control body.
2) Python hash-randomization in the null RNG: replace hash(family) with a fixed
   family seed map.

Voynich target connections remain sealed.
"""
from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path
import numpy as np, requests
import vms_textual_connections_c1 as base

EXPECTED_NORM_SHA='bd1eaaed049ccdaad43725859c406f39111058cf4c289be341409f799d5d1a7d'
EXPECTED_WORDS=20561
FAM_SEED={'RARE':11,'SUBWORD':23,'MORPH':37}
OUT=Path('artifacts/vms_textual_connections_c1_canonical_v01'); OUT.mkdir(parents=True,exist_ok=True)


def fetch_caesar_canonical():
    r=requests.get(base.CAESAR_URL,timeout=60,headers={'User-Agent':'VoynichTextConnections/0.1-canonical'}); r.raise_for_status()
    raw=r.content; txt=raw.decode('utf-8-sig',errors='replace')
    toks=base.strict_words(txt)
    norm=' '.join(toks)
    h=hashlib.sha256(norm.encode()).hexdigest()
    if len(toks)!=EXPECTED_WORDS or h!=EXPECTED_NORM_SHA:
        raise RuntimeError(f'FROZEN_CONTROL_IDENTITY_MISMATCH words={len(toks)} norm_sha={h} expected_words={EXPECTED_WORDS} expected_sha={EXPECTED_NORM_SHA}')
    return toks,{'raw_sha256':hashlib.sha256(raw).hexdigest(),'normalized_sha256':h,'registered_normalized_sha256':EXPECTED_NORM_SHA,'token_count':len(toks),'url':base.CAESAR_URL}


def score_window_fixed(units,fam,defs):
    n=len(units); D=np.full((n,n),-1e99,float)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            s,_=base.direct_score(units[i],units[j],fam,defs); D[i,j]=s
    U=np.maximum(D,D.T)
    pred=set()
    for i in range(n):
        js=[j for j in range(n) if j!=i]; j=max(js,key=lambda q:U[i,q]); pred.add(tuple(sorted((i,j))))
    truth={(i,i+1) for i in range(n-1)}
    tp=len(pred&truth); prec=tp/len(pred) if pred else 0.; rec=tp/len(truth); fpr=1-prec
    dirs=[1 if D[i,j]>D[j,i] else 0 for i,j in pred&truth]
    diracc=float(np.mean(dirs)) if dirs else float('nan')
    obs=float(np.mean([U[i,j] for i,j in truth])); pool=[U[i,j] for i in range(n) for j in range(i+1,n) if (i,j) not in truth]
    rng=random.Random(base.SEED+1000+sum(len(x[0]) for x in units)+FAM_SEED[fam])
    null=[float(np.mean(rng.sample(pool,len(truth)))) for _ in range(base.N_NULL)]
    m=float(np.mean(null)); sd=float(np.std(null,ddof=1)); z=(obs-m)/sd if sd else float('nan')
    return {'pred':sorted([list(x) for x in pred]),'truth':sorted([list(x) for x in truth]),'tp':tp,'precision':prec,'recall':rec,'false_edge_rate':fpr,'direction_accuracy':diracc,
            'boundary_mean':obs,'null_mean':m,'null_sd':sd,'effect':obs-m,'effect_over_null_sd':z,'boundary_pass':bool(z>=2)}


def main():
    caesar,src=fetch_caesar_canonical(); lens=base.vms_page_lengths(); pages=base.make_pages(caesar,lens)
    units=[pages[i:i+4] for i in range(0,len(pages),4)]
    if len(units)<base.WINDOW_UNITS+2: raise RuntimeError('too few pseudo-units')
    defs=base.build_defs(caesar); windows=[]
    for st in range(0,len(units)-base.WINDOW_UNITS+1,base.STEP_UNITS):
        us=units[st:st+base.WINDOW_UNITS]
        fr={fam:score_window_fixed(us,fam,defs) for fam in base.FAMS}; co=base.combine(fr)
        windows.append({'start_unit':st,'families':fr,'combined':co})
    agg={}
    for fam in base.FAMS:
        rr=[w['families'][fam] for w in windows]
        d=[r['direction_accuracy'] for r in rr if math.isfinite(r['direction_accuracy'])]
        zpass=sum(r['boundary_pass'] for r in rr)/len(rr)
        q={'mean_recall':float(np.mean([r['recall'] for r in rr])),'mean_false_edge_rate':float(np.mean([r['false_edge_rate'] for r in rr])),
           'direction_accuracy':float(np.mean(d)) if d else float('nan'),'boundary_pass_fraction':zpass}
        q['qualified']=bool(q['mean_recall']>=.70 and q['mean_false_edge_rate']<=.20 and q['direction_accuracy']>=.70 and q['boundary_pass_fraction']>=.75)
        agg[fam]=q
    crr=[w['combined'] for w in windows]
    comb={'mean_recall':float(np.mean([x['recall'] for x in crr])),'mean_false_edge_rate':float(np.mean([x['false_edge_rate'] for x in crr]))}
    nqual=sum(agg[f]['qualified'] for f in base.FAMS); comb['n_qualified_families']=nqual
    comb['qualified']=bool(nqual>=2 and comb['mean_recall']>=.75 and comb['mean_false_edge_rate']<=.15)
    out={'protocol':base.PROTOCOL,'stage':'C1_CANONICAL','target_connections_opened':False,'source':src,'vms_page_length_count':len(lens),'pseudo_pages':len(pages),'pseudo_units':len(units),'n_windows':len(windows),'family_aggregate':agg,'combined':comb,
         'audit_fixes':['normalized frozen control identity','deterministic family null seeds'],'gate':'UNCHANGED FROM PREREGISTRATION'}
    (OUT/'c1_canonical.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Pure-text connection C1 canonical closeout','', 'Target connections opened: **NO**','',f"Frozen normalized Caesar SHA-256: `{src['normalized_sha256']}`",f"Pseudo-units: {len(units)}; windows: {len(windows)}",'']
    for f,q in agg.items(): lines.append(f"- {f}: recall {q['mean_recall']:.3f}; false-edge {q['mean_false_edge_rate']:.3f}; direction {q['direction_accuracy']:.3f}; boundary>=2SD windows {q['boundary_pass_fraction']:.3f}; **{'PASS' if q['qualified'] else 'FAIL'}**")
    lines += ['',f"Combined: recall {comb['mean_recall']:.3f}; false-edge {comb['mean_false_edge_rate']:.3f}; qualified families {nqual}/3; **{'PASS' if comb['qualified'] else 'FAIL'}**",'', 'If FAIL, Voynich target remains sealed and v0.1 closes negative by preregistration.']
    (OUT/'C1_CANONICAL_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'source':src,'family_aggregate':agg,'combined':comb,'n_windows':len(windows),'pseudo_units':len(units)},indent=2),flush=True)

if __name__=='__main__': main()
