#!/usr/bin/env python3
"""Sealed diagnostic for fractionation gate v0.1.

This script does NOT read Voynich text. It leaves the v0.1 decision rule
unchanged and asks two questions prompted by the 1/12 matched-null failure:

1. What is the empirical false-positive rate of the frozen v0.1 procedure
   (Z>=5, b>=2, eight internal positional shuffles) on a larger matched-null
   battery?
2. Was the original false-positive's Z=5.08 stable, or an artefact of
   estimating the null SD from only eight shuffles?

No Voynich exposure occurs here.
"""
from __future__ import annotations
import importlib.util, json, math, random, statistics, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'run_fractionation_gate_v01.py'
OUT=HERE/'results'
OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location('fracv01',BASE)
v=importlib.util.module_from_spec(spec); sys.modules['fracv01']=v
spec.loader.exec_module(v)


def wilson(k,n,z=1.959963984540054):
    if n==0:return [None,None]
    p=k/n; den=1+z*z/n
    cen=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,cen-half),min(1,cen+half)]


def matched_case(vname, case_id, j, ov, hom, nullp):
    # Seeds deliberately disjoint from v0.1's 30 positives / 12 nulls.
    seed=200000 + case_id*131
    true_b=2+((j+5*case_id)%7)
    words=v.markov_words(seed,n=500)
    tok=v.frac_encode(words,seed+1,true_b,ov,hom,nullp)
    q=v.positional_shuffle(tok,random.Random(seed+3))
    r=v.calibrated(q,seed+4,nsh=8)   # EXACT frozen v0.1 calibration
    return {'variant':vname,'true_source_b':true_b,**r}


def expanded_battery(n_per_variant=50):
    variants={'clean':(0.0,1,0.0),'hom':(.35,2,0.0),'null':(.35,2,.08)}
    rows=[]; cid=0
    for name,(ov,hom,np_) in variants.items():
        for j in range(n_per_variant):
            rows.append(matched_case(name,cid,j,ov,hom,np_)); cid+=1
    return rows


def original_false_positive_highres(nsh=512):
    # Reproduce v0.1 matched-null row 0 exactly.
    seed=10000; true_b=2
    words=v.markov_words(seed,n=500)
    tok=v.frac_encode(words,seed+1,true_b,0.0,1,0.0)
    q=v.positional_shuffle(tok,random.Random(seed+3))
    obs,_=v.scan(q)
    rng=random.Random(seed+4)
    vals=[]
    for _ in range(nsh):
        sh=v.positional_shuffle(q,rng)
        best,_=v.scan(sh)
        vals.append(best['score'])
    prefixes={}
    for m in (8,16,32,64,128,256,512):
        arr=vals[:m]; mu=statistics.mean(arr); sd=statistics.stdev(arr)
        z=(obs['score']-mu)/sd
        prefixes[str(m)]={'null_mean':mu,'null_sd':sd,'z':z,
                          'empirical_ge':sum(x>=obs['score'] for x in arr),
                          'empirical_tail':sum(x>=obs['score'] for x in arr)/m}
    return {'observed':obs,'prefix_calibration':prefixes,
            'max_null_score':max(vals),'min_null_score':min(vals)}


def main():
    rows=expanded_battery(50)
    k=sum(r['decision'] for r in rows); n=len(rows)
    byvar={}
    for name in ('clean','hom','null'):
        rr=[r for r in rows if r['variant']==name]
        kk=sum(r['decision'] for r in rr)
        byvar[name]={'n':len(rr),'false_positives':kk,'fpr':kk/len(rr),
                     'wilson95':wilson(kk,len(rr)),
                     'max_z':max(r['z'] for r in rr),
                     'b_counts':{str(b):sum(r['b']==b for r in rr) for b in v.B_VALUES}}
    high=original_false_positive_highres(512)
    out={'voynich_exposed':False,
         'frozen_rule':{'z_gate':v.Z_GATE,'b_gate':2,'inner_shuffles':8},
         'expanded_matched_null':{'n':n,'false_positives':k,'fpr':k/n,
                                  'wilson95':wilson(k,n),'by_variant':byvar,'rows':rows},
         'original_false_positive_highres':high}
    (OUT/'null_calibration_diagnostic_v01.json').write_text(json.dumps(out,indent=2))

    p512=high['prefix_calibration']['512']
    lines=['# Fractionation v0.1 sealed null-calibration diagnostic','',
           'Voynich exposure: **False**','',
           f"- Expanded matched nulls: {k}/{n} false positives = {k/n:.4f}",
           f"- Wilson 95% interval: [{out['expanded_matched_null']['wilson95'][0]:.4f}, {out['expanded_matched_null']['wilson95'][1]:.4f}]",
           f"- Original row-0 false positive recalibrated with 512 shuffles: Z={p512['z']:.4f}, null SD={p512['null_sd']:.6f}, empirical tail={p512['empirical_tail']:.4f}",
           '', '## By variant']
    for name,x in byvar.items():
        lines.append(f"- {name}: {x['false_positives']}/{x['n']} = {x['fpr']:.4f}; Wilson95=[{x['wilson95'][0]:.4f},{x['wilson95'][1]:.4f}]; max Z={x['max_z']:.3f}")
    lines += ['', '## Original false-positive calibration stability']
    for m,x in high['prefix_calibration'].items():
        lines.append(f"- nsh={m}: Z={x['z']:.4f}; null SD={x['null_sd']:.6f}; empirical >= observed {x['empirical_ge']}/{m}")
    (OUT/'NULL_CALIBRATION_DIAGNOSTIC_V01.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
