#!/usr/bin/env python3
"""Fractionation-composition falsification gate v0.2.

Reason for new version (pre-Voynich): v0.1 remained sealed because 1/12
matched nulls crossed Z>=5. A sealed 150-null diagnostic showed 0/150 FPs
and demonstrated that the original Z=5.08 collapsed to Z=2.29 when the
internal null SD was estimated with 512 rather than 8 shuffles. v0.2 fixes
that measurement instability BEFORE any Voynich scoring by using 64
internal shuffles throughout and a larger preflight null battery.

No scientific decision threshold is relaxed:
  candidate = (internally calibrated Z >= 5.0) AND (best block b >= 2)
Gate before Voynich exposure:
  planted TPR >= .90
  exact planted-b recovery >= .80
  position-matched-null FPR <= .05
  f57v Scribal Manual FPR <= .05
  P70-C Scribal generator FPR <= .05

All synthetic seeds are disjoint from v0.1 and its diagnostic.
"""
from __future__ import annotations
import importlib.util, json, random, statistics, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'; OUT.mkdir(parents=True,exist_ok=True)
BASE=HERE/'run_fractionation_gate_v01.py'
spec=importlib.util.spec_from_file_location('fracv01base',BASE)
v=importlib.util.module_from_spec(spec); sys.modules['fracv01base']=v
spec.loader.exec_module(v)

NSH=64
Z_GATE=5.0


def cal(tokens,seed):
    return v.calibrated(tokens,seed,nsh=NSH)


def synthetic_battery():
    variants={'clean':(0.0,1,0.0),'hom':(.35,2,0.0),'null':(.35,2,.08)}
    positives=[]; nulls=[]; cid=0
    for name,(ov,hom,np_) in variants.items():
        # 15 positives and 20 separately-seeded matched nulls per variant.
        for j in range(15):
            seed=400000+cid*149; true_b=2+((j+2*cid)%7)
            words=v.markov_words(seed,n=500)
            tok=v.frac_encode(words,seed+1,true_b,ov,hom,np_)
            positives.append({'variant':name,'true_b':true_b,**cal(tok,seed+2)})
            cid+=1
        for j in range(20):
            seed=500000+cid*157; true_b=2+((j+3*cid)%7)
            words=v.markov_words(seed,n=500)
            tok=v.frac_encode(words,seed+1,true_b,ov,hom,np_)
            q=v.positional_shuffle(tok,random.Random(seed+3))
            nulls.append({'variant':name,'source_b':true_b,**cal(q,seed+4)})
            cid+=1
    return positives,nulls


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m


def controls():
    slim=ROOT/'voynich_transcriptions_slim.json'
    g7=load_module('g7v02',ROOT/'Paper/Generators/gen_scribal_manual.py')
    gspec=g7.load_f57v_spec(str(slim)); grows=[]
    for i in range(50):
        toks=[list(t) for t in g7.produce_manuscript(gspec,n_tokens=600,seed=600000+i) if t]
        best,_=v.scan(toks)
        r=cal(toks,605000+i) if best['b']>=2 else {**best,'null_mean':None,'null_sd':None,'z':None,'decision':False}
        grows.append({'seed':600000+i,**r})

    gp=load_module('gpv02',ROOT/'Paper/Generators/gen_scribal_p70c.py')
    pspec=gp.build_p70c_spec(str(ROOT/'Paper/p70c_full_spec_v1.json'),str(ROOT/'Paper/enriched_records.pkl'))
    prows=[]
    for i in range(30):
        toks=[list(t) for t in gp.produce_manuscript(pspec,n_tokens=600,seed=700000+i) if t]
        best,_=v.scan(toks)
        r=cal(toks,705000+i) if best['b']>=2 else {**best,'null_mean':None,'null_sd':None,'z':None,'decision':False}
        prows.append({'seed':700000+i,**r})
    return grows,prows


def get_tokens(data,tid):
    out=[]
    for _,lines in data.get('pages',{}).items():
        for _,line in lines.items():
            txt=line.get('t',{}).get(tid,'')
            if txt: out.extend(txt.split())
    return out


def clean(tokens): return [list(t) for t in tokens if t and not t.startswith('<')]


def score_voynich():
    slim=ROOT/'voynich_transcriptions_slim.json'
    data=json.loads(slim.read_text()); out={}
    for tid in ('ZLZI','TTLI'):
        raw=get_tokens(data,tid)
        if not raw: continue
        reps=(('raw',raw),('bench_collapsed',[v.collapse_benches(t) for t in raw]))
        for rep,vals in reps:
            toks=clean(vals)
            r=cal(toks,800000+len(out))
            r['n_tokens']=len(toks)
            r['odd_rate']=sum(len(t)%2 for t in toks)/len(toks)
            out[f'{tid}_{rep}']=r
    return out


def main():
    pos,nulls=synthetic_battery()
    tpr=sum(r['decision'] for r in pos)/len(pos)
    brec=sum(r['b']==r['true_b'] for r in pos)/len(pos)
    nfpr=sum(r['decision'] for r in nulls)/len(nulls)
    grows,prows=controls()
    gfpr=sum(r['decision'] for r in grows)/len(grows)
    pfpr=sum(r['decision'] for r in prows)/len(prows)
    gate=(tpr>=.90 and brec>=.80 and nfpr<=.05 and gfpr<=.05 and pfpr<=.05)
    res={'protocol':{'version':'0.2','internal_shuffles':NSH,'z_gate':Z_GATE,'b_gate':2,
                     'tie_break':'smallest b within 1e-10 of max','voynich_sealed_until_gate':True,
                     'change_from_v01':'null-SD estimation 8 -> 64 shuffles; larger fresh preflight batteries'},
         'synthetic':{'n_positive':len(pos),'tpr':tpr,'exact_b_recovery':brec,'positives':pos,
                      'n_matched_null':len(nulls),'matched_null_fpr':nfpr,'matched_nulls':nulls},
         'g7':{'n':len(grows),'fpr':gfpr,'rows':grows},
         'p70c':{'n':len(prows),'fpr':pfpr,'rows':prows},'gate_passed':gate}
    if gate:
        res['voynich_exposed']=True; res['voynich']=score_voynich()
    else:
        res['voynich_exposed']=False; res['voynich']={}
    (OUT/'fractionation_gate_v02.json').write_text(json.dumps(res,indent=2,allow_nan=False))
    lines=['# Fractionation-composition gate v0.2','',
           f'- Internal positional-null shuffles per calibration: {NSH}',
           f'- Synthetic TPR: {tpr:.3f} ({sum(r["decision"] for r in pos)}/{len(pos)})',
           f'- Exact planted-b recovery: {brec:.3f} ({sum(r["b"]==r["true_b"] for r in pos)}/{len(pos)})',
           f'- Position-matched-null FPR: {nfpr:.3f} ({sum(r["decision"] for r in nulls)}/{len(nulls)})',
           f'- f57v Scribal Manual FPR: {gfpr:.3f} ({sum(r["decision"] for r in grows)}/{len(grows)})',
           f'- P70-C Scribal FPR: {pfpr:.3f} ({sum(r["decision"] for r in prows)}/{len(prows)})',
           f'- **Gate passed: {gate}**','']
    if gate:
        lines.append('## Frozen Voynich exposure')
        for k,r in res['voynich'].items():
            effect=r['score']-r['null_mean']
            ratio=effect/r['null_sd'] if r['null_sd'] else float('inf')
            lines.append(f"- {k}: b={r['b']}; score effect={effect:.6f}; null SD={r['null_sd']:.6f}; Z={ratio:.3f}; decision={r['decision']}; odd-rate={r['odd_rate']:.3f}; n={r['n_tokens']}")
    else:
        lines.append('Voynich remained sealed because v0.2 preflight failed.')
    (OUT/'FRACTIONATION_GATE_V02_RESULT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
