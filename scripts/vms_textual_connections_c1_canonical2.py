#!/usr/bin/env python3
"""Canonical C1 rerun using frozen Caesar main-body identity.

No protocol or threshold changes. Voynich target connections remain sealed.
The frozen body hash is calculated over ASCII-token-normalized text between
`COMMENTARIUS PRIMUS` title marker and the Project Gutenberg end marker.
"""
from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path
import numpy as np, requests
import vms_textual_connections_c1 as base

EXPECTED_MAIN_SHA='1135e90a23cc19460e5827cec1b9dddf8b1aeaddf8b598b1dcc06b6f8c2255d3'
EXPECTED_WORDS=20516
FAM_SEED={'RARE':11,'SUBWORD':23,'MORPH':37}
OUT=Path('artifacts/vms_textual_connections_c1_canonical2_v01'); OUT.mkdir(parents=True,exist_ok=True)
START='C. IULI CAESARIS DE BELLO GALLICO COMMENTARIUS PRIMUS'
END="End of Project Gutenberg's"


def fetch_caesar():
    r=requests.get(base.CAESAR_URL,timeout=60,headers={'User-Agent':'VoynichTextConnections/0.1-canonical2'}); r.raise_for_status()
    raw=r.content; txt=raw.decode('utf-8-sig',errors='replace')
    a=txt.find(START); b=txt.find(END)
    if a<0 or b<0 or b<=a: raise RuntimeError('FROZEN_BODY_MARKERS_MISSING')
    toks=base.strict_words(txt[a:b]); norm=' '.join(toks); h=hashlib.sha256(norm.encode()).hexdigest()
    if len(toks)!=EXPECTED_WORDS or h!=EXPECTED_MAIN_SHA:
        raise RuntimeError(f'FROZEN_MAIN_BODY_IDENTITY_MISMATCH words={len(toks)} norm_sha={h} expected_words={EXPECTED_WORDS} expected_sha={EXPECTED_MAIN_SHA}')
    return toks,{'current_raw_sha256':hashlib.sha256(raw).hexdigest(),'normalized_main_sha256':h,'registered_normalized_main_sha256':EXPECTED_MAIN_SHA,'token_count':len(toks),'url':base.CAESAR_URL}


def score_window(units,fam,defs):
    n=len(units); D=np.full((n,n),-1e99,float)
    for i in range(n):
        for j in range(n):
            if i!=j: D[i,j]=base.direct_score(units[i],units[j],fam,defs)[0]
    U=np.maximum(D,D.T); pred=set()
    for i in range(n):
        js=[j for j in range(n) if j!=i]; pred.add(tuple(sorted((i,max(js,key=lambda q:U[i,q])))))
    truth={(i,i+1) for i in range(n-1)}
    tp=len(pred&truth); prec=tp/len(pred) if pred else 0.; rec=tp/len(truth); fpr=1-prec
    dirs=[1 if D[i,j]>D[j,i] else 0 for i,j in pred&truth]; diracc=float(np.mean(dirs)) if dirs else float('nan')
    obs=float(np.mean([U[i,j] for i,j in truth])); pool=[U[i,j] for i in range(n) for j in range(i+1,n) if (i,j) not in truth]
    rng=random.Random(base.SEED+1000+sum(len(x[0]) for x in units)+FAM_SEED[fam])
    null=[float(np.mean(rng.sample(pool,len(truth)))) for _ in range(base.N_NULL)]
    m=float(np.mean(null)); sd=float(np.std(null,ddof=1)); z=(obs-m)/sd if sd else float('nan')
    return {'pred':sorted(map(list,pred)),'truth':sorted(map(list,truth)),'tp':tp,'precision':prec,'recall':rec,'false_edge_rate':fpr,'direction_accuracy':diracc,'boundary_mean':obs,'null_mean':m,'null_sd':sd,'effect':obs-m,'effect_over_null_sd':z,'boundary_pass':bool(z>=2)}


def main():
    caesar,src=fetch_caesar(); lens=base.vms_page_lengths(); pages=base.make_pages(caesar,lens); units=[pages[i:i+4] for i in range(0,len(pages),4)]
    defs=base.build_defs(caesar); windows=[]
    for st in range(0,len(units)-base.WINDOW_UNITS+1,base.STEP_UNITS):
        us=units[st:st+base.WINDOW_UNITS]; fr={fam:score_window(us,fam,defs) for fam in base.FAMS}; windows.append({'start_unit':st,'families':fr,'combined':base.combine(fr)})
    agg={}
    for fam in base.FAMS:
        rr=[w['families'][fam] for w in windows]; d=[r['direction_accuracy'] for r in rr if math.isfinite(r['direction_accuracy'])]
        q={'mean_recall':float(np.mean([r['recall'] for r in rr])),'mean_false_edge_rate':float(np.mean([r['false_edge_rate'] for r in rr])),'direction_accuracy':float(np.mean(d)) if d else float('nan'),'boundary_pass_fraction':sum(r['boundary_pass'] for r in rr)/len(rr)}
        q['qualified']=bool(q['mean_recall']>=.70 and q['mean_false_edge_rate']<=.20 and q['direction_accuracy']>=.70 and q['boundary_pass_fraction']>=.75); agg[fam]=q
    cc=[w['combined'] for w in windows]; comb={'mean_recall':float(np.mean([x['recall'] for x in cc])),'mean_false_edge_rate':float(np.mean([x['false_edge_rate'] for x in cc]))}; nq=sum(agg[f]['qualified'] for f in base.FAMS)
    comb.update(n_qualified_families=nq,qualified=bool(nq>=2 and comb['mean_recall']>=.75 and comb['mean_false_edge_rate']<=.15))
    out={'protocol':base.PROTOCOL,'stage':'C1_CANONICAL2','target_connections_opened':False,'source':src,'pseudo_pages':len(pages),'pseudo_units':len(units),'n_windows':len(windows),'family_aggregate':agg,'combined':comb,'gate':'UNCHANGED'}
    (OUT/'c1_canonical2.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Pure-text C1 canonical closeout','', 'Target connections opened: **NO**','',f"Frozen Caesar main-body SHA-256: `{src['normalized_main_sha256']}`",f"Pseudo-units: {len(units)}; windows: {len(windows)}",'']
    for f,q in agg.items(): lines.append(f"- {f}: recall {q['mean_recall']:.3f}; false-edge {q['mean_false_edge_rate']:.3f}; direction {q['direction_accuracy']:.3f}; boundary>=2SD windows {q['boundary_pass_fraction']:.3f}; **{'PASS' if q['qualified'] else 'FAIL'}**")
    lines += ['',f"Combined: recall {comb['mean_recall']:.3f}; false-edge {comb['mean_false_edge_rate']:.3f}; qualified families {nq}/3; **{'PASS' if comb['qualified'] else 'FAIL'}**",'', 'FAIL => Voynich target remains sealed; v0.1 closes negative by preregistration.']
    (OUT/'C1_CANONICAL_CLOSEOUT.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__': main()
