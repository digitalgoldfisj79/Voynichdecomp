#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_jlcd_neighbour_bound as nb
SEED=20260829;REPS=2000

def score(a,b):
    va=np.array([math.log2(a['short_n']+1),math.log2(a['long_n']+1),math.log2(a['total_n']+1)])
    vb=np.array([math.log2(b['short_n']+1),math.log2(b['long_n']+1),math.log2(b['total_n']+1)])
    return float(np.abs(va-vb).sum())

def matched_sets(allpairs,target):
    used={t for p in target for t in (p['short'],p['long'])}; controls=[]; kept=[]
    pool=[p for p in allpairs if p['inserted'] not in ('e','i') and p['total_n']>=8 and min(p['short_n'],p['long_n'])>=2]
    for p in target:
        cand=[q for q in pool if q['short_len']==p['short_len'] and q['short'] not in used and q['long'] not in used]
        if not cand:continue
        cand.sort(key=lambda q:(score(p,q),q['short'],q['long'],q['inserted']))
        q=cand[0];kept.append(dict(p));controls.append(dict(q));used|={q['short'],q['long']}
    for i,p in enumerate(kept):p['pair_id']=i
    for i,p in enumerate(controls):p['pair_id']=i
    return kept,controls

def contrast(occs,target,control,rep,seed):
    gt,mt=nb.prepare(occs,target,rep,'full');gc,mc=nb.prepare(occs,control,rep,'full')
    dt=nb.perm(gt,seed,REPS,keep_null=True);dc=nb.perm(gc,seed+1,REPS,keep_null=True)
    nt=np.asarray(dt.pop('_null'),float);nc=np.asarray(dc.pop('_null'),float)
    et=dt['effect_bits'];ec=dc['effect_bits'];obs=et-ec
    null=(nt-nt.mean())-(nc-nc.mean());sd=float(null.std(ddof=1));z=obs/sd if sd>0 else float('nan');p=float((1+np.sum(np.abs(null)>=abs(obs)))/(REPS+1))
    return {'target':dt,'control':dc,'target_meta':mt,'control_meta':mc,'contrast_effect_bits':float(obs),'contrast_null_sd':sd,'contrast_z':float(z),'contrast_empirical_p_2s':p,'reps':REPS}

def fmt(rep,d):
    z=d['contrast_z'];lead='the metric does not resolve this — ' if abs(z)<2 else ''
    return f"{lead}{rep} e/i-minus-matched-control contrast: effect={d['contrast_effect_bits']:.6f} bits/occurrence; matched-null SD={d['contrast_null_sd']:.6f}; z={z:.2f}; p={d['contrast_empirical_p_2s']:.4f}."

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sm=json.loads(a.section_map.read_text())['mapping'];occs=nb.parse(a.source,sm);pairs=nb.discover_pairs(occs);target0=nb.disjoint(pairs,True);target,control=matched_sets(pairs,target0)
    R={'programme':'JLCD_specificity_match','source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'target_pairs_initial':len(target0),'matched_pairs':len(target),'representations':{},'target_pairs':target,'control_pairs':control}
    for i,rep in enumerate(('eva','char')):R['representations'][rep]=contrast(occs,target,control,rep,SEED+i*10000)
    support=all(R['representations'][r]['contrast_z']>=2 for r in ('eva','char'));against=all(R['representations'][r]['contrast_z']<=-2 for r in ('eva','char'));R['joachim_specificity_support']=support;R['evidence_against_ei_specificity']=against
    with (a.out/'specificity_match.pkl').open('wb') as f:pickle.dump(R,f,pickle.HIGHEST_PROTOCOL)
    (a.out/'SPECIFICITY_MATCH.json').write_text(json.dumps(R,indent=2,default=str))
    L=['# JLCD v0.1 — matched e/i specificity contrast','','# RETRACTED FINDINGS','','None.','','# CURRENT FINDINGS','',f"Matched {len(target)} e/i insertion pairs one-to-one to {len(control)} non-e/i insertion pairs; matching uses identical short-word length and closest log-frequency support, with no token type reused.",'']
    for rep in ('eva','char'):
        d=R['representations'][rep];L += [fmt(rep,d),f"  e/i effect={d['target']['effect_bits']:.6f}, null SD={d['target']['null_sd']:.6f}, z={d['target']['z']:.2f}, eligible={d['target_meta']['eligible_rows']}.",f"  control effect={d['control']['effect_bits']:.6f}, null SD={d['control']['null_sd']:.6f}, z={d['control']['z']:.2f}, eligible={d['control_meta']['eligible_rows']}.",'']
    L += ['## Decision',f"Joachim e/i specificity support: {'PASS' if support else 'FAIL'}.",f"Negative specificity contrast replicated in both representations: {'YES' if against else 'NO'}.",'','The contrast asks the exact discriminating question: if e/i additions are unusually important length counters, their context shift should exceed equally supported one-unit additions of other glyphs. A negative replicated contrast would instead show that e/i near-neighbours are less externally context-separated than matched non-e/i near-neighbours under this observable.']
    (a.out/'SPECIFICITY_MATCH.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
