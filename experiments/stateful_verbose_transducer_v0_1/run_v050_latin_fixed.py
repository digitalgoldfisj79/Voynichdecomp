#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import svt_v02 as svt
import run_v034_primitive_period as v034
import latin_proiel_portability as lat

LENGTH = 1536
OFFSET = 31000
SHORTLIST_K = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=list(svt.MODES), required=True)
    ap.add_argument('--replicate', type=int, required=True)
    args = ap.parse_args()

    language, model = lat.load_latin(args.repo / '.cache' / f'latin-fixed-{args.mode}-{args.replicate}')
    replicate = OFFSET + args.replicate
    trial = svt.make_svt_trial(language, 'dev', LENGTH, args.mode, replicate)
    head = trial.head

    screen=[]
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            screen.append({'mode':mode,'period':int(period),'screen_score':v034.screen_score(head,language,model,mode,period)})
    screen.sort(key=lambda r:r['screen_score'], reverse=True)
    truth_rank = next(i+1 for i,r in enumerate(screen) if r['mode']==head.mode and int(r['period'])==int(head.period))
    shortlist=screen[:SHORTLIST_K]
    refined=[v034.fit_structure(head,language,model,r['mode'],int(r['period']),'svt-v050-latin-fixed') for r in shortlist]
    selected=max(refined,key=lambda r:r['selected']['score'])
    canonical=v034.canonicalise(head,language,model,selected,'svt-v050-latin-canonical')
    exact=(canonical['canonical_mode']==head.mode and int(canonical['canonical_period'])==int(head.period))

    payload={
      'programme':'SVT-v0.5.0-Latin-portability','arm':'L1_fixed_boundary','binding':True,'voynich_opened':False,
      'latin_source_repo':lat.REPO,'latin_source_commit':lat.COMMIT,'iso':'la','length':LENGTH,'replicate':replicate,
      'true_mode':head.mode,'true_period':int(head.period),'screen_truth_rank':int(truth_rank),'screen_top6':shortlist,
      'canonical_mode':canonical['canonical_mode'],'canonical_period':int(canonical['canonical_period']),
      'canonical_recovery':float(canonical['canonical_recovery']),'canonical_exact':bool(exact)
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
