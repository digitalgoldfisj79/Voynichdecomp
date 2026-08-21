#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import svt_v02 as svt
import v04_semimarkov_segmenter as seg
import latin_proiel_portability as lat

LENGTH=1536
OFFSET=33000


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=list(svt.MODES),required=True)
    ap.add_argument('--replicate',type=int,required=True)
    args=ap.parse_args()

    language,_=lat.load_latin(args.repo/'.cache'/f'latin-seg-{args.mode}-{args.replicate}')
    rep=OFFSET+args.replicate
    trial=svt.make_svt_trial(language,'dev',LENGTH,args.mode,rep)
    fitted=seg.fit(trial.surface,trial.surface_line_starts,len(language.alphabet),int(svt.core.stable_seed('svt-v050-latin-seg',trial.head.seed)))
    f1=seg.boundary_f1(fitted.starts,trial.head_positions)
    cerr=abs(len(fitted.starts)-len(trial.head_positions))/max(1,len(trial.head_positions))
    legacy=svt.v0.top_segmentations(trial.surface,trial.surface_line_starts,len(language.alphabet),beam=1)
    legacy_f1=seg.boundary_f1(legacy[0].starts,trial.head_positions) if legacy else 0.0
    payload={
      'programme':'SVT-v0.5.0-Latin-portability','arm':'L2_segmentation','binding':True,'voynich_opened':False,
      'latin_source_repo':lat.REPO,'latin_source_commit':lat.COMMIT,'iso':'la','replicate':rep,
      'mode_generator_only':args.mode,'true_units':len(trial.head_positions),'predicted_units':len(fitted.starts),
      'count_relative_error':float(cerr),'boundary_f1':float(f1),'legacy_surprisal_f1':float(legacy_f1),
      'selected_restart':int(fitted.restart),'em_iterations':int(fitted.iterations)
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
