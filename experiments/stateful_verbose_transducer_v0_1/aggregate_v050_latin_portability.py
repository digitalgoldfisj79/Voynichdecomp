#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, statistics
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[]
    for p in sorted(args.input_dir.rglob('*.json')):
        try: r=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        if r.get('programme')=='SVT-v0.5.0-Latin-portability' and r.get('binding') is True: rows.append(r)
    l1=[r for r in rows if r.get('arm')=='L1_fixed_boundary']; l2=[r for r in rows if r.get('arm')=='L2_segmentation']
    if len(l1)!=8 or len(l2)!=8: raise SystemExit(f'expected 8+8 rows, found {len(l1)}+{len(l2)}')

    rec=[float(r['canonical_recovery']) for r in l1]
    fixed={
      'trials':8,'exact_structure_correct':sum(bool(r['canonical_exact']) for r in l1),
      'mean_recovery':statistics.mean(rec),'median_recovery':statistics.median(rec),'minimum_recovery':min(rec),
      'trials_ge_090':sum(x>=0.90 for x in rec),'screen_truth_in_top6':sum(int(r['screen_truth_rank'])<=6 for r in l1)
    }
    fixed['pass']=bool(fixed['exact_structure_correct']==8 and fixed['mean_recovery']>=0.95 and fixed['median_recovery']>=0.97 and fixed['minimum_recovery']>=0.85 and fixed['trials_ge_090']==8)

    f1=np.asarray([float(r['boundary_f1']) for r in l2]); cerr=np.asarray([float(r['count_relative_error']) for r in l2]); legacy=np.asarray([float(r['legacy_surprisal_f1']) for r in l2])
    segmentation={
      'trials':8,'mean_boundary_f1':float(f1.mean()),'median_boundary_f1':float(np.median(f1)),'minimum_boundary_f1':float(f1.min()),
      'trials_ge_085':int(np.sum(f1>=0.85)),'mean_abs_count_relative_error':float(cerr.mean()),'legacy_mean_boundary_f1':float(legacy.mean())
    }
    segmentation['pass']=bool(segmentation['mean_boundary_f1']>=0.90 and segmentation['median_boundary_f1']>=0.90 and segmentation['minimum_boundary_f1']>=0.85 and segmentation['trials_ge_085']==8 and segmentation['mean_abs_count_relative_error']<=0.05)

    payload={'programme':'SVT-v0.5.0-Latin-portability','binding':True,'voynich_opened':False,'fixed_boundary':fixed,'segmentation':segmentation,'gate_pass':bool(fixed['pass'] and segmentation['pass']),'per_trial':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='per_trial'},indent=2,sort_keys=True))
    raise SystemExit(0 if payload['gate_pass'] else 2)

if __name__=='__main__': main()
