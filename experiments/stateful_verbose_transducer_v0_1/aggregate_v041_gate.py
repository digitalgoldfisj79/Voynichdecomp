#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows=[]
    for p in sorted(args.input_dir.rglob('*.json')):
        try:
            r=json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if r.get('programme')=='SVT-v0.4.1' and r.get('stage')=='end_to_end_hidden_segmentation_blind_state_key':
            rows.append(r)
    if len(rows)!=8:
        raise SystemExit(f'expected 8 binding rows, found {len(rows)}')

    rec=np.asarray([float(r['sequence_recovery']) for r in rows])
    f1=np.asarray([float(r['boundary_f1']) for r in rows])
    err=np.asarray([float(r['count_relative_error']) for r in rows])
    exact=sum(bool(r['canonical_exact']) for r in rows)
    out={
        'programme':'SVT-v0.4.1',
        'binding':True,
        'voynich_opened':False,
        'trials':8,
        'canonical_exact_count':int(exact),
        'mean_sequence_recovery':float(rec.mean()),
        'median_sequence_recovery':float(np.median(rec)),
        'minimum_sequence_recovery':float(rec.min()),
        'mean_boundary_f1':float(f1.mean()),
        'minimum_boundary_f1':float(f1.min()),
        'mean_abs_count_relative_error':float(err.mean()),
        'per_trial':rows,
    }
    out['gate_pass']=bool(
        exact==8 and out['mean_sequence_recovery']>=0.90 and out['median_sequence_recovery']>=0.90
        and out['minimum_sequence_recovery']>=0.85 and out['mean_boundary_f1']>=0.90
        and out['minimum_boundary_f1']>=0.85 and out['mean_abs_count_relative_error']<=0.05
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='per_trial'},indent=2,sort_keys=True))
    if not out['gate_pass']:
        raise SystemExit(2)

if __name__=='__main__':
    main()
