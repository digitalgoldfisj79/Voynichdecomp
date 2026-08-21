#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lattice_coupling_v051 as lc

DEV_OFFSET = {"de": 23000, "la": 33000}
BIND_OFFSET = {"de": 37000, "la": 39000}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=("dev", "binding"), required=True)
    ap.add_argument("--iso", choices=("de", "la"), required=True)
    ap.add_argument("--mode", choices=("periodic", "line_reset"), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    if args.stage == "dev":
        split = "dev"
        replicate = DEV_OFFSET[args.iso] + args.replicate
    else:
        split = "test"
        replicate = BIND_OFFSET[args.iso] + args.replicate

    payload = lc.run_trial(args.repo, args.iso, split, args.mode, replicate)
    payload["stage"] = args.stage
    payload["binding"] = args.stage == "binding"
    payload["spent_material"] = args.stage == "dev"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "stage": args.stage, "iso": args.iso, "mode": args.mode, "replicate": replicate,
        "map_f1": payload["surface_map_boundary_f1_eval_only"],
        "selected_f1": payload["boundary_f1_eval_only"],
        "selected_rank": payload["selected_surface_rank"],
        "cipher_z": payload["selected_cipher_z"],
        "count_abs": payload["count_abs_error_eval_only"],
        "exact_structure": payload["exact_structure_eval_only"],
        "recovery": payload["sequence_recovery_eval_only"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
