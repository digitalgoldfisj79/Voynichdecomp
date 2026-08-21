#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_v041_end_to_end as v041
import svt_v02 as svt
import v04_semimarkov_segmenter as seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=list(svt.MODES), required=True)
    ap.add_argument('--replicate', type=int, required=True)
    args = ap.parse_args()

    rep = v041.OFFSET + args.replicate
    language, model = v041.load_language(args.repo, f'svt-v041-clock-{args.mode}-{args.replicate}')
    trial = svt.make_svt_trial(language, 'dev', v041.LENGTH, args.mode, rep)

    fitted_seg = seg.fit(
        trial.surface,
        trial.surface_line_starts,
        len(language.alphabet),
        int(svt.core.stable_seed('svt-v041-seg', trial.head.seed)),
    )
    pred_head = v041.inferred_head(trial.surface, fitted_seg.starts, trial.surface_line_starts, trial.head.seed)

    # Diagnostic 1: keep inferred boundaries but supply the true state schedule.
    oracle_structure_fit = v041.fit_structure(
        pred_head, language, model, trial.head.mode, int(trial.head.period), 'svt-v041-clock-oracle-structure'
    )
    oracle_structure_prediction = oracle_structure_fit['selected']['prediction']
    oracle_structure_recovery = v041.sequence_recovery(trial.head.plain, oracle_structure_prediction)

    # Diagnostic 2: true boundaries and true state schedule, same key solver, on the same fresh trial.
    true_head = type('Head', (), {
        'cipher': [int(x) for x in trial.head.cipher],
        'line_starts': [int(x) for x in trial.head.line_starts],
        'seed': int(trial.head.seed),
    })()
    oracle_boundary_fit = v041.fit_structure(
        true_head, language, model, trial.head.mode, int(trial.head.period), 'svt-v041-clock-oracle-boundary'
    )
    oracle_boundary_prediction = oracle_boundary_fit['selected']['prediction']
    oracle_boundary_recovery = v041.sequence_recovery(trial.head.plain, oracle_boundary_prediction)

    # Directly quantify state-clock drift at boundaries that the segmenter got exactly right.
    true_pos_to_idx = {int(p): i for i, p in enumerate(trial.head_positions)}
    true_phase = svt.v0._phase(len(trial.head.cipher), int(trial.head.period), trial.head.mode, trial.head.line_starts or [0])
    pred_phase = svt.v0._phase(len(pred_head.cipher), int(trial.head.period), trial.head.mode, pred_head.line_starts or [0])
    exact = 0
    phase_match = 0
    index_deltas = []
    for pi, pos in enumerate(fitted_seg.starts):
        ti = true_pos_to_idx.get(int(pos))
        if ti is None:
            continue
        exact += 1
        index_deltas.append(int(pi - ti))
        if int(pred_phase[pi]) == int(true_phase[ti]):
            phase_match += 1

    payload = {
        'programme': 'SVT-v0.4.1-diagnostic',
        'binding': False,
        'voynich_opened': False,
        'replicate': int(rep),
        'true_mode': trial.head.mode,
        'true_period': int(trial.head.period),
        'boundary_f1': float(seg.boundary_f1(fitted_seg.starts, trial.head_positions)),
        'predicted_units': int(len(fitted_seg.starts)),
        'true_units': int(len(trial.head_positions)),
        'oracle_structure_predicted_boundaries_recovery': float(oracle_structure_recovery),
        'oracle_boundaries_true_structure_recovery': float(oracle_boundary_recovery),
        'exact_boundary_matches': int(exact),
        'phase_match_on_exact_boundaries': float(phase_match / max(1, exact)),
        'mean_abs_unit_index_drift_on_exact_boundaries': float(np.mean(np.abs(np.asarray(index_deltas, dtype=float)))) if index_deltas else 0.0,
        'max_abs_unit_index_drift_on_exact_boundaries': int(max((abs(x) for x in index_deltas), default=0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
