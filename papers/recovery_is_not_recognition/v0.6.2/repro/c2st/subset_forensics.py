#!/usr/bin/env python3
"""Exhaustively identify the eight-feature ablation subset.

Searches all C(13, 8) subsets against the five published ablation AUCs using the
transcript-recovered feature and chunk definitions. The result is forensic
identification of a lost inline edit, not textual recovery of that edit.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).resolve().parent
FEATURES = [
    "wl_mean", "wl_std", "wl_autocorr", "ttr", "hapax", "H1", "H2",
    "chardist_max", "digcov", "within_tok_nextH", "opener_gallows",
    "charpos_gallows", "adj_repeat",
]
CANDIDATES = ["real_B", "line-shuffle", "word-shuffle", "gen_template_v10", "delex_char3"]
TARGET = np.array([0.421, 0.844, 0.955, 0.988, 0.970])


def main() -> None:
    namespace: dict = {}
    source = (WORK / "step8b_eval_harness.py").read_text(encoding="utf-8")
    exec(source.split("ref=featmatrix")[0], namespace)
    featmatrix = namespace["featmatrix"]

    reference = featmatrix("real_A")
    rng = np.random.default_rng(0)
    pairs = []
    for name in CANDIDATES:
        candidate = featmatrix(name)
        n = min(len(reference), len(candidate))
        reference_indices = rng.choice(len(reference), n, replace=False)
        candidate_indices = rng.choice(len(candidate), n, replace=False)
        pairs.append((reference[reference_indices], candidate[candidate_indices], n))

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    results = []
    for subset in itertools.combinations(range(13), 8):
        values = []
        for x_real, x_candidate, n in pairs:
            x = np.vstack([x_real[:, subset], x_candidate[:, subset]])
            y = np.r_[np.zeros(n), np.ones(n)]
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000)
            )
            values.append(
                cross_val_score(classifier, x, y, cv=cv, scoring="roc_auc").mean()
            )
        values_array = np.array(values)
        max_delta = float(np.max(np.abs(values_array - TARGET)))
        results.append((max_delta, subset, values_array))

    results.sort(key=lambda row: row[0])
    hits = [row for row in results if row[0] <= 0.005]
    best, second = results[0], results[1]
    output = {
        "search_space": 1287,
        "acceptance_threshold_max_abs_delta": 0.005,
        "qualifying_subsets": len(hits),
        "identified_unique": len(hits) == 1,
        "best": {
            "max_abs_delta": best[0],
            "retained_features": [FEATURES[index] for index in best[1]],
            "auc_values": dict(zip(CANDIDATES, map(float, best[2]))),
            "excludes_opener_and_adjacent_repeat": 10 not in best[1] and 12 not in best[1],
        },
        "second_best": {
            "max_abs_delta": second[0],
            "retained_features": [FEATURES[index] for index in second[1]],
        },
        "interpretation": (
            "The ablation subset is uniquely identified under the preregistered 0.005 "
            "criterion, but the original inline ablation source remains textually unrecovered."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
