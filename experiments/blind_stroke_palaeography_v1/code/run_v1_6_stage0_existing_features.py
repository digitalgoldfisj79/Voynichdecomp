#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import io
import json
import re
import subprocess
import tarfile
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

SOURCE_JOB_ID = "6a5a1540d216bd6f3a1fb177"
EXPECTED_BUNDLE_SHA256 = "7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa"
SEED = 20260719
BOOTSTRAPS = 2000
PERMUTATIONS = 999
CALIBRATION_WRITERS = 30


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pair_table(x: np.ndarray, writers: np.ndarray) -> dict[str, np.ndarray]:
    x = l2(x)
    i, j = np.triu_indices(len(x), k=1)
    return {
        "i": i,
        "j": j,
        "score": np.sum(x[i] * x[j], axis=1),
        "label": (writers[i] == writers[j]).astype(np.int8),
        "writer_i": writers[i],
        "writer_j": writers[j],
    }


def eer(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    fnr = 1.0 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[k] + fnr[k]) / 2.0)


def metrics(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
    prevalence = float(np.average(y, weights=weight)) if weight is not None else float(np.mean(y))
    ap = float(average_precision_score(y, score, sample_weight=weight))
    return {
        "roc_auc": float(roc_auc_score(y, score, sample_weight=weight)),
        "average_precision": ap,
        "prevalence": prevalence,
        "ap_lift": ap - prevalence,
        "equal_error_rate": eer(y, score, weight),
    }


def retrieval(x: np.ndarray, writers: np.ndarray) -> dict[str, float]:
    x = l2(x)
    similarity = x @ x.T
    aps: list[float] = []
    top1: list[float] = []
    top5: list[float] = []
    for q in range(len(x)):
        relevant = writers == writers[q]
        relevant[q] = False
        if not relevant.any():
            continue
        order = np.argsort(-similarity[q])
        order = order[order != q]
        rel = relevant[order].astype(np.int8)
        ranks = np.flatnonzero(rel) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        top1.append(float(rel[:1].any()))
        top5.append(float(rel[:5].any()))
    return {
        "map": float(np.mean(aps)),
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "eligible_queries": len(aps),
    }


def weighted_pair_weights(table: dict[str, np.ndarray], counts: dict[str, int]) -> np.ndarray:
    wi = table["writer_i"]
    wj = table["writer_j"]
    y = table["label"]
    a = np.fromiter((counts.get(str(w), 0) for w in wi), dtype=np.float64, count=len(wi))
    b = np.fromiter((counts.get(str(w), 0) for w in wj), dtype=np.float64, count=len(wj))
    return np.where(y == 1, a, a * b)


def bootstrap(
    tables: dict[str, dict[str, np.ndarray]], writers: np.ndarray
) -> dict[str, Any]:
    unique = np.array(sorted(set(map(str, writers))))
    rng = np.random.default_rng(SEED + 1)
    series: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in tables
    }
    differences: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = dict(zip(*np.unique(sampled, return_counts=True)))
        replicate: dict[str, dict[str, float]] = {}
        for name, table in tables.items():
            weight = weighted_pair_weights(table, counts)
            keep = weight > 0
            replicate[name] = metrics(
                table["label"][keep], table["score"][keep], weight[keep]
            )
            for key, value in replicate[name].items():
                series[name][key].append(value)
        for nuisance in ["acquisition", "ink", "combined"]:
            differences[nuisance].append(
                replicate["selected"]["roc_auc"] - replicate[nuisance]["roc_auc"]
            )
    out: dict[str, Any] = {"replicates": BOOTSTRAPS, "metrics": {}, "auc_differences": {}}
    for name, values in series.items():
        out["metrics"][name] = {
            key: {
                "lower_95": float(np.quantile(v, 0.025)),
                "upper_95": float(np.quantile(v, 0.975)),
            }
            for key, v in values.items()
        }
    for name, values in differences.items():
        out["auc_differences"][name] = {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return out


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if mask.any():
            value += mask.mean() * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return float(value)


def calibration(table: dict[str, np.ndarray], writers: np.ndarray) -> dict[str, Any]:
    unique = np.array(sorted(set(map(str, writers))))
    rng = np.random.default_rng(SEED + 2)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    calibration_set = set(shuffled[:CALIBRATION_WRITERS])
    evaluation_set = set(shuffled[CALIBRATION_WRITERS:])
    wi = np.array(list(map(str, table["writer_i"])))
    wj = np.array(list(map(str, table["writer_j"])))
    fit = np.array([(a in calibration_set and b in calibration_set) for a, b in zip(wi, wj)])
    test = np.array([(a in evaluation_set and b in evaluation_set) for a, b in zip(wi, wj)])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    model.fit(table["score"][fit, None], table["label"][fit])
    probability = model.predict_proba(table["score"][test, None])[:, 1]
    y = table["label"][test]
    prevalence = float(y.mean())
    return {
        "calibration_writers": sorted(calibration_set),
        "evaluation_writers": sorted(evaluation_set),
        "calibration_pairs": int(fit.sum()),
        "evaluation_pairs": int(test.sum()),
        "evaluation_positive_pairs": int(y.sum()),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence_predictor_brier": float(brier_score_loss(y, np.full(len(y), prevalence))),
        "ece_10_bins": ece(y, probability, bins=10),
        "intercept": float(model.intercept_[0]),
        "coefficient": float(model.coef_[0, 0]),
    }


def reconstruct_bundle() -> bytes:
    process = subprocess.run(
        ["hf", "jobs", "logs", SOURCE_JOB_ID], capture_output=True, text=True, check=True
    )
    chunks: dict[int, str] = {}
    expected_chunks = None
    for line in process.stdout.splitlines():
        if line.startswith("V15_BUNDLE_BEGIN "):
            expected_chunks = int(json.loads(line.split(" ", 1)[1])["chunks"])
        match = re.match(r"^V15_BUNDLE_CHUNK\s+(\d+)\s+(.+)$", line)
        if match:
            chunks[int(match.group(1))] = match.group(2).strip()
    if expected_chunks is None or sorted(chunks) != list(range(expected_chunks)):
        raise RuntimeError("incomplete v1.5.1 bundle in job logs")
    raw = base64.b64decode("".join(chunks[i] for i in range(expected_chunks)), validate=True)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(f"bundle hash mismatch: {observed}")
    return raw


def permutation_test(table: dict[str, np.ndarray], writers: np.ndarray) -> dict[str, Any]:
    observed = roc_auc_score(table["label"], table["score"])
    rng = np.random.default_rng(SEED + 3)
    null = np.empty(PERMUTATIONS, dtype=np.float64)
    for index in range(PERMUTATIONS):
        permuted = writers.copy()
        rng.shuffle(permuted)
        y = (permuted[table["i"]] == permuted[table["j"]]).astype(np.int8)
        null[index] = roc_auc_score(y, table["score"])
    return {
        "permutations": PERMUTATIONS,
        "observed_auc": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_value": float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)),
    }


def main() -> int:
    bundle = reconstruct_bundle()
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:*") as archive:
        feature_bytes = archive.extractfile("exact_features.npz").read()
    with np.load(io.BytesIO(feature_bytes), allow_pickle=False) as z:
        writers = z["test_writers"].astype(str)
        selected = z["test_selected"].astype(np.float64)
        raw = z["test"].astype(np.float64)
        acquisition = z["acquisition_test"].astype(np.float64)
        ink = z["ink_test"].astype(np.float64)
    combined = StandardScaler().fit_transform(np.concatenate([acquisition, ink], axis=1))
    representations = {
        "selected": selected,
        "raw": raw,
        "acquisition": acquisition,
        "ink": ink,
        "combined": combined,
    }
    tables = {name: pair_table(x, writers) for name, x in representations.items()}
    point = {name: metrics(t["label"], t["score"]) for name, t in tables.items()}
    retrieval_results = {name: retrieval(x, writers) for name, x in representations.items()}
    nuisance_correlations = {
        name: {
            "spearman_r": float(spearmanr(tables["selected"]["score"], tables[name]["score"]).statistic),
            "p_value": float(spearmanr(tables["selected"]["score"], tables[name]["score"]).pvalue),
        }
        for name in ["acquisition", "ink", "combined"]
    }
    result = {
        "schema": "blind-pal-saghog-v1.6-stage0-existing-features",
        "source_job_id": SOURCE_JOB_ID,
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "seed": SEED,
        "pages": len(writers),
        "writers": len(set(writers)),
        "positive_pairs": int(tables["selected"]["label"].sum()),
        "negative_pairs": int((tables["selected"]["label"] == 0).sum()),
        "point_metrics": point,
        "writer_cluster_bootstrap": bootstrap(tables, writers),
        "retrieval": retrieval_results,
        "selected_calibration": calibration(tables["selected"], writers),
        "selected_permutation": permutation_test(tables["selected"], writers),
        "selected_nuisance_pair_score_correlations": nuisance_correlations,
        "limitations": [
            "Secondary characterization of the already observed terminal-test feature matrix.",
            "No physical page IDs, source IDs, content labels, or perturbation embeddings are present in exact_features.npz.",
            "Calibration uses a deterministic 30-writer subset of the terminal writers and is descriptive, not an independent confirmation.",
            "This result cannot authorize opening Voynich and cannot satisfy the full v1.6 external validation gates.",
        ],
        "seal": {
            "voynich_opened": False,
            "davis_labels_loaded": False,
            "f115r_loaded": False,
        },
    }
    print("V16_STAGE0_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
