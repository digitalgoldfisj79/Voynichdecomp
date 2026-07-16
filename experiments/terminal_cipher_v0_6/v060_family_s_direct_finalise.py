#!/usr/bin/env python3
"""Finalise S3 development metrics when every lattice arm has abstained."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import v060_family_s_stage_s1 as s1
import v060_family_s_stage_s2 as s2
import v060_family_s_stage_s3 as s3
from v060_supabase_checkpoint_transport import DEFAULT_SIGNER_URL, signed_url

PHASE1_OBJECT = "v060/s3/evaluation/dev/phase1.json"
PHASE2_OBJECT = "v060/s3/evaluation/dev/phase2.json"
FINAL_OBJECT = "v060/s3/evaluation/dev/final.json"
TARGET_LENGTH = 384


def json_bytes(payload: dict[str, Any], pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def download_json(object_path: str) -> dict[str, Any]:
    response = requests.get(
        signed_url(DEFAULT_SIGNER_URL, "download", object_path), timeout=600
    )
    response.raise_for_status()
    return response.json()


def upload_json(object_path: str, payload: dict[str, Any]) -> None:
    response = requests.put(
        signed_url(DEFAULT_SIGNER_URL, "upload", object_path),
        data=json_bytes(payload, pretty=True),
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    response.raise_for_status()


def load_language(repo: Path) -> core.LanguageData:
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        repo / ".cache" / "v060-family-s3-direct-finalise",
    )
    return languages["en"]


def boundaries_from_lengths(lengths: Iterable[int]) -> list[int]:
    result: list[int] = []
    cursor = 0
    for width in lengths:
        cursor += int(width)
        result.append(cursor)
    return result


def plaintext_accuracy(truth: list[int], predicted: list[int]) -> float:
    return max(
        0.0,
        1.0 - Levenshtein.distance(truth, predicted) / max(1, len(truth), len(predicted)),
    )


def unit_language_score(
    plaintext: list[int],
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    if not plaintext:
        return -1e6
    unit_to_id = {unit: index for index, unit in enumerate(inventory)}
    ids = [unit_to_id[unit] for unit in s1.unitise(plaintext, inventory)]
    trigram, unigram, _probabilities = unit_model
    score = 0.0
    if ids:
        score += 0.15 * float(unigram[ids[0]])
    if len(ids) >= 2:
        score += 0.15 * float(unigram[ids[1]])
    for first, second, third in zip(ids, ids[1:], ids[2:]):
        score += float(trigram[first, second, third])
        score += 0.15 * float(unigram[third])
    return score / max(1, len(plaintext))


def calibrated_logit(features: list[float], calibration: dict[str, Any]) -> float:
    vector = np.asarray(features, dtype=np.float64)
    mean = np.asarray(calibration["feature_mean"], dtype=np.float64)
    std = np.asarray(calibration["feature_std"], dtype=np.float64)
    weights = np.asarray(calibration["weights"], dtype=np.float64)
    return float(((vector - mean) / std) @ weights + float(calibration["bias"]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovery = [float(row["plaintext_accuracy"]) for row in rows]
    boundaries = [float(row["boundary_f1"]) for row in rows]
    return {
        "trials": len(rows),
        "plaintext": {
            "mean": statistics.fmean(recovery),
            "median": statistics.median(recovery),
            "minimum": min(recovery),
            "at_least_75_count": sum(value >= 0.75 for value in recovery),
            "at_least_75_rate": statistics.fmean(value >= 0.75 for value in recovery),
            "exact_count": sum(bool(row["plaintext_exact"]) for row in rows),
        },
        "boundary_f1": {
            "mean": statistics.fmean(boundaries),
            "median": statistics.median(boundaries),
            "minimum": min(boundaries),
        },
        "selection": {"direct_count": len(rows), "lattice_count": 0},
        "lattice": {"available_count": 0, "abstention_count": len(rows)},
        "gate": {
            "pass": (
                statistics.fmean(recovery) >= 0.75
                and statistics.median(recovery) >= 0.85
                and sum(value >= 0.75 for value in recovery) >= 13
                and statistics.fmean(boundaries) >= 0.85
                and min(recovery) >= 0.40
            ),
            "thresholds": {
                "mean_plaintext": 0.75,
                "median_plaintext": 0.85,
                "at_least_75_count": 13,
                "mean_boundary_f1": 0.85,
                "minimum_plaintext": 0.40,
            },
        },
    }


def main() -> None:
    repo = Path(sys.argv[1])
    started = time.perf_counter()
    phase1 = download_json(PHASE1_OBJECT)
    phase2 = download_json(PHASE2_OBJECT)
    if int(phase2.get("lattice_abstention_count", -1)) != 16:
        raise RuntimeError("torch-free finalisation is legal only for 16/16 lattice abstention")
    language = load_language(repo)
    inventory = s1.candidate_inventory(language)
    unit_model = s2.build_unit_model(language, inventory)
    calibration = phase1["calibration"]
    p1_by_rep = {int(row["replicate"]): row for row in phase1["rows"]}
    p2_by_rep = {int(row["replicate"]): row for row in phase2["rows"]}
    rows: list[dict[str, Any]] = []
    for replicate in range(16):
        trial = s1.make_trial(language, "dev", TARGET_LENGTH, replicate)
        p1 = p1_by_rep[replicate]
        p2 = p2_by_rep[replicate]
        if bool(p2["lattice_available"]):
            raise RuntimeError(f"unexpected lattice candidate at replicate {replicate}")
        direct = [int(value) for value in p1["direct_plaintext"]]
        top_lengths = [int(value) for value in p1["segmentations"][0]["lengths"]]
        predicted_boundaries = boundaries_from_lengths(top_lengths)
        boundary_f1 = s3.boundary_f1(trial.boundaries, predicted_boundaries)
        direct_features = [
            float(p1["direct_beam_mean_logp"]),
            unit_language_score(direct, inventory, unit_model),
            -abs(len(direct) - TARGET_LENGTH) / TARGET_LENGTH,
        ]
        direct_logit = calibrated_logit(direct_features, calibration)
        accuracy = plaintext_accuracy(trial.plain, direct)
        row = {
            "iso": "en",
            "split": "dev",
            "replicate": replicate,
            "cipher_length": len(trial.cipher),
            "true_units": len(trial.units),
            "top_boundary_units": len(top_lengths),
            "boundary_f1": boundary_f1,
            "lattice_available": False,
            "lattice_abstention_reason": p2["lattice_abstention_reason"],
            "selected_hypothesis": "direct",
            "selected_calibrated_logit": direct_logit,
            "direct_calibrated_logit": direct_logit,
            "lattice_calibrated_logit": None,
            "direct_length": len(direct),
            "lattice_length": None,
            "selected_length": len(direct),
            "direct_accuracy": accuracy,
            "lattice_accuracy": None,
            "plaintext_accuracy": accuracy,
            "plaintext_exact": direct == trial.plain,
        }
        rows.append(row)
        print("V060_S3_FINAL_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    summary = summarize(rows)
    payload: dict[str, Any] = {
        "config": {
            "split": "dev",
            "length": TARGET_LENGTH,
            "replicates": 16,
            "beam_width": 4,
            "lattice_paths": 8,
            "screen_iterations": 700000,
            "screen_restarts": 50,
            "final_iterations": 700000,
            "final_restarts": 200,
            "checkpoint_sha256": phase1["checkpoint_sha256"],
            "phase1_sha256": phase1["sha256"],
            "phase2_sha256": phase2["sha256"],
            "finalisation_mode": "direct_only_after_16_of_16_lattice_abstention",
        },
        "calibration": calibration,
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
    upload_json(FINAL_OBJECT, payload)
    print("V060_S3_FINAL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_S3_FINAL_SHA256", payload["sha256"], flush=True)
    print("V060_S3_FINAL_RESULT", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
