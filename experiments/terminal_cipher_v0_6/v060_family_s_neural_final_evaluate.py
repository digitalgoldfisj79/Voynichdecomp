#!/usr/bin/env python3
"""Frozen three-phase development evaluation for the final v0.6 Family S3 ensemble."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import torch
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import v060_family_s_stage_s1 as s1
import v060_family_s_stage_s2 as s2
import v060_family_s_stage_s3 as s3
from v060_family_s_neural_common import (
    Example,
    NeuralTransducer,
    SyntheticGenerator,
    canonical_first_occurrence,
)
from v060_supabase_checkpoint_transport import DEFAULT_SIGNER_URL, sha256_file, signed_url

CHECKPOINT_MANIFESTS = {
    1731: "v060/s3/seed1731/u30000/s3_neural_seed1731_u30000.pt.manifest.json",
    1732: "v060/s3/seed1732/u30000/s3_neural_seed1732_u30000.pt.manifest.json",
}
PHASE1_OBJECT = "v060/s3/evaluation/dev/phase1.json"
PHASE2_OBJECT = "v060/s3/evaluation/dev/phase2.json"
FINAL_OBJECT = "v060/s3/evaluation/dev/final.json"
LENGTH_PRIOR = {1: 0.20, 2: 0.45, 3: 0.35}
TARGET_LENGTH = 384
BEAM_WIDTH = 4
LATTICE_PATHS = 8
EPSILON = 1e-7


def json_bytes(payload: dict[str, Any], pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def upload_bytes(object_path: str, payload: bytes, signer_url: str) -> None:
    response = requests.put(
        signed_url(signer_url, "upload", object_path),
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    response.raise_for_status()


def download_bytes(object_path: str, signer_url: str) -> bytes:
    response = requests.get(
        signed_url(signer_url, "download", object_path),
        timeout=600,
    )
    response.raise_for_status()
    return response.content


def download_json(object_path: str, signer_url: str) -> dict[str, Any]:
    return json.loads(download_bytes(object_path, signer_url))


def restore_checkpoint(manifest_object: str, destination: Path, signer_url: str) -> dict[str, Any]:
    manifest = download_json(manifest_object, signer_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for part in sorted(manifest["parts"], key=lambda row: int(row["index"])):
            response = requests.get(
                signed_url(signer_url, "download", part["object_path"]),
                stream=True,
                timeout=600,
            )
            response.raise_for_status()
            digest = hashlib.sha256()
            total = 0
            for block in response.iter_content(4 * 1024 * 1024):
                if not block:
                    continue
                output.write(block)
                digest.update(block)
                total += len(block)
            if total != int(part["bytes"]) or digest.hexdigest() != part["sha256"]:
                raise RuntimeError(f"checkpoint part verification failed: {part['object_path']}")
    if destination.stat().st_size != int(manifest["original_bytes"]):
        raise RuntimeError(f"checkpoint size mismatch: {manifest_object}")
    if sha256_file(destination) != manifest["original_sha256"]:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {manifest_object}")
    return manifest


def load_models(device: torch.device, working: Path, signer_url: str) -> tuple[list[NeuralTransducer], list[dict[str, Any]]]:
    models: list[NeuralTransducer] = []
    manifests: list[dict[str, Any]] = []
    for seed in (1731, 1732):
        checkpoint_path = working / f"s3_neural_seed{seed}_u30000.pt"
        manifest = restore_checkpoint(CHECKPOINT_MANIFESTS[seed], checkpoint_path, signer_url)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["seed"]) != seed or int(checkpoint["updates"]) != 30000:
            raise RuntimeError(f"checkpoint metadata mismatch for seed {seed}")
        model = NeuralTransducer(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().to(device)
        models.append(model)
        manifests.append(manifest)
    return models, manifests


def trial_tensors(trial: s1.SegmentationTrial, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = canonical_first_occurrence(trial.cipher)
    source = torch.tensor([[value + 1 for value in canonical]], dtype=torch.long, device=device)
    line_flags = torch.zeros_like(source)
    line_flags[0, 0] = 1
    source_padding = torch.zeros_like(source, dtype=torch.bool)
    return source, line_flags, source_padding


def example_tensors(example: Example, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.tensor([example.source], dtype=torch.long, device=device)
    line_flags = torch.tensor([example.line_flags], dtype=torch.long, device=device)
    source_padding = torch.zeros_like(source, dtype=torch.bool)
    return source, line_flags, source_padding


@torch.inference_mode()
def encode_ensemble(
    models: list[NeuralTransducer],
    source: torch.Tensor,
    line_flags: torch.Tensor,
    source_padding: torch.Tensor,
) -> tuple[list[torch.Tensor], np.ndarray]:
    memories: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for model in models:
        memory, boundary_logits = model.encode(source, line_flags, source_padding)
        memories.append(memory)
        probabilities.append(torch.sigmoid(boundary_logits))
    boundary = torch.stack(probabilities, dim=0).mean(dim=0)[0]
    return memories, boundary.detach().cpu().numpy().astype(np.float64)


@torch.inference_mode()
def beam_decode(
    models: list[NeuralTransducer],
    memories: list[torch.Tensor],
    source_padding: torch.Tensor,
    target_length: int = TARGET_LENGTH,
    beam_width: int = BEAM_WIDTH,
) -> tuple[list[int], float]:
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    alphabet_size = models[0].alphabet_size
    bos_id = models[0].bos_id
    device = memories[0].device
    for _step in range(target_length):
        candidates: list[tuple[tuple[int, ...], float]] = []
        for sequence, score in beams:
            decoder_input = torch.tensor(
                [[bos_id, *sequence]], dtype=torch.long, device=device
            )
            posterior = None
            for model, memory in zip(models, memories):
                logits = model.decode(decoder_input, memory, source_padding)[:, -1, :]
                current = torch.softmax(logits, dim=-1)
                posterior = current if posterior is None else posterior + current
            assert posterior is not None
            posterior = posterior / len(models)
            logp = torch.log(posterior.clamp_min(1e-12))[0].detach().cpu().tolist()
            for symbol in range(alphabet_size):
                candidates.append((sequence + (symbol,), score + float(logp[symbol])))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        beams = candidates[:beam_width]
    best_sequence, best_score = beams[0]
    return list(best_sequence), best_score / target_length


@torch.inference_mode()
def teacher_scores(
    models: list[NeuralTransducer],
    memories: list[torch.Tensor],
    source_padding: torch.Tensor,
    candidates: list[list[int]],
) -> list[float]:
    results = [-1e6] * len(candidates)
    groups: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        scored_length = min(len(candidate), TARGET_LENGTH)
        if scored_length > 0:
            groups.setdefault(scored_length, []).append(index)
    device = memories[0].device
    bos_id = models[0].bos_id
    for length, indices in groups.items():
        targets = torch.tensor(
            [candidates[index][:length] for index in indices],
            dtype=torch.long,
            device=device,
        )
        bos = torch.full((len(indices), 1), bos_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos, targets[:, :-1]], dim=1)
        posterior = None
        for model, memory in zip(models, memories):
            expanded_memory = memory.expand(len(indices), -1, -1)
            expanded_padding = source_padding.expand(len(indices), -1)
            logits = model.decode(decoder_input, expanded_memory, expanded_padding)
            current = torch.softmax(logits, dim=-1)
            posterior = current if posterior is None else posterior + current
        assert posterior is not None
        posterior = posterior / len(models)
        selected = posterior.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        means = torch.log(selected.clamp_min(1e-12)).mean(dim=1).detach().cpu().tolist()
        for index, value in zip(indices, means):
            results[index] = float(value)
    return results


def k_best_segmentations(boundary_probability: np.ndarray, k: int = LATTICE_PATHS) -> list[dict[str, Any]]:
    probability = np.clip(boundary_probability, EPSILON, 1.0 - EPSILON)
    n = len(probability)
    dp: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(n + 1)]
    dp[0] = [(0.0, ())]
    for end in range(1, n + 1):
        options: list[tuple[float, tuple[int, ...]]] = []
        for width in (1, 2, 3):
            start = end - width
            if start < 0:
                continue
            segment_score = math.log(LENGTH_PRIOR[width]) + math.log(probability[end - 1])
            if width > 1:
                segment_score += float(np.log(1.0 - probability[start : end - 1]).sum())
            for previous_score, previous_lengths in dp[start]:
                options.append(
                    (previous_score + segment_score, previous_lengths + (width,))
                )
        options.sort(key=lambda item: (-item[0], item[1]))
        dp[end] = options[:k]
    if len(dp[n]) < k:
        raise RuntimeError(f"only {len(dp[n])} complete segmentations")
    return [
        {"score": float(score), "lengths": list(lengths)}
        for score, lengths in dp[n][:k]
    ]


def boundaries_from_lengths(lengths: Iterable[int]) -> list[int]:
    result: list[int] = []
    cursor = 0
    for width in lengths:
        cursor += int(width)
        result.append(cursor)
    return result


def pieces_from_lengths(cipher: list[int], lengths: list[int]) -> list[str]:
    text = s3.visible_text(cipher)
    pieces: list[str] = []
    cursor = 0
    for width in lengths:
        right = cursor + int(width)
        pieces.append(text[cursor:right])
        cursor = right
    if cursor != len(text):
        raise RuntimeError("segmentation lengths do not reconstruct ciphertext")
    return pieces


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


def candidate_features(
    plaintext: list[int],
    neural_score: float,
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[float]:
    return [
        float(neural_score),
        unit_language_score(plaintext, inventory, unit_model),
        -abs(len(plaintext) - TARGET_LENGTH) / TARGET_LENGTH,
    ]


def corrupt_replacements(target: list[int], rate: float, alphabet_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    result = list(target)
    count = max(1, round(rate * len(result)))
    for position in rng.sample(range(len(result)), min(count, len(result))):
        current = result[position]
        replacement = rng.randrange(alphabet_size - 1)
        if replacement >= current:
            replacement += 1
        result[position] = replacement
    return result


def corrupt_deletion(target: list[int], rate: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    count = max(1, round(rate * len(target)))
    removed = set(rng.sample(range(len(target)), min(count, len(target) - 1)))
    return [value for index, value in enumerate(target) if index not in removed]


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (features - mean) / std
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive = max(1, int(labels.sum()))
    negative = max(1, len(labels) - positive)
    sample_weight = np.where(
        labels > 0.5,
        len(labels) / (2.0 * positive),
        len(labels) / (2.0 * negative),
    )
    weight_total = float(sample_weight.sum())
    for _ in range(1000):
        logits = np.clip(x @ weights + bias, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = sample_weight * (probability - labels)
        gradient = (x.T @ residual) / weight_total + 0.001 * weights
        gradient_bias = float(residual.sum() / weight_total)
        weights -= 0.05 * gradient
        bias -= 0.05 * gradient_bias
    return {
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "examples": int(len(labels)),
        "positives": positive,
        "negatives": negative,
    }


def calibrated_logit(features: list[float], calibration: dict[str, Any]) -> float:
    vector = np.asarray(features, dtype=np.float64)
    mean = np.asarray(calibration["feature_mean"], dtype=np.float64)
    std = np.asarray(calibration["feature_std"], dtype=np.float64)
    weights = np.asarray(calibration["weights"], dtype=np.float64)
    return float(((vector - mean) / std) @ weights + float(calibration["bias"]))


def build_calibration(
    models: list[NeuralTransducer],
    language: core.LanguageData,
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    generator = SyntheticGenerator(
        language,
        core.stable_seed("v060-s3-selection-calibration", 1731, 1732),
        plaintext_length=TARGET_LENGTH,
    )
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    for index in range(32):
        example = generator.sample()
        source, line_flags, source_padding = example_tensors(example, device)
        memories, _boundary = encode_ensemble(models, source, line_flags, source_padding)
        true = list(example.target)
        candidates = [true]
        candidate_labels = [1.0]
        for rate in (0.05, 0.10, 0.20, 0.35):
            candidates.append(
                corrupt_replacements(
                    true,
                    rate,
                    len(language.alphabet),
                    core.stable_seed("v060-s3-selection-corruption", index, f"replace-{rate}"),
                )
            )
            candidate_labels.append(0.0)
        for rate in (0.05, 0.10):
            candidates.append(
                corrupt_deletion(
                    true,
                    rate,
                    core.stable_seed("v060-s3-selection-corruption", index, f"delete-{rate}"),
                )
            )
            candidate_labels.append(0.0)
        neural = teacher_scores(models, memories, source_padding, candidates)
        for candidate, neural_score, label in zip(candidates, neural, candidate_labels):
            feature_rows.append(
                candidate_features(candidate, neural_score, inventory, unit_model)
            )
            labels.append(label)
    calibration = fit_logistic(
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
    )
    calibration["generator_seed"] = int(
        core.stable_seed("v060-s3-selection-calibration", 1731, 1732)
    )
    return calibration


def load_language(repo: Path) -> core.LanguageData:
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        repo / ".cache" / "v060-family-s3-neural-final",
    )
    return languages["en"]


def phase1(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("phase1 requires CUDA")
    signer_url = args.signer_url
    with tempfile.TemporaryDirectory(prefix="v060-s3-final-phase1-") as directory:
        models, manifests = load_models(device, Path(directory), signer_url)
        language = load_language(args.repo)
        inventory = s1.candidate_inventory(language)
        unit_model = s2.build_unit_model(language, inventory)
        calibration = build_calibration(models, language, inventory, unit_model, device)
        trials = [s1.make_trial(language, "dev", TARGET_LENGTH, replicate) for replicate in range(16)]
        rows: list[dict[str, Any]] = []
        for trial in trials:
            source, line_flags, source_padding = trial_tensors(trial, device)
            memories, boundary_probability = encode_ensemble(
                models, source, line_flags, source_padding
            )
            direct_plaintext, beam_score = beam_decode(
                models, memories, source_padding
            )
            segmentations = k_best_segmentations(boundary_probability)
            rows.append(
                {
                    "replicate": trial.replicate,
                    "trial_seed": trial.seed,
                    "cipher_length": len(trial.cipher),
                    "direct_plaintext": direct_plaintext,
                    "direct_beam_mean_logp": beam_score,
                    "segmentations": segmentations,
                }
            )
            print(
                "V060_S3_PHASE1_TRIAL",
                json.dumps(
                    {
                        "replicate": trial.replicate,
                        "cipher_length": len(trial.cipher),
                        "segmentations": len(segmentations),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        payload: dict[str, Any] = {
            "phase": 1,
            "split": "dev",
            "length": TARGET_LENGTH,
            "replicates": 16,
            "checkpoint_sha256": {
                str(seed): manifests[index]["original_sha256"]
                for index, seed in enumerate((1731, 1732))
            },
            "calibration": calibration,
            "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload["sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
        upload_bytes(PHASE1_OBJECT, json_bytes(payload, pretty=True), signer_url)
        print(
            "V060_S3_PHASE1_COMPLETE",
            json.dumps(
                {
                    "object_path": PHASE1_OBJECT,
                    "sha256": payload["sha256"],
                    "trials": len(rows),
                    "calibration_examples": calibration["examples"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def phase2_trial(
    trial: s1.SegmentationTrial,
    phase1_row: dict[str, Any],
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    for rank, segmentation in enumerate(phase1_row["segmentations"]):
        lengths = [int(value) for value in segmentation["lengths"]]
        decoded = s3.decode_candidate(
            pieces_from_lengths(trial.cipher, lengths),
            inventory,
            unit_model,
            trial.seed,
            f"neural-screen-rank-{rank}",
            700000,
            50,
        )
        if decoded is not None:
            candidates.append(
                {
                    "rank": rank,
                    "lengths": lengths,
                    "boundary_posterior_score": float(segmentation["score"]),
                    "combined_score": float(decoded["combined_score"]),
                }
            )
    if not candidates:
        raise RuntimeError(f"no valid lattice candidate for replicate {trial.replicate}")
    candidates.sort(key=lambda row: (-row["combined_score"], row["rank"]))
    selected = candidates[0]
    final = s3.decode_candidate(
        pieces_from_lengths(trial.cipher, selected["lengths"]),
        inventory,
        unit_model,
        trial.seed,
        f"neural-final-rank-{selected['rank']}",
        700000,
        200,
    )
    if final is None:
        raise RuntimeError(f"final lattice candidate invalid for replicate {trial.replicate}")
    return {
        "replicate": trial.replicate,
        "selected_rank": selected["rank"],
        "selected_lengths": selected["lengths"],
        "selected_screen_combined_score": selected["combined_score"],
        "lattice_plaintext": final["plaintext"],
        "lattice_final_combined_score": float(final["combined_score"]),
        "valid_screen_candidates": len(candidates),
        "elapsed_seconds": time.perf_counter() - started,
    }


def phase2(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    signer_url = args.signer_url
    phase1_payload = download_json(PHASE1_OBJECT, signer_url)
    language = load_language(args.repo)
    inventory = s1.candidate_inventory(language)
    unit_model = s2.build_unit_model(language, inventory)
    trials = [s1.make_trial(language, "dev", TARGET_LENGTH, replicate) for replicate in range(16)]
    phase1_by_replicate = {
        int(row["replicate"]): row for row in phase1_payload["rows"]
    }

    def run_one(trial: s1.SegmentationTrial) -> dict[str, Any]:
        row = phase2_trial(
            trial,
            phase1_by_replicate[trial.replicate],
            inventory,
            unit_model,
        )
        print(
            "V060_S3_PHASE2_TRIAL",
            json.dumps(
                {
                    "replicate": row["replicate"],
                    "selected_rank": row["selected_rank"],
                    "valid_screen_candidates": row["valid_screen_candidates"],
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(executor.map(run_one, trials))
    rows.sort(key=lambda row: int(row["replicate"]))
    payload: dict[str, Any] = {
        "phase": 2,
        "split": "dev",
        "phase1_sha256": phase1_payload["sha256"],
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
    upload_bytes(PHASE2_OBJECT, json_bytes(payload, pretty=True), signer_url)
    print(
        "V060_S3_PHASE2_COMPLETE",
        json.dumps(
            {
                "object_path": PHASE2_OBJECT,
                "sha256": payload["sha256"],
                "trials": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def plaintext_accuracy(truth: list[int], predicted: list[int]) -> float:
    return max(
        0.0,
        1.0 - Levenshtein.distance(truth, predicted) / max(1, len(truth), len(predicted)),
    )


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
        "selection": {
            "direct_count": sum(row["selected_hypothesis"] == "direct" for row in rows),
            "lattice_count": sum(row["selected_hypothesis"] == "lattice" for row in rows),
        },
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


def phase3(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("phase3 requires CUDA")
    signer_url = args.signer_url
    phase1_payload = download_json(PHASE1_OBJECT, signer_url)
    phase2_payload = download_json(PHASE2_OBJECT, signer_url)
    with tempfile.TemporaryDirectory(prefix="v060-s3-final-phase3-") as directory:
        models, manifests = load_models(device, Path(directory), signer_url)
        language = load_language(args.repo)
        inventory = s1.candidate_inventory(language)
        unit_model = s2.build_unit_model(language, inventory)
        calibration = phase1_payload["calibration"]
        phase1_by_replicate = {
            int(row["replicate"]): row for row in phase1_payload["rows"]
        }
        phase2_by_replicate = {
            int(row["replicate"]): row for row in phase2_payload["rows"]
        }
        rows: list[dict[str, Any]] = []
        for replicate in range(16):
            trial = s1.make_trial(language, "dev", TARGET_LENGTH, replicate)
            p1 = phase1_by_replicate[replicate]
            p2 = phase2_by_replicate[replicate]
            direct = [int(value) for value in p1["direct_plaintext"]]
            lattice = [int(value) for value in p2["lattice_plaintext"]]
            source, line_flags, source_padding = trial_tensors(trial, device)
            memories, _boundary_probability = encode_ensemble(
                models, source, line_flags, source_padding
            )
            neural_scores = teacher_scores(
                models, memories, source_padding, [direct, lattice]
            )
            direct_features = candidate_features(
                direct, neural_scores[0], inventory, unit_model
            )
            lattice_features = candidate_features(
                lattice, neural_scores[1], inventory, unit_model
            )
            direct_logit = calibrated_logit(direct_features, calibration)
            lattice_logit = calibrated_logit(lattice_features, calibration)
            if lattice_logit > direct_logit:
                selected_name = "lattice"
                selected = lattice
                selected_logit = lattice_logit
            else:
                selected_name = "direct"
                selected = direct
                selected_logit = direct_logit
            top_lengths = [
                int(value) for value in p1["segmentations"][0]["lengths"]
            ]
            predicted_boundaries = boundaries_from_lengths(top_lengths)
            boundary_f1 = s3.boundary_f1(trial.boundaries, predicted_boundaries)
            row = {
                "iso": "en",
                "split": "dev",
                "replicate": replicate,
                "cipher_length": len(trial.cipher),
                "true_units": len(trial.units),
                "top_boundary_units": len(top_lengths),
                "boundary_f1": boundary_f1,
                "selected_hypothesis": selected_name,
                "selected_calibrated_logit": selected_logit,
                "direct_calibrated_logit": direct_logit,
                "lattice_calibrated_logit": lattice_logit,
                "direct_length": len(direct),
                "lattice_length": len(lattice),
                "selected_length": len(selected),
                "direct_accuracy": plaintext_accuracy(trial.plain, direct),
                "lattice_accuracy": plaintext_accuracy(trial.plain, lattice),
                "plaintext_accuracy": plaintext_accuracy(trial.plain, selected),
                "plaintext_exact": selected == trial.plain,
            }
            rows.append(row)
            print("V060_S3_FINAL_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        summary = summarize(rows)
        payload: dict[str, Any] = {
            "config": {
                "split": "dev",
                "length": TARGET_LENGTH,
                "replicates": 16,
                "beam_width": BEAM_WIDTH,
                "lattice_paths": LATTICE_PATHS,
                "screen_iterations": 700000,
                "screen_restarts": 50,
                "final_iterations": 700000,
                "final_restarts": 200,
                "checkpoint_sha256": {
                    str(seed): manifests[index]["original_sha256"]
                    for index, seed in enumerate((1731, 1732))
                },
                "phase1_sha256": phase1_payload["sha256"],
                "phase2_sha256": phase2_payload["sha256"],
            },
            "calibration": calibration,
            "rows": rows,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload["sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
        upload_bytes(FINAL_OBJECT, json_bytes(payload, pretty=True), signer_url)
        print("V060_S3_FINAL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
        print("V060_S3_FINAL_SHA256", payload["sha256"], flush=True)
        print("V060_S3_FINAL_RESULT", json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase1", "phase2", "phase3"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--signer-url", default=DEFAULT_SIGNER_URL)
    args = parser.parse_args()
    if args.phase == "phase1":
        phase1(args)
    elif args.phase == "phase2":
        phase2(args)
    else:
        phase3(args)


if __name__ == "__main__":
    main()
