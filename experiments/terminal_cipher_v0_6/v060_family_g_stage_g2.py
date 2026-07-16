#!/usr/bin/env python3
"""Family G2 blind carrier detection and recovery development solver.

The solver enumerates the complete carrier inventory frozen in
V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md.  It evaluates every candidate
with train-only language, recurrence, compression and alphabet-utilisation
features, refines a fixed shortlist with fresh-mono search, calibrates the
family-level maximum against 256 matched null covers, and either returns one
carrier or abstains.

Only English development data are accessible.  No locked-test or Voynich input
path exists in this program.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_g_stage_g1 as g1

PAYLOAD_LENGTH = 96
SHORTLIST_IDENTITY = 4
SHORTLIST_INVARIANT = 12
SCREEN_MONO_ITERATIONS = 50_000
SCREEN_MONO_RESTARTS = 5
FINAL_MONO_ITERATIONS = 700_000
FINAL_MONO_RESTARTS = 50
NULLS_PER_PAYLOAD = 4
RECURRENCE_CAP = 32


@dataclasses.dataclass(frozen=True)
class Candidate:
    carrier: str
    parameters: tuple[tuple[str, Any], ...]

    def parameter_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.parameters}

    def identity(self) -> str:
        return json.dumps(
            {"carrier": self.carrier, "parameters": self.parameter_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclasses.dataclass
class ReferenceStats:
    identity_mean: float
    identity_std: float
    recurrence_mean: float
    recurrence_std: float
    entropy_mean: float
    entropy_std: float
    collision_mean: float
    collision_std: float
    compression_mean: float
    compression_std: float
    recurrence_trigram: np.ndarray
    recurrence_unigram: np.ndarray


def freeze_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def make_candidate(carrier: str, **parameters: Any) -> Candidate:
    return Candidate(
        carrier=carrier,
        parameters=tuple(
            sorted((key, freeze_value(value)) for key, value in parameters.items())
        ),
    )


def candidate_inventory() -> list[Candidate]:
    inventory: list[Candidate] = []
    for scope, edge in (
        ("line", "first"),
        ("line", "last"),
        ("token", "first"),
        ("token", "last"),
    ):
        inventory.append(
            make_candidate("acrostic_telestic", scope=scope, edge=edge)
        )
    for token_k in range(1, 6):
        for edge in ("first", "last"):
            inventory.append(
                make_candidate("fixed_token", token_k=token_k, edge=edge)
            )
    for unit in ("character", "token"):
        for period in range(2, 13):
            for offset in range(period):
                inventory.append(
                    make_candidate(
                        "regular", unit=unit, period=period, offset=offset
                    )
                )
    densities = (0.10, 0.20, 0.30, 0.40)
    for width in range(4, 13):
        mask_sizes = sorted(
            {
                min(8, max(1, int(round(width * density))))
                for density in densities
            }
        )
        for mask_size in mask_sizes:
            for mask in itertools.combinations(range(width), mask_size):
                inventory.append(
                    make_candidate("grille", width=width, mask=mask)
                )
    identities = [candidate.identity() for candidate in inventory]
    if len(identities) != len(set(identities)):
        raise RuntimeError("candidate inventory contains duplicates")
    return inventory


def inventory_manifest(inventory: list[Candidate]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for candidate in inventory:
        counts[candidate.carrier] = counts.get(candidate.carrier, 0) + 1
    payload = {
        "counts": counts,
        "total": len(inventory),
        "candidates": [candidate.identity() for candidate in inventory],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def recurrence_signature(values: Iterable[int], cap: int = RECURRENCE_CAP) -> list[int]:
    last: dict[int, int] = {}
    out: list[int] = []
    for index, raw in enumerate(values):
        value = int(raw)
        previous = last.get(value)
        out.append(0 if previous is None else min(cap, index - previous))
        last[value] = index
    return out


def build_recurrence_model(
    chunks: list[list[int]], alpha: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    size = RECURRENCE_CAP + 1
    trigrams = np.full((size, size, size), alpha, dtype=np.float64)
    contexts = np.full((size, size), alpha * size, dtype=np.float64)
    unigrams = np.full(size, alpha, dtype=np.float64)
    total = alpha * size
    for chunk in chunks:
        signature = recurrence_signature(chunk)
        for value in signature:
            unigrams[value] += 1.0
            total += 1.0
        for first, second, third in zip(signature, signature[1:], signature[2:]):
            trigrams[first, second, third] += 1.0
            contexts[first, second] += 1.0
    return np.log(trigrams / contexts[:, :, None]), np.log(unigrams / total)


def average_lm_score(
    values: list[int], trigram: np.ndarray, unigram: np.ndarray
) -> float:
    if not values:
        return -1e9
    array = np.asarray(values, dtype=np.int32)
    identity = np.arange(trigram.shape[0], dtype=np.int32)
    return float(mono.score_key(array, identity, trigram, unigram)) / len(values)


def recurrence_lm_score(
    values: list[int], trigram: np.ndarray, unigram: np.ndarray
) -> float:
    signature = recurrence_signature(values)
    if not signature:
        return -1e9
    score = 0.15 * float(unigram[signature[0]])
    if len(signature) >= 2:
        score += 0.15 * float(unigram[signature[1]])
    for first, second, third in zip(signature, signature[1:], signature[2:]):
        score += float(trigram[first, second, third])
        score += 0.15 * float(unigram[third])
    return score / len(signature)


def entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = np.bincount(np.asarray(values, dtype=np.int32))
    probabilities = counts[counts > 0].astype(np.float64) / len(values)
    return float(-(probabilities * np.log(probabilities)).sum())


def collision_rate(values: list[int]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    counts = np.bincount(np.asarray(values, dtype=np.int32))
    collisions = int(sum(int(count) * (int(count) - 1) // 2 for count in counts))
    return collisions / (n * (n - 1) / 2.0)


def compression_ratio(values: list[int]) -> float:
    if not values:
        return 1.0
    raw = bytes(int(value) & 0xFF for value in values)
    return len(zlib.compress(raw, level=9)) / len(raw)


def safe_std(values: list[float]) -> float:
    value = statistics.pstdev(values)
    return value if value > 1e-9 else 1.0


def build_reference_stats(
    language: core.LanguageData,
    identity_model: tuple[np.ndarray, np.ndarray],
) -> ReferenceStats:
    chunks = core.source_chunks(language, "train", PAYLOAD_LENGTH)
    if len(chunks) < 512:
        raise RuntimeError(f"insufficient train chunks for G2 reference: {len(chunks)}")
    reference_chunks = [list(chunk) for chunk in chunks[:512]]
    recurrence_model = build_recurrence_model(reference_chunks)
    identity_values = [
        average_lm_score(chunk, *identity_model) for chunk in reference_chunks
    ]
    recurrence_values = [
        recurrence_lm_score(chunk, *recurrence_model) for chunk in reference_chunks
    ]
    entropy_values = [entropy(chunk) for chunk in reference_chunks]
    collision_values = [collision_rate(chunk) for chunk in reference_chunks]
    compression_values = [compression_ratio(chunk) for chunk in reference_chunks]
    return ReferenceStats(
        identity_mean=statistics.fmean(identity_values),
        identity_std=safe_std(identity_values),
        recurrence_mean=statistics.fmean(recurrence_values),
        recurrence_std=safe_std(recurrence_values),
        entropy_mean=statistics.fmean(entropy_values),
        entropy_std=safe_std(entropy_values),
        collision_mean=statistics.fmean(collision_values),
        collision_std=safe_std(collision_values),
        compression_mean=statistics.fmean(compression_values),
        compression_std=safe_std(compression_values),
        recurrence_trigram=recurrence_model[0],
        recurrence_unigram=recurrence_model[1],
    )


def z(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def candidate_features(
    values: list[int],
    identity_model: tuple[np.ndarray, np.ndarray],
    reference: ReferenceStats,
) -> dict[str, float]:
    identity_average = average_lm_score(values, *identity_model)
    recurrence_average = recurrence_lm_score(
        values, reference.recurrence_trigram, reference.recurrence_unigram
    )
    entropy_value = entropy(values)
    collision_value = collision_rate(values)
    compression_value = compression_ratio(values)
    identity_z = z(identity_average, reference.identity_mean, reference.identity_std)
    recurrence_z = z(
        recurrence_average, reference.recurrence_mean, reference.recurrence_std
    )
    invariant_distance = (
        z(entropy_value, reference.entropy_mean, reference.entropy_std) ** 2
        + z(
            collision_value,
            reference.collision_mean,
            reference.collision_std,
        )
        ** 2
        + z(
            compression_value,
            reference.compression_mean,
            reference.compression_std,
        )
        ** 2
    )
    invariant_score = recurrence_z - 0.25 * invariant_distance
    return {
        "identity_average": identity_average,
        "recurrence_average": recurrence_average,
        "entropy": entropy_value,
        "collision_rate": collision_value,
        "compression_ratio": compression_value,
        "identity_z": identity_z,
        "recurrence_z": recurrence_z,
        "invariant_distance": invariant_distance,
        "invariant_score": invariant_score,
    }


def extract_candidate(
    layout: g1.CoverLayout, candidate: Candidate, limit: int = PAYLOAD_LENGTH
) -> list[int]:
    params = candidate.parameter_dict()
    out: list[int] = []

    if candidate.carrier == "acrostic_telestic":
        scope = params["scope"]
        edge = params["edge"]
        if scope == "line":
            for line in layout.lines:
                out.append(line[0] if edge == "first" else line[-1])
                if len(out) >= limit:
                    return out
        else:
            for line, spans in zip(layout.lines, layout.token_spans):
                for start, end in spans:
                    out.append(line[start] if edge == "first" else line[end - 1])
                    if len(out) >= limit:
                        return out
        return out

    if candidate.carrier == "fixed_token":
        token_k = int(params["token_k"])
        edge = params["edge"]
        for line, spans in zip(layout.lines, layout.token_spans):
            start, end = spans[token_k - 1]
            out.append(line[start] if edge == "first" else line[end - 1])
            if len(out) >= limit:
                return out
        return out

    if candidate.carrier == "regular":
        period = int(params["period"])
        offset = int(params["offset"])
        if params["unit"] == "character":
            ordinal = 0
            for line in layout.lines:
                for value in line:
                    if ordinal % period == offset:
                        out.append(int(value))
                        if len(out) >= limit:
                            return out
                    ordinal += 1
        else:
            ordinal = 0
            for line, spans in zip(layout.lines, layout.token_spans):
                for start, end in spans:
                    if ordinal % period == offset:
                        for value in line[start:end]:
                            out.append(int(value))
                            if len(out) >= limit:
                                return out
                    ordinal += 1
        return out

    if candidate.carrier == "grille":
        width = int(params["width"])
        mask = set(int(value) for value in params["mask"])
        for line in layout.lines:
            for column, value in enumerate(line):
                if column % width in mask:
                    out.append(int(value))
                    if len(out) >= limit:
                        return out
        return out

    raise ValueError(candidate.carrier)


def normalise_parameters(carrier: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if carrier == "grille":
        return {
            "width": int(parameters["width"]),
            "mask": tuple(int(value) for value in parameters["mask"]),
        }
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in parameters.items()
    }


def screen_mono_candidate(
    values: list[int],
    language: core.LanguageData,
    identity_model: tuple[np.ndarray, np.ndarray],
    reference: ReferenceStats,
    seed: int,
) -> dict[str, Any]:
    trigram, unigram = identity_model
    array = np.asarray(values, dtype=np.int32)
    initial = mono.frequency_key(values, language)
    solved_key, solved_score = mono.anneal_mono(
        array,
        initial,
        trigram,
        unigram,
        SCREEN_MONO_ITERATIONS,
        SCREEN_MONO_RESTARTS,
        int(seed & 0x7FFFFFFFFFFFFFFF),
    )
    active = len(set(values))
    penalty_per_char = (
        0.5 * max(1, active - 1) * math.log(max(2, len(values))) / len(values)
    )
    mono_average = float(solved_score) / len(values)
    mono_evidence = z(
        mono_average - penalty_per_char,
        reference.identity_mean,
        reference.identity_std,
    )
    prediction = solved_key[array].astype(np.int32).tolist()
    return {
        "mono_average": mono_average,
        "mono_penalty_per_char": penalty_per_char,
        "mono_evidence": mono_evidence,
        "key": solved_key,
        "prediction": prediction,
    }


def solve_cover(
    layout: g1.CoverLayout,
    inventory: list[Candidate],
    language: core.LanguageData,
    identity_model: tuple[np.ndarray, np.ndarray],
    reference: ReferenceStats,
    seed: int,
    refine_prediction: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    screened: list[dict[str, Any]] = []
    for index, candidate in enumerate(inventory):
        extracted = extract_candidate(layout, candidate)
        if len(extracted) != PAYLOAD_LENGTH:
            continue
        features = candidate_features(extracted, identity_model, reference)
        screened.append(
            {
                "index": index,
                "candidate": candidate,
                "extracted": extracted,
                "features": features,
            }
        )
    if len(screened) != len(inventory):
        raise RuntimeError(
            f"only {len(screened)} of {len(inventory)} frozen candidates had capacity"
        )

    identity_rank = sorted(
        screened, key=lambda row: row["features"]["identity_z"], reverse=True
    )[:SHORTLIST_IDENTITY]
    invariant_rank = sorted(
        screened,
        key=lambda row: row["features"]["invariant_score"],
        reverse=True,
    )[:SHORTLIST_INVARIANT]
    shortlist: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in identity_rank + invariant_rank:
        if row["index"] not in seen:
            shortlist.append(row)
            seen.add(row["index"])

    refined: list[dict[str, Any]] = []
    for row in shortlist:
        mono_result = screen_mono_candidate(
            row["extracted"],
            language,
            identity_model,
            reference,
            core.stable_seed("v060-g2-screen-mono", seed, row["index"]),
        )
        identity_evidence = row["features"]["identity_z"]
        mono_evidence = mono_result["mono_evidence"]
        if mono_evidence > identity_evidence:
            selected_arm = "mono"
            primary = mono_evidence
            prediction = mono_result["prediction"]
        else:
            selected_arm = "plaintext"
            primary = identity_evidence
            prediction = list(row["extracted"])
        evidence = primary + 0.15 * row["features"]["invariant_score"]
        refined.append(
            row
            | {
                "mono": mono_result,
                "selected_arm": selected_arm,
                "prediction": prediction,
                "evidence": evidence,
            }
        )
    best = max(refined, key=lambda row: row["evidence"])

    if refine_prediction and best["selected_arm"] == "mono":
        trigram, unigram = identity_model
        array = np.asarray(best["extracted"], dtype=np.int32)
        final_key, final_score = mono.anneal_mono(
            array,
            best["mono"]["key"],
            trigram,
            unigram,
            FINAL_MONO_ITERATIONS,
            FINAL_MONO_RESTARTS,
            int(core.stable_seed("v060-g2-final-mono", seed, best["index"]) & 0x7FFFFFFFFFFFFFFF),
        )
        best["prediction"] = final_key[array].astype(np.int32).tolist()
        best["final_mono_average"] = float(final_score) / len(best["extracted"])

    candidate = best["candidate"]
    return {
        "candidate_index": best["index"],
        "carrier": candidate.carrier,
        "parameters": candidate.parameter_dict(),
        "selected_arm": best["selected_arm"],
        "evidence": float(best["evidence"]),
        "prediction": best["prediction"],
        "features": best["features"],
        "shortlist_size": len(shortlist),
        "elapsed_seconds": time.perf_counter() - started,
    }


def embed_payload_cover(
    language: core.LanguageData,
    generator: str,
    carrier: str,
    replicate: int,
    payload: list[int],
) -> tuple[g1.CoverLayout, dict[str, Any]]:
    seed = core.stable_seed("v060-g1", generator, carrier, replicate)
    rng = random.Random(seed)
    layout = g1.make_cover(language, generator, rng)
    positions, parameters = g1.carrier_positions(
        layout, generator, carrier, replicate, rng
    )
    positions = positions[:PAYLOAD_LENGTH]
    encrypted = (
        g1.parameter_index(generator, replicate) + g1.CARRIER_CLASSES.index(carrier)
    ) % 2 == 1
    embedded = list(payload)
    if encrypted:
        packet = core.encrypt_sequence(
            payload,
            "mono",
            language,
            random.Random(core.stable_seed("v060-g1-key", seed)),
            parameter_mode="dev",
        )
        embedded = list(packet.cipher)
    for (line_index, column), value in zip(positions, embedded):
        layout.lines[line_index][column] = int(value)
    return layout, {
        "seed": seed,
        "generator": generator,
        "true_carrier": carrier,
        "true_parameters": normalise_parameters(carrier, parameters),
        "encrypted": encrypted,
        "payload": payload,
    }


def make_null_cover(
    language: core.LanguageData,
    generator: str,
    carrier: str,
    replicate: int,
    null_replicate: int,
) -> tuple[g1.CoverLayout, dict[str, Any]]:
    seed = core.stable_seed(
        "v060-g2-null", generator, carrier, replicate, null_replicate
    )
    layout = g1.make_cover(language, generator, random.Random(seed))
    return layout, {
        "seed": seed,
        "generator": generator,
        "matched_carrier_cell": carrier,
        "matched_replicate": replicate,
        "null_replicate": null_replicate,
    }


def auroc(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    for left in positive:
        for right in negative:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / max(1, len(positive) * len(negative))


def operating_threshold(null_scores: list[float]) -> float:
    if len(null_scores) < 3:
        raise RuntimeError("at least three null scores are required")
    # Strict greater-than threshold at the third-highest null leaves at most
    # two of 256 nulls above threshold: empirical FPR <= 0.78125%.
    return sorted(null_scores)[-3]


def summarise(
    payload_rows: list[dict[str, Any]], null_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    payload_scores = [float(row["evidence"]) for row in payload_rows]
    null_scores = [float(row["evidence"]) for row in null_rows]
    threshold = operating_threshold(null_scores)
    for row in payload_rows:
        row["detected"] = float(row["evidence"]) > threshold
    for row in null_rows:
        row["false_positive"] = float(row["evidence"]) > threshold

    detected = [row for row in payload_rows if row["detected"]]
    fpr = statistics.fmean(row["false_positive"] for row in null_rows)
    recoveries = [float(row["recovery"]) if row["detected"] else 0.0 for row in payload_rows]
    summary = {
        "payload_covers": len(payload_rows),
        "null_covers": len(null_rows),
        "threshold": threshold,
        "auroc": auroc(payload_scores, null_scores),
        "false_positive_rate": fpr,
        "detected_payload_count": len(detected),
        "carrier_class_accuracy_detected": (
            statistics.fmean(row["carrier_correct"] for row in detected)
            if detected
            else 0.0
        ),
        "exact_parameter_accuracy_detected": (
            statistics.fmean(row["parameters_correct"] for row in detected)
            if detected
            else 0.0
        ),
        "mean_recovery_with_abstention_as_zero": statistics.fmean(recoveries),
        "at_least_70_count": sum(value >= 0.70 for value in recoveries),
        "minimum_detected_recovery": (
            min(float(row["recovery"]) for row in detected) if detected else 0.0
        ),
    }
    full_grid = len(payload_rows) == 64 and len(null_rows) == 256
    summary["full_frozen_grid"] = full_grid
    summary["gate"] = {
        "pass": bool(
            full_grid
            and summary["auroc"] >= 0.95
            and summary["false_positive_rate"] <= 0.01
            and summary["carrier_class_accuracy_detected"] >= 0.85
            and summary["exact_parameter_accuracy_detected"] >= 0.75
            and summary["mean_recovery_with_abstention_as_zero"] >= 0.80
            and summary["at_least_70_count"] >= 54
        )
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", choices=g1.COVER_GENERATORS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--execution-smoke", action="store_true")
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-g2",
    )
    language = languages["en"]
    identity_model = mono.build_language_model(language)
    reference = build_reference_stats(language, identity_model)
    inventory = candidate_inventory()
    manifest = inventory_manifest(inventory)
    if manifest["total"] != 2935:
        raise RuntimeError(f"unexpected frozen inventory size {manifest['total']} != 2935")
    print(
        "V060_G2_INVENTORY",
        json.dumps(
            {
                "total": manifest["total"],
                "counts": manifest["counts"],
                "sha256": manifest["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    chunks = core.source_chunks(language, "dev", PAYLOAD_LENGTH)
    if len(chunks) < 64:
        raise RuntimeError(f"Family G2 requires 64 disjoint dev chunks; found {len(chunks)}")
    generators = (args.generator,) if args.generator else g1.COVER_GENERATORS

    payload_specs: list[tuple[str, str, int, list[int]]] = []
    null_specs: list[tuple[str, str, int, int]] = []
    for generator in generators:
        for carrier in g1.CARRIER_CLASSES:
            for replicate in range(4):
                chunk_index = (
                    g1.COVER_GENERATORS.index(generator) * 16
                    + g1.CARRIER_CLASSES.index(carrier) * 4
                    + replicate
                )
                payload_specs.append(
                    (generator, carrier, replicate, list(chunks[chunk_index]))
                )
                for null_replicate in range(NULLS_PER_PAYLOAD):
                    null_specs.append(
                        (generator, carrier, replicate, null_replicate)
                    )
    if args.execution_smoke:
        payload_specs = payload_specs[:1]
        null_specs = null_specs[:1]

    def run_payload(spec: tuple[str, str, int, list[int]]) -> dict[str, Any]:
        generator, carrier, replicate, payload = spec
        layout, truth = embed_payload_cover(
            language, generator, carrier, replicate, payload
        )
        solved = solve_cover(
            layout,
            inventory,
            language,
            identity_model,
            reference,
            truth["seed"],
            refine_prediction=True,
        )
        selected_parameters = normalise_parameters(
            solved["carrier"], solved["parameters"]
        )
        row = truth | solved
        row["carrier_correct"] = solved["carrier"] == truth["true_carrier"]
        row["parameters_correct"] = (
            row["carrier_correct"]
            and selected_parameters == truth["true_parameters"]
        )
        row["recovery"] = mono.fast_accuracy(payload, solved["prediction"])
        row["selected_status_correct"] = solved["selected_arm"] == (
            "mono" if truth["encrypted"] else "plaintext"
        )
        row.pop("payload", None)
        row.pop("prediction", None)
        print("V060_G2_PAYLOAD", json.dumps(row, sort_keys=True), flush=True)
        return row

    def run_null(spec: tuple[str, str, int, int]) -> dict[str, Any]:
        layout, metadata = make_null_cover(language, *spec)
        solved = solve_cover(
            layout,
            inventory,
            language,
            identity_model,
            reference,
            metadata["seed"],
            refine_prediction=False,
        )
        solved.pop("prediction", None)
        row = metadata | solved
        print("V060_G2_NULL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, max(1, len(payload_specs)))
    ) as executor:
        payload_rows = list(executor.map(run_payload, payload_specs))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, max(1, len(null_specs)))
    ) as executor:
        null_rows = list(executor.map(run_null, null_specs))

    if args.execution_smoke:
        summary = {
            "execution_smoke": True,
            "payload_rows": len(payload_rows),
            "null_rows": len(null_rows),
            "inventory_total": manifest["total"],
            "inventory_sha256": manifest["sha256"],
        }
    else:
        summary = summarise(payload_rows, null_rows)

    payload = {
        "config": {
            "generator": args.generator,
            "payload_length": PAYLOAD_LENGTH,
            "nulls_per_payload": NULLS_PER_PAYLOAD,
            "shortlist_identity": SHORTLIST_IDENTITY,
            "shortlist_invariant": SHORTLIST_INVARIANT,
            "screen_mono_iterations": SCREEN_MONO_ITERATIONS,
            "screen_mono_restarts": SCREEN_MONO_RESTARTS,
            "final_mono_iterations": FINAL_MONO_ITERATIONS,
            "final_mono_restarts": FINAL_MONO_RESTARTS,
            "execution_smoke": args.execution_smoke,
            "protocol": "V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md",
        },
        "inventory": {
            "total": manifest["total"],
            "counts": manifest["counts"],
            "sha256": manifest["sha256"],
        },
        "payload_rows": payload_rows,
        "null_rows": null_rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_G2_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_G2_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
