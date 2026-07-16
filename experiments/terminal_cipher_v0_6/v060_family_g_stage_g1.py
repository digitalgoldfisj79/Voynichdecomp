#!/usr/bin/env python3
"""Family G1 oracle-carrier synthetic calibration.

Implements the four carrier classes frozen in
V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md.  The true carrier and its
parameters are supplied to the extractor; plaintext-versus-fresh-mono status
and the mono key remain hidden.  No test or Voynich data are accessible here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono

COVER_GENERATORS = tuple(core.CONTROL_FAMILIES)
CARRIER_CLASSES = ("acrostic_telestic", "fixed_token", "regular", "grille")
PAYLOAD_LENGTH = 96
MONO_ITERATIONS = 700_000
MONO_RESTARTS = 50
TOKENS_PER_LINE = 12
LINE_COUNT = 192


@dataclasses.dataclass
class CoverLayout:
    lines: list[list[int]]
    token_spans: list[list[tuple[int, int]]]


def split_tokens(stream: list[int], space: int) -> list[list[int]]:
    tokens: list[list[int]] = []
    current: list[int] = []
    for value in stream:
        if value == space:
            if current:
                tokens.append(current)
                current = []
        else:
            current.append(int(value))
    if current:
        tokens.append(current)
    return tokens


def make_cover(
    language: core.LanguageData,
    generator: str,
    rng: random.Random,
) -> CoverLayout:
    space = language.char_to_id.get(" ", 0)
    target_tokens = LINE_COUNT * TOKENS_PER_LINE
    stream_length = max(30_000, target_tokens * 7)
    stream = core.generate_control(language, generator, stream_length, rng)
    tokens = split_tokens(stream, space)
    if len(tokens) < target_tokens:
        fallback = [list(word) for word in language.train_words if word]
        if not fallback:
            fallback = [[core.weighted_choice(rng, language.probabilities)]]
        while len(tokens) < target_tokens:
            tokens.append(list(rng.choice(fallback)))

    lines: list[list[int]] = []
    spans: list[list[tuple[int, int]]] = []
    cursor = 0
    for _ in range(LINE_COUNT):
        line: list[int] = []
        line_spans: list[tuple[int, int]] = []
        for token_index in range(TOKENS_PER_LINE):
            token = list(tokens[cursor])
            cursor += 1
            if token_index:
                line.append(space)
            start = len(line)
            line.extend(token)
            line_spans.append((start, len(line)))
        lines.append(line)
        spans.append(line_spans)
    return CoverLayout(lines=lines, token_spans=spans)


def parameter_index(generator: str, replicate: int) -> int:
    return COVER_GENERATORS.index(generator) * 4 + replicate


def carrier_positions(
    layout: CoverLayout,
    generator: str,
    carrier: str,
    replicate: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    index = parameter_index(generator, replicate)
    positions: list[tuple[int, int]] = []

    if carrier == "acrostic_telestic":
        variants = (
            ("line", "first"),
            ("line", "last"),
            ("token", "first"),
            ("token", "last"),
        )
        scope, edge = variants[index % len(variants)]
        if scope == "line":
            for line_index, line in enumerate(layout.lines):
                positions.append((line_index, 0 if edge == "first" else len(line) - 1))
        else:
            for line_index, spans in enumerate(layout.token_spans):
                for start, end in spans:
                    positions.append((line_index, start if edge == "first" else end - 1))
        return positions, {"scope": scope, "edge": edge}

    if carrier == "fixed_token":
        k = 1 + (index % 5)
        edge = "first" if (index // 5) % 2 == 0 else "last"
        for line_index, spans in enumerate(layout.token_spans):
            start, end = spans[k - 1]
            positions.append((line_index, start if edge == "first" else end - 1))
        return positions, {"token_k": k, "edge": edge}

    if carrier == "regular":
        unit = "character" if index % 2 == 0 else "token"
        period = 2 + (index % 11)
        offset = (3 * index + replicate) % period
        if unit == "character":
            flat = [
                (line_index, column)
                for line_index, line in enumerate(layout.lines)
                for column in range(len(line))
            ]
            positions = [pos for ordinal, pos in enumerate(flat) if ordinal % period == offset]
        else:
            token_ordinal = 0
            for line_index, spans in enumerate(layout.token_spans):
                for start, end in spans:
                    if token_ordinal % period == offset:
                        positions.extend((line_index, column) for column in range(start, end))
                    token_ordinal += 1
        return positions, {"unit": unit, "period": period, "offset": offset}

    if carrier == "grille":
        width = 4 + (index % 9)
        density = (0.10, 0.20, 0.30, 0.40)[index % 4]
        mask_size = min(8, max(1, int(round(width * density))))
        mask = sorted(rng.sample(range(width), mask_size))
        for line_index, line in enumerate(layout.lines):
            positions.extend(
                (line_index, column)
                for column in range(len(line))
                if column % width in mask
            )
        return positions, {
            "width": width,
            "density": density,
            "mask": mask,
            "mask_size": mask_size,
        }

    raise ValueError(carrier)


def decode_hidden_status(
    extracted: list[int],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    seed: int,
    iterations: int,
    restarts: int,
) -> tuple[list[int], dict[str, Any]]:
    trigram, unigram = model
    cipher = np.asarray(extracted, dtype=np.int32)
    identity = np.arange(len(language.alphabet), dtype=np.int32)
    identity_score = float(mono.score_key(cipher, identity, trigram, unigram))
    initial = mono.frequency_key(extracted, language)
    solved_key, solved_score = mono.anneal_mono(
        cipher,
        initial,
        trigram,
        unigram,
        iterations,
        restarts,
        int(seed & 0x7FFFFFFFFFFFFFFF),
    )
    active = len(set(extracted))
    mdl_penalty = 0.5 * max(1, active - 1) * math.log(max(2, len(extracted)))
    choose_mono = float(solved_score) - identity_score > mdl_penalty
    key = solved_key if choose_mono else identity
    prediction = key[cipher].astype(np.int32).tolist()
    return prediction, {
        "selected_arm": "mono" if choose_mono else "plaintext",
        "identity_score": identity_score,
        "mono_score": float(solved_score),
        "mdl_penalty": mdl_penalty,
        "score_gain_after_penalty": float(solved_score) - identity_score - mdl_penalty,
    }


def run_trial(
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    generator: str,
    carrier: str,
    replicate: int,
    payload: list[int],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = core.stable_seed("v060-g1", generator, carrier, replicate)
    rng = random.Random(seed)
    layout = make_cover(language, generator, rng)
    positions, parameters = carrier_positions(
        layout, generator, carrier, replicate, rng
    )
    if len(positions) < len(payload):
        raise RuntimeError(
            f"insufficient carrier capacity {generator}/{carrier}/{replicate}: "
            f"{len(positions)} < {len(payload)}"
        )
    positions = positions[: len(payload)]

    encrypted = (parameter_index(generator, replicate) + CARRIER_CLASSES.index(carrier)) % 2 == 1
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
    extracted = [layout.lines[line_index][column] for line_index, column in positions]
    if extracted != embedded:
        raise RuntimeError("carrier extraction failed its exact execution invariant")

    prediction, selection = decode_hidden_status(
        extracted,
        language,
        model,
        core.stable_seed("v060-g1-solve", seed),
        iterations,
        restarts,
    )
    accuracy = mono.fast_accuracy(payload, prediction)
    row = {
        "iso": language.iso,
        "split": "dev",
        "generator": generator,
        "carrier": carrier,
        "replicate": replicate,
        "seed": seed,
        "payload_length": len(payload),
        "encrypted": encrypted,
        "carrier_parameters": parameters,
        "carrier_capacity": len(positions),
        "accuracy": accuracy,
        "exact": prediction == payload,
        "status_selected_correctly": selection["selected_arm"] == ("mono" if encrypted else "plaintext"),
        "selection": selection,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["accuracy"]) for row in rows]
    encrypted = [float(row["accuracy"]) for row in rows if row["encrypted"]]
    full_grid = len(rows) == 64
    summary = {
        "trials": len(rows),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "at_least_85_count": sum(value >= 0.85 for value in values),
        "encrypted_trials": len(encrypted),
        "encrypted_mean": statistics.fmean(encrypted) if encrypted else None,
        "status_selection_accuracy": statistics.fmean(
            row["status_selected_correctly"] for row in rows
        ),
        "full_frozen_grid": full_grid,
    }
    summary["gate"] = {
        "pass": bool(
            full_grid
            and summary["mean"] >= 0.90
            and summary["minimum"] >= 0.70
            and summary["at_least_85_count"] >= 58
            and summary["encrypted_mean"] is not None
            and summary["encrypted_mean"] >= 0.85
        )
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", choices=COVER_GENERATORS)
    parser.add_argument("--carrier", choices=CARRIER_CLASSES)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-end", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execution-smoke", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.replicate_start < args.replicate_end <= 4:
        raise ValueError("replicate range must satisfy 0 <= start < end <= 4")

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-g1",
    )
    language = languages["en"]
    model = mono.build_language_model(language)
    chunks = core.source_chunks(language, "dev", PAYLOAD_LENGTH)
    if len(chunks) < 64:
        raise RuntimeError(f"Family G1 requires 64 disjoint dev chunks; found {len(chunks)}")

    generators = (args.generator,) if args.generator else COVER_GENERATORS
    carriers = (args.carrier,) if args.carrier else CARRIER_CLASSES
    specs: list[tuple[str, str, int, list[int]]] = []
    for generator in generators:
        for carrier in carriers:
            for replicate in range(args.replicate_start, args.replicate_end):
                chunk_index = (
                    COVER_GENERATORS.index(generator) * 16
                    + CARRIER_CLASSES.index(carrier) * 4
                    + replicate
                )
                specs.append((generator, carrier, replicate, list(chunks[chunk_index])))

    iterations = 500 if args.execution_smoke else MONO_ITERATIONS
    restarts = 1 if args.execution_smoke else MONO_RESTARTS
    if args.execution_smoke:
        specs = specs[:1]

    def execute(spec: tuple[str, str, int, list[int]]) -> dict[str, Any]:
        row = run_trial(language, model, *spec, iterations, restarts)
        print("V060_G1_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(specs))) as executor:
        rows = list(executor.map(execute, specs))

    summary = summarize(rows)
    payload = {
        "config": {
            "generator": args.generator,
            "carrier": args.carrier,
            "replicate_start": args.replicate_start,
            "replicate_end": args.replicate_end,
            "payload_length": PAYLOAD_LENGTH,
            "mono_iterations": iterations,
            "mono_restarts": restarts,
            "execution_smoke": args.execution_smoke,
            "protocol": "V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md",
        },
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_G1_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_G1_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
