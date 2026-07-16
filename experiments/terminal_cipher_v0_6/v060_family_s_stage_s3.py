#!/usr/bin/env python3
"""v0.6 Family S3: joint ciphertext-only segmentation and polygraphic decoding."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_s_stage_s1 as s1
import v060_family_s_stage_s2 as s2

CODE_LENGTH_PROBABILITIES = {1: 0.20, 2: 0.45, 3: 0.35}
MODEL_TYPES = ("unigram", "bpe")
VOCAB_SIZES = (48, 63, 78)


def visible_text(cipher: list[int]) -> str:
    alphabet = "ABCDEFGHIJ"
    return "".join(alphabet[value] for value in cipher)


def train_segmentation(
    text: str, model_type: str, vocab_size: int, working: Path
) -> list[str]:
    input_path = working / "cipher.txt"
    input_path.write_text(text + "\n", encoding="utf-8")
    prefix = working / f"sp-{model_type}-{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(prefix),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=1.0,
        max_sentencepiece_length=3,
        hard_vocab_limit=False,
        split_by_whitespace=False,
        split_by_unicode_script=False,
        split_by_number=False,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        num_threads=1,
        minloglevel=2,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    pieces = processor.encode(text, out_type=str)
    pieces = [piece.replace("▁", "") for piece in pieces if piece.replace("▁", "")]
    if "".join(pieces) != text:
        raise RuntimeError("SentencePiece segmentation does not reconstruct stream")
    if any(len(piece) > 3 for piece in pieces):
        raise RuntimeError("segmentation exceeded frozen maximum code length")
    return pieces


def canonical_symbols(pieces: list[str]) -> tuple[list[int], list[str]]:
    mapping: dict[str, int] = {}
    symbols: list[int] = []
    vocabulary: list[str] = []
    for piece in pieces:
        if piece not in mapping:
            mapping[piece] = len(mapping)
            vocabulary.append(piece)
        symbols.append(mapping[piece])
    return symbols, vocabulary


def boundary_positions(pieces: list[str]) -> list[int]:
    out: list[int] = []
    cursor = 0
    for piece in pieces:
        cursor += len(piece)
        out.append(cursor)
    return out


def boundary_f1(truth: list[int], predicted: list[int]) -> float:
    truth_set = set(truth[:-1])
    predicted_set = set(predicted[:-1])
    tp = len(truth_set & predicted_set)
    precision = tp / max(1, len(predicted_set))
    recall = tp / max(1, len(truth_set))
    return 2 * precision * recall / max(1e-12, precision + recall)


def plaintext_accuracy(truth: list[int], predicted: list[int]) -> float:
    return max(
        0.0,
        1.0 - Levenshtein.distance(truth, predicted) / max(1, len(truth), len(predicted)),
    )


def decode_candidate(
    pieces: list[str],
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
    trial_seed: int,
    label: str,
    iterations: int,
    restarts: int,
) -> dict[str, Any] | None:
    symbols, vocabulary = canonical_symbols(pieces)
    if len(vocabulary) > len(inventory):
        return None
    trigram, unigram, probabilities = unit_model
    initial = s2.frequency_key(symbols, probabilities, len(inventory))
    cipher_array = np.asarray(symbols, dtype=np.int32)
    key, raw_score = mono.anneal_mono(
        cipher_array,
        initial,
        trigram,
        unigram,
        iterations,
        restarts,
        int(core.stable_seed("v060-s3-map", trial_seed, label) & 0x7FFFFFFFFFFFFFFF),
    )
    unit_ids = key[cipher_array].tolist()
    plaintext = s2.expand_units(unit_ids, inventory)
    token_count = len(pieces)
    expected_tokens = sum(len(piece) for piece in pieces) / 2.15
    token_sd = max(3.0, 0.10 * expected_tokens)
    length_logprior = sum(
        math.log(CODE_LENGTH_PROBABILITIES[len(piece)]) for piece in pieces
    )
    normalized_lm = float(raw_score) / max(1, token_count) * expected_tokens
    count_penalty = 0.5 * ((token_count - expected_tokens) / token_sd) ** 2
    vocabulary_penalty = 0.5 * len(vocabulary) * math.log(max(2, token_count))
    combined = normalized_lm + length_logprior - count_penalty - vocabulary_penalty
    return {
        "pieces": pieces,
        "symbols": symbols,
        "vocabulary": vocabulary,
        "key": key,
        "raw_unit_score": float(raw_score),
        "combined_score": float(combined),
        "plaintext": plaintext,
        "token_count": token_count,
        "distinct_pieces": len(vocabulary),
    }


def solve_trial(
    trial: s1.SegmentationTrial,
    inventory: list[tuple[int, ...]],
    unit_model: tuple[np.ndarray, np.ndarray, np.ndarray],
    screen_iterations: int,
    screen_restarts: int,
    final_iterations: int,
    final_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    text = visible_text(trial.cipher)
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"v060-s3-{trial.replicate}-") as directory:
        working = Path(directory)
        for model_type in MODEL_TYPES:
            for vocab_size in VOCAB_SIZES:
                try:
                    pieces = train_segmentation(
                        text, model_type, vocab_size, working / f"{model_type}-{vocab_size}"
                    )
                except Exception as error:
                    candidates.append({
                        "error": f"{type(error).__name__}: {error}",
                        "model_type": model_type,
                        "vocab_size": vocab_size,
                    })
                    continue
                decoded = decode_candidate(
                    pieces,
                    inventory,
                    unit_model,
                    trial.seed,
                    f"screen-{model_type}-{vocab_size}",
                    screen_iterations,
                    screen_restarts,
                )
                if decoded is None:
                    continue
                decoded["model_type"] = model_type
                decoded["vocab_size"] = vocab_size
                candidates.append(decoded)
    valid = [candidate for candidate in candidates if "combined_score" in candidate]
    if not valid:
        raise RuntimeError("no valid segmentation candidate")
    selected = max(valid, key=lambda row: row["combined_score"])
    final = decode_candidate(
        selected["pieces"],
        inventory,
        unit_model,
        trial.seed,
        f"final-{selected['model_type']}-{selected['vocab_size']}",
        final_iterations,
        final_restarts,
    )
    if final is None:
        raise RuntimeError("selected segmentation became invalid")
    predicted_boundaries = boundary_positions(selected["pieces"])
    return {
        "iso": trial.iso,
        "split": trial.split,
        "replicate": trial.replicate,
        "cipher_length": len(trial.cipher),
        "true_units": len(trial.units),
        "selected_model_type": selected["model_type"],
        "selected_vocab_size": selected["vocab_size"],
        "predicted_units": len(selected["pieces"]),
        "distinct_pieces": selected["distinct_pieces"],
        "boundary_f1": boundary_f1(trial.boundaries, predicted_boundaries),
        "plaintext_accuracy": plaintext_accuracy(trial.plain, final["plaintext"]),
        "plaintext_exact": final["plaintext"] == trial.plain,
        "combined_score": selected["combined_score"],
        "candidate_count": len(valid),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovery = [float(row["plaintext_accuracy"]) for row in rows]
    boundaries = [float(row["boundary_f1"]) for row in rows]
    return {
        "trials": len(rows),
        "plaintext": {
            "mean": statistics.fmean(recovery),
            "median": statistics.median(recovery),
            "minimum": min(recovery),
            "at_least_75_rate": statistics.fmean(value >= 0.75 for value in recovery),
            "exact_rate": statistics.fmean(row["plaintext_exact"] for row in rows),
        },
        "boundary_f1": {
            "mean": statistics.fmean(boundaries),
            "median": statistics.median(boundaries),
            "minimum": min(boundaries),
        },
        "gate": {
            "pass": (
                statistics.fmean(recovery) >= 0.75
                and statistics.median(recovery) >= 0.85
                and sum(value >= 0.75 for value in recovery) >= 13
                and statistics.fmean(boundaries) >= 0.85
                and min(recovery) >= 0.40
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--screen-iterations", type=int, default=700000)
    parser.add_argument("--screen-restarts", type=int, default=50)
    parser.add_argument("--final-iterations", type=int, default=700000)
    parser.add_argument("--final-restarts", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-s3"
    )
    language = languages[args.iso]
    inventory = s1.candidate_inventory(language)
    unit_model = s2.build_unit_model(language, inventory)
    trials = [
        s1.make_trial(language, args.split, args.length, replicate)
        for replicate in range(args.replicates)
    ]

    def run_one(trial: s1.SegmentationTrial) -> dict[str, Any]:
        row = solve_trial(
            trial,
            inventory,
            unit_model,
            args.screen_iterations,
            args.screen_restarts,
            args.final_iterations,
            args.final_restarts,
        )
        print("V060_S3_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, trials))
    summary = summarize(rows)
    payload = {
        "config": vars(args) | {"repo": str(args.repo), "output": str(args.output)},
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_S3_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_S3_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
