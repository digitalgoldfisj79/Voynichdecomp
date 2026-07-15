#!/usr/bin/env python3
"""Pair-block homophonic search preserving the inferred inventory exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_block_search_v052 as block
from homophonic_confirm_v052_quadgram import build_quadgram_model, quadgram_score_key


@njit(cache=True, nogil=True)
def rng_step(state: np.uint64) -> np.uint64:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)


@njit(cache=True, nogil=True)
def rng_int(state: np.uint64, upper: int) -> tuple[np.uint64, int]:
    state = rng_step(state)
    return state, int(state % np.uint64(upper))


@njit(cache=True, nogil=True)
def shuffled_key(state: np.uint64, initial_key: np.ndarray) -> tuple[np.uint64, np.ndarray]:
    key = initial_key.copy()
    for index in range(key.shape[0] - 1, 0, -1):
        state, other = rng_int(state, index + 1)
        temporary = key[index]
        key[index] = key[other]
        key[other] = temporary
    return state, key


@njit(cache=True, nogil=True)
def pair_block_polish_fixed(
    cipher: np.ndarray,
    key: np.ndarray,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    fixed_counts: np.ndarray,
    sweeps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    alphabet_size = fixed_counts.shape[0]
    key_length = key.shape[0]
    current = key.copy()
    current_score = quadgram_score_key(cipher, current, quadgram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)
    order = np.arange(alphabet_size, dtype=np.int32)
    indices = np.empty(key_length, dtype=np.int32)

    for _ in range(sweeps):
        for index in range(alphabet_size - 1, 0, -1):
            state, other = rng_int(state, index + 1)
            temporary = order[index]
            order[index] = order[other]
            order[other] = temporary

        improvements = 0
        for left in range(alphabet_size - 1):
            label_a = int(order[left])
            required_a = int(fixed_counts[label_a])
            for right in range(left + 1, alphabet_size):
                label_b = int(order[right])
                required_b = int(fixed_counts[label_b])
                block_size = 0
                original_mask = 0
                for key_index in range(key_length):
                    value = int(current[key_index])
                    if value == label_a or value == label_b:
                        indices[block_size] = key_index
                        if value == label_a:
                            original_mask |= 1 << block_size
                        block_size += 1
                if block_size == 0 or block_size != required_a + required_b or block_size > 20:
                    continue

                best_mask = original_mask
                best_score = current_score
                mask_limit = 1 << block_size
                for mask in range(mask_limit):
                    count_a = 0
                    for position in range(block_size):
                        if (mask >> position) & 1:
                            count_a += 1
                    if count_a != required_a:
                        continue
                    for position in range(block_size):
                        current[indices[position]] = (
                            label_a if ((mask >> position) & 1) else label_b
                        )
                    candidate_score = quadgram_score_key(
                        cipher, current, quadgram_logp, unigram_logp
                    )
                    if candidate_score > best_score + 1e-9:
                        best_score = candidate_score
                        best_mask = mask

                for position in range(block_size):
                    current[indices[position]] = (
                        label_a if ((best_mask >> position) & 1) else label_b
                    )
                if best_score > current_score + 1e-9:
                    current_score = best_score
                    improvements += 1

        if improvements == 0:
            break

    return current, current_score


@njit(cache=True, nogil=True)
def fixed_inventory_block_search(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    slot_pool: np.ndarray,
    max_counts: np.ndarray,
    restarts: int,
    sweeps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    del slot_pool, max_counts
    alphabet_size = quadgram_logp.shape[0]
    fixed_counts = np.zeros(alphabet_size, dtype=np.int32)
    for value in initial_key:
        fixed_counts[int(value)] += 1

    best_key = initial_key.copy()
    best_score = quadgram_score_key(cipher, best_key, quadgram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)

    for restart in range(restarts):
        if restart == 0:
            candidate = initial_key.copy()
        elif restart % 3 == 1:
            candidate = best_key.copy()
            perturbations = 2 + restart % 9
            for _ in range(perturbations):
                state, first = rng_int(state, candidate.shape[0])
                state, second = rng_int(state, candidate.shape[0])
                if first != second:
                    temporary = candidate[first]
                    candidate[first] = candidate[second]
                    candidate[second] = temporary
        else:
            state, candidate = shuffled_key(state, initial_key)

        candidate, candidate_score = pair_block_polish_fixed(
            cipher,
            candidate,
            quadgram_logp,
            unigram_logp,
            fixed_counts,
            sweeps,
            int(state & np.uint64(0x7FFFFFFFFFFFFFFF)),
        )
        if candidate_score > best_score:
            best_score = candidate_score
            best_key = candidate.copy()

    return best_key, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", required=True, choices=("en", "he"))
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    block.block_search = fixed_inventory_block_search
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    model = build_quadgram_model(language)
    slot_pool, max_counts = block.family_arrays(language)
    fixed_inventory_block_search(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        model[0], model[1], slot_pool, max_counts, 1, 1, 1,
    )

    schedules = ((3, 3), (6, 5), (12, 8), (24, 12))
    candidates = []
    selected = None
    selected_score = -1.0
    for restarts, sweeps in schedules:
        rows = block.run_grid(
            language,
            model,
            args.split,
            args.length,
            args.replicates,
            restarts,
            sweeps,
            args.workers,
            args.offset,
        )
        summary = block.summarize(rows)
        candidate = {"restarts": restarts, "sweeps": sweeps, "summary": summary}
        candidates.append(candidate)
        print("V052_FIXED_BLOCK_CANDIDATE", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = (restarts, sweeps)

    payload = {
        "programme": "v0.5.2-fixed-inventory-pair-block-search",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "replicates": args.replicates,
        "offset": args.offset,
        "candidates": candidates,
        "selected_schedule": {"restarts": selected[0], "sweeps": selected[1]},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_FIXED_BLOCK_SELECTED", json.dumps(payload["selected_schedule"], sort_keys=True), flush=True)
    print("V052_FIXED_BLOCK_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
