#!/usr/bin/env python3
"""Nested inventory beam with fixed-inventory inner optimisation for v0.5.2."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
import homophonic_fixed_inventory_block_v052 as fixed_block
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import build_quadgram_model, quadgram_score_key


@dataclass
class BeamState:
    key: np.ndarray
    counts: np.ndarray
    score: float
    depth: int
    path: tuple[tuple[int, int], ...]


def inventory_counts(key: np.ndarray, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int32)
    for value in key:
        counts[int(value)] += 1
    return counts


def inventory_signature(counts: np.ndarray) -> bytes:
    return counts.astype(np.int16, copy=False).tobytes()


def inventory_transfer_distance(first: np.ndarray, second_labels: list[int]) -> int:
    second = Counter(map(int, second_labels))
    l1 = 0
    for label in range(len(first)):
        l1 += abs(int(first[label]) - int(second.get(label, 0)))
    return l1 // 2


def polish_fixed_inventory(
    cipher: np.ndarray,
    key: np.ndarray,
    counts: np.ndarray,
    model: tuple[np.ndarray, np.ndarray],
    sweeps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    polished, score = fixed_block.pair_block_polish_fixed(
        cipher,
        key,
        model[0],
        model[1],
        counts,
        sweeps,
        int(seed & 0x7FFFFFFFFFFFFFFF),
    )
    return polished, float(score)


def quick_inventory_moves(
    state: BeamState,
    cipher: np.ndarray,
    max_counts: np.ndarray,
    model: tuple[np.ndarray, np.ndarray],
    proposal_limit: int,
) -> list[tuple[float, int, int, int, np.ndarray, np.ndarray]]:
    """Return the strongest immediate seed for each donor/recipient count transfer."""
    proposals: list[tuple[float, int, int, int, np.ndarray, np.ndarray]] = []
    alphabet_size = len(state.counts)
    for donor in range(alphabet_size):
        if int(state.counts[donor]) <= 0:
            continue
        donor_indices = np.flatnonzero(state.key == donor)
        if donor_indices.size == 0:
            continue
        for recipient in range(alphabet_size):
            if recipient == donor or int(state.counts[recipient]) >= int(max_counts[recipient]):
                continue
            best_score = -1e300
            best_index = -1
            best_key: np.ndarray | None = None
            for index_value in donor_indices:
                index = int(index_value)
                candidate_key = state.key.copy()
                candidate_key[index] = recipient
                score = float(quadgram_score_key(cipher, candidate_key, model[0], model[1]))
                if score > best_score:
                    best_score = score
                    best_index = index
                    best_key = candidate_key
            if best_key is None:
                continue
            candidate_counts = state.counts.copy()
            candidate_counts[donor] -= 1
            candidate_counts[recipient] += 1
            proposals.append(
                (best_score, donor, recipient, best_index, best_key, candidate_counts)
            )
    proposals.sort(key=lambda item: item[0], reverse=True)
    return proposals[:proposal_limit]


def nested_beam_search(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    max_counts: np.ndarray,
    model: tuple[np.ndarray, np.ndarray],
    depth_limit: int,
    beam_width: int,
    proposal_limit: int,
    inner_sweeps: int,
    seed: int,
) -> tuple[BeamState, list[dict[str, Any]]]:
    alphabet_size = len(max_counts)
    initial_counts = inventory_counts(initial_key, alphabet_size)
    polished_key, polished_score = polish_fixed_inventory(
        cipher,
        initial_key,
        initial_counts,
        model,
        inner_sweeps,
        seed,
    )
    initial_state = BeamState(
        key=polished_key,
        counts=initial_counts,
        score=polished_score,
        depth=0,
        path=(),
    )
    beam = [initial_state]
    best = initial_state
    trace: list[dict[str, Any]] = [
        {
            "depth": 0,
            "beam_states": 1,
            "best_score": best.score,
            "best_path": [],
        }
    ]

    for depth in range(1, depth_limit + 1):
        deduplicated: dict[bytes, BeamState] = {}
        for state_index, state in enumerate(beam):
            proposals = quick_inventory_moves(
                state,
                cipher,
                max_counts,
                model,
                proposal_limit,
            )
            for proposal_index, (
                _quick_score,
                donor,
                recipient,
                _changed_index,
                candidate_key,
                candidate_counts,
            ) in enumerate(proposals):
                candidate_seed = (
                    seed
                    + depth * 1_000_003
                    + state_index * 10_007
                    + proposal_index * 101
                    + donor * 17
                    + recipient
                )
                candidate_key, candidate_score = polish_fixed_inventory(
                    cipher,
                    candidate_key,
                    candidate_counts,
                    model,
                    inner_sweeps,
                    candidate_seed,
                )
                candidate = BeamState(
                    key=candidate_key,
                    counts=candidate_counts,
                    score=candidate_score,
                    depth=depth,
                    path=state.path + ((donor, recipient),),
                )
                signature = inventory_signature(candidate_counts)
                incumbent = deduplicated.get(signature)
                if incumbent is None or candidate.score > incumbent.score:
                    deduplicated[signature] = candidate
                if candidate.score > best.score:
                    best = candidate

        if not deduplicated:
            break
        beam = sorted(
            deduplicated.values(),
            key=lambda state: state.score,
            reverse=True,
        )[:beam_width]
        trace.append(
            {
                "depth": depth,
                "beam_states": len(beam),
                "candidate_inventories": len(deduplicated),
                "beam_best_score": beam[0].score,
                "global_best_score": best.score,
                "global_best_path": [list(move) for move in best.path],
            }
        )

    return best, trace


def solve_trial(
    trial: dict[str, Any],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    schedule: tuple[int, int, int, int],
) -> dict[str, Any]:
    depth_limit, beam_width, proposal_limit, inner_sweeps = schedule
    cipher = np.asarray(trial["cipher"], dtype=np.int32)
    initial_key = homophonic.frequency_slot_key(
        trial["cipher"],
        trial["inferred_labels"],
        trial["expected_slot_probabilities"],
    )
    max_counts = np.asarray(
        [homophonic.multiplicity(float(probability)) for probability in language.probabilities],
        dtype=np.int32,
    )
    initial_counts = inventory_counts(initial_key, len(language.alphabet))
    best, trace = nested_beam_search(
        cipher,
        initial_key,
        max_counts,
        model,
        depth_limit,
        beam_width,
        proposal_limit,
        inner_sweeps,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = best.key[cipher].tolist()
    baseline = initial_key[cipher].tolist()
    true_distance_initial = inventory_transfer_distance(initial_counts, trial["true_labels"])
    true_distance_final = inventory_transfer_distance(best.counts, trial["true_labels"])
    return {
        "replicate": int(trial["replicate"]),
        "seed": int(trial["seed"]),
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
        "baseline_accuracy": mono.fast_accuracy(trial["plain"], baseline),
        "exact": prediction == trial["plain"],
        "initial_inventory_overlap": float(trial["inventory_overlap"]),
        "final_inventory_overlap": homophonic.multiset_overlap(best.key, trial["true_labels"]),
        "initial_inventory_transfer_distance": true_distance_initial,
        "final_inventory_transfer_distance": true_distance_final,
        "selected_depth": best.depth,
        "selected_path": [list(move) for move in best.path],
        "score": best.score,
        "trace": trace,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
        "median_accuracy": statistics.median(row["accuracy"] for row in rows),
        "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in rows),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "mean_initial_inventory_overlap": statistics.fmean(
            row["initial_inventory_overlap"] for row in rows
        ),
        "mean_final_inventory_overlap": statistics.fmean(
            row["final_inventory_overlap"] for row in rows
        ),
        "mean_initial_inventory_transfer_distance": statistics.fmean(
            row["initial_inventory_transfer_distance"] for row in rows
        ),
        "mean_final_inventory_transfer_distance": statistics.fmean(
            row["final_inventory_transfer_distance"] for row in rows
        ),
        "mean_selected_depth": statistics.fmean(row["selected_depth"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages["en"]
    model = build_quadgram_model(language)

    # Compile the exact inner optimiser before parallel execution.
    dummy_counts = np.zeros(len(language.alphabet), dtype=np.int32)
    dummy_counts[0] = 2
    dummy_counts[1] = 2
    fixed_block.pair_block_polish_fixed(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        model[0],
        model[1],
        dummy_counts,
        1,
        1,
    )

    trials = [
        homophonic.make_trial(language, "dev", 384, replicate)
        for replicate in range(args.replicates)
    ]
    schedules = (
        (3, 4, 8, 4),
        (4, 6, 12, 6),
        (4, 10, 20, 8),
    )
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_score = -1.0

    for schedule in schedules:
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(solve_trial, trial, language, model, schedule)
                for trial in trials
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                rows.append(future.result())
                if completed % 2 == 0 or completed == len(futures):
                    print(
                        f"V052_NESTED_PROGRESS {completed}/{len(futures)} schedule={schedule}",
                        flush=True,
                    )
        rows.sort(key=lambda row: row["replicate"])
        summary = summarize(rows)
        candidate = {
            "depth_limit": schedule[0],
            "beam_width": schedule[1],
            "proposal_limit": schedule[2],
            "inner_sweeps": schedule[3],
            "summary": summary,
            "rows": rows,
        }
        candidates.append(candidate)
        print("V052_NESTED_CANDIDATE", json.dumps({k: v for k, v in candidate.items() if k != "rows"}, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = candidate

    assert selected is not None
    gate = {"english_70_percent_pass": selected["summary"]["mean_accuracy"] >= 0.70}
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "v0.5.2-nested-inventory-beam",
        "iso": "en",
        "split": "dev",
        "length": 384,
        "replicates": args.replicates,
        "candidates": candidates,
        "selected": selected,
        "gate": gate,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_NESTED_SELECTED", json.dumps({k: v for k, v in selected.items() if k != "rows"}, sort_keys=True), flush=True)
    print("V052_NESTED_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052_NESTED_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
