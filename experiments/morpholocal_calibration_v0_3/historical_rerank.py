#!/usr/bin/env python3
"""Higher-order reranking of historical-lattice mapping candidates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v03_historical_lattice", HERE / "historical_lattice.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import historical_lattice.py")
hl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hl
spec.loader.exec_module(hl)
_SHARED = {}


def worker(seed: int):
    mapping, objective = hl.anneal(
        _SHARED["pair"], _SHARED["uni"], _SHARED["counts"],
        _SHARED["transition"], _SHARED["stationary"],
        seed, _SHARED["steps"], 1,
    )
    sequence = [mapping[event.cell] for event in _SHARED["train_events"]]
    scores = {str(order): _SHARED["models"][order].score(sequence) for order in (3, 5, 6)}
    return {"seed": seed, "mapping": mapping, "bigram_objective": objective, "ngram_bits": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=hl.base.DEFAULT_REPO)
    parser.add_argument("--seed", type=int, default=3030317)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    words = hl.load_words(args.repo)
    parts = hl.partitions(words)
    inventory = hl.build_inventory(parts["lm_train"])
    lm_sequence = hl.encode_words(parts["lm_train"], inventory)
    transition, stationary = hl.transition_model(lm_sequence, len(inventory.units))
    models = {order: hl.build_ngram(lm_sequence, order, len(inventory.units)) for order in (3, 5, 6)}
    plaintext = parts["development"][:1800]
    train_units = hl.encode_words(plaintext[:1400], inventory)
    test_units = hl.encode_words(plaintext[1400:], inventory)
    counts = hl.allocate_homophones(
        train_units, len(inventory.units), max(80, len(inventory.units) + 24), "unequal"
    )
    key = hl.make_key(args.seed, counts)
    train_events = hl.encipher(train_units, key, args.seed ^ 0x1111, "frequency_weighted")
    test_events = hl.encipher(test_units, key, args.seed ^ 0x2222, "frequency_weighted", True)
    pair, uni = hl.cell_statistics(train_events, len(key))
    _SHARED.update({
        "pair": pair, "uni": uni, "counts": counts,
        "transition": transition, "stationary": stationary,
        "steps": args.steps, "train_events": train_events, "models": models,
    })
    seeds = [args.seed ^ 0x3333 ^ (i * 0x9E3779B1) for i in range(args.candidates)]
    started = time.time()
    rows = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(worker, seeds), 1):
            rows.append(row)
            print(
                f"LATTICE_CANDIDATE completed={index}/{len(seeds)} "
                f"elapsed={time.time()-started:.1f}s", flush=True,
            )
    selections = {
        "bigram": max(rows, key=lambda row: (row["bigram_objective"], tuple(row["mapping"]))),
        "3gram": min(rows, key=lambda row: (row["ngram_bits"]["3"], tuple(row["mapping"]))),
        "5gram": min(rows, key=lambda row: (row["ngram_bits"]["5"], tuple(row["mapping"]))),
        "6gram": min(rows, key=lambda row: (row["ngram_bits"]["6"], tuple(row["mapping"]))),
    }
    truth_text = hl.reconstruct(test_units, inventory)
    evaluation = {}
    for name, row in selections.items():
        predicted_units = [row["mapping"][event.cell] for event in test_events]
        predicted_text = hl.reconstruct(predicted_units, inventory)
        evaluation[name] = {
            "seed": row["seed"],
            "mapping_accuracy": sum(a == b for a, b in zip(row["mapping"], key)) / len(key),
            "latent_unit_error": sum(a != b for a, b in zip(predicted_units, test_units)) / len(test_units),
            "character_ter": hl.levenshtein(predicted_text, truth_text) / len(truth_text),
            "bigram_objective": row["bigram_objective"],
            "ngram_bits": row["ngram_bits"],
        }
    accuracies = [
        sum(a == b for a, b in zip(row["mapping"], key)) / len(key) for row in rows
    ]
    report = {
        "programme": "morpholocal-calibration-v0.3-historical-lattice-rerank-development",
        "formal": False, "seed": args.seed,
        "candidates": args.candidates, "steps_per_candidate": args.steps,
        "workers": args.workers, "inventory_hash": inventory.hash,
        "surface_symbols": len(key), "train_units": len(train_units),
        "test_units": len(test_units), "oracle_ter": 0.0,
        "evaluation": evaluation,
        "candidate_summary": {
            "mapping_accuracy_median": statistics.median(accuracies),
            "best_mapping_accuracy": max(accuracies),
            "unique_mappings": len({tuple(row["mapping"]) for row in rows}),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("LATTICE_RERANK_SUMMARY", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
