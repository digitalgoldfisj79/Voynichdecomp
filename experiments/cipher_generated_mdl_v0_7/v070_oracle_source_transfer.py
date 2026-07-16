#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import v070_entry as entry

p = entry.programme


def oracle_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for family in p.POSITIVE_FAMILIES:
        for corpus in p.CORPORA:
            for replicate in range(2):
                tasks.append({
                    "trial_type": "positive",
                    "family": family,
                    "corpus": corpus,
                    "replicate": replicate,
                    "seed": p.stable_seed("oracle", "positive", family, corpus, replicate),
                })
    for family in p.CONTROL_FAMILIES:
        for replicate in range(4):
            tasks.append({
                "trial_type": "control",
                "family": family,
                "replicate": replicate,
                "seed": p.stable_seed("oracle", "control", family, replicate),
            })
    return tasks


def generate_with_truth(repo: Path, module: Any, registry: Any, task: dict[str, Any]):
    seed = int(task["seed"])
    plan = module.document_plan(registry, seed, n_docs=12, tokens_per_doc=180)
    if task["trial_type"] == "positive":
        family = str(task["family"])
        corpus = str(task["corpus"])
        scheme, null_count, size_profile, mechanism = p.gen.family_spec(family)
        latent = p.gen.corpus_lines(module, p.gen.load_words(repo, corpus), plan, seed, null_count)
        events, keys, sizes = p.gen.render_lines(
            module, registry, plan, latent, seed, scheme, null_count,
            size_profile, mechanism, True,
        )
        return events, {
            "keys": keys,
            "scheme": scheme,
            "null_count": null_count,
            "size_profile": size_profile,
            "sizes": tuple(int(x) for x in sizes),
        }

    family = str(task["family"])
    latent = p.gen.ordered_control_lines(plan, family, seed)
    mechanisms = ("prf", "rotor", "feedback", "line_keyed")
    mechanism = mechanisms[int(task["replicate"]) % len(mechanisms)]
    scheme = "global" if int(task["replicate"]) % 2 == 0 else "currier"
    size_profile = "unequal" if scheme == "global" else "balanced"
    events, keys, sizes = p.gen.render_lines(
        module, registry, plan, latent, seed, scheme, 0,
        size_profile, mechanism, True,
    )
    return events, {
        "keys": keys,
        "scheme": scheme,
        "null_count": 0,
        "size_profile": size_profile,
        "sizes": tuple(int(x) for x in sizes),
    }


def choose_oracle_fit(module, registry, external, source_meta, train, truth):
    selector, selector_scores = module.select_selector(list(train), registry)
    candidates = []
    for profile in p.BASE_PROFILES:
        assignments = {str(label): tuple(int(x) for x in values) for label, values in truth["keys"].items()}
        policy, policy_scores = p.gen.base.infer_policy(module, list(train), assignments, truth["scheme"], registry)
        fitted = {
            "scheme": truth["scheme"],
            "null_count": truth["null_count"],
            "size_profile": truth["size_profile"],
            "external_profile": profile,
            "sizes": truth["sizes"],
            "assignments": assignments,
            "selector": selector,
            "selector_scores": selector_scores,
            "policy": policy,
            "policy_scores": policy_scores,
            "solver_meta": {"oracle_mapping": True},
        }
        policy_bits = float(p.gen.base.policy_nll(
            module, list(train), assignments, truth["scheme"], policy, registry,
        ))
        selector_bits = float(module.selector_nll(list(train), registry, selector))
        structural = p.model_record_bits(module, registry, fitted, train, full=False)
        for order in p.ORDERS:
            source = p.source_bits(module, train, fitted, external, source_meta, order)
            score = structural + math.log2(len(p.ORDERS)) + math.log2(len(p.POLICIES)) + source + policy_bits + selector_bits
            candidates.append({"fitted": fitted, "source_order": order, "selection_score": float(score)})
    candidates.sort(key=lambda row: (row["selection_score"], row["fitted"]["external_profile"], row["source_order"]))
    return candidates[0]


def run_task(repo_text: str, task: dict[str, Any]):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    repo = Path(repo_text)
    gr, module, registry, external, source_meta = p.get_assets(repo)
    events, truth = generate_with_truth(repo, module, registry, task)
    train = [event for event in events if not event.test]
    test = [event for event in events if event.test]
    if {int(e.doc) for e in train} & {int(e.doc) for e in test}:
        raise RuntimeError("document leakage")
    selected = choose_oracle_fit(module, registry, external, source_meta, train, truth)
    scored = p.score_fit(module, registry, external, source_meta, train, test, selected)
    fitted = selected["fitted"]
    expected = None
    if task["trial_type"] == "positive":
        expected = "loo_general" if task["corpus"] == "greek_general" else "loo_dmm"
    primary = expected is None or fitted["external_profile"] == expected
    decision = (
        scored["total_gain_bits_per_token"] >= 0.05
        and scored["heldout_gain_bits_per_token"] >= 0.02
        and scored["full_difference_bits"] < 0.0
        and scored["conditional_difference_bits"] < 0.0
        and primary
    )
    return {
        **task,
        "n_train": len(train),
        "n_test": len(test),
        "selected": {
            "source_profile": fitted["external_profile"],
            "source_order": selected["source_order"],
            "policy": fitted["policy"],
            "selector": fitted["selector"],
        },
        "primary_leave_target_out": bool(primary),
        "accounting": scored,
        "source_message_selected": bool(decision),
        "oracle_success": bool(decision) if task["trial_type"] == "positive" else None,
        "false_positive": bool(decision) if task["trial_type"] == "control" else None,
    }


def aggregate(results):
    positives = [row for row in results if row["trial_type"] == "positive"]
    controls = [row for row in results if row["trial_type"] == "control"]
    sensitivity = sum(row["oracle_success"] for row in positives) / len(positives)
    fpr = sum(row["false_positive"] for row in controls) / len(controls)

    def rates(rows, key, outcome):
        buckets = collections.defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        return {
            name: {
                "successes": sum(bool(item[outcome]) for item in items),
                "trials": len(items),
                "rate": sum(bool(item[outcome]) for item in items) / len(items),
            }
            for name, items in sorted(buckets.items())
        }

    family = rates(positives, "family", "oracle_success")
    corpus = rates(positives, "corpus", "oracle_success")
    control_family = rates(controls, "family", "false_positive")
    positive_gains = [row["accounting"]["heldout_gain_bits_per_token"] for row in positives]
    control_gains = [row["accounting"]["heldout_gain_bits_per_token"] for row in controls]
    gate = (
        sensitivity >= 0.75
        and all(row["successes"] >= 1 for row in family.values())
        and all(row["rate"] >= 0.625 for row in corpus.values())
        and fpr <= 0.15
        and all(row["successes"] <= 1 for row in control_family.values())
        and statistics.median(positive_gains) > 0.0
        and statistics.median(control_gains) <= 0.0
    )
    return {
        "positive": {
            "successes": sum(row["oracle_success"] for row in positives),
            "trials": len(positives),
            "sensitivity": sensitivity,
            "family": family,
            "corpus": corpus,
            "median_heldout_gain": statistics.median(positive_gains),
        },
        "control": {
            "false_positives": sum(row["false_positive"] for row in controls),
            "trials": len(controls),
            "false_positive_rate": fpr,
            "family": control_family,
            "median_heldout_gain": statistics.median(control_gains),
        },
        "gate_pass": bool(gate),
        "decision": "GO_TO_INFERRED_DEVELOPMENT" if gate else "STOP_V070_ORACLE_FAILED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=p.ROOT)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = oracle_tasks()
    started = time.time()
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(run_task, str(args.repo), task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print("V070_ORACLE_PROGRESS", json.dumps({
                "completed": completed,
                "total": len(tasks),
                "type": row["trial_type"],
                "family": row["family"],
                "selected": row["source_message_selected"],
                "gain": row["accounting"]["heldout_gain_bits_per_token"],
                "elapsed_seconds": time.time() - started,
            }, sort_keys=True), flush=True)
    results.sort(key=lambda row: (row["trial_type"], row["family"], row.get("corpus", ""), row["replicate"]))
    summary = aggregate(results)
    payload = {
        "programme": "cipher-generated-mdl-v0.7-stage-a0-oracle",
        "results": results,
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V070_ORACLE_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V070_ORACLE_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
