#!/usr/bin/env python3
"""Oracle and production-null ablation for morpholocal calibration v0.3.2.

This diagnostic bypasses mapping inference entirely.  For each frozen synthetic
positive fixture it scores the *true* latent mapping, structure, policy and
selector against:

* each fixed production null in the v0.3.1 registry;
* the training-selected registry member;
* a Bayesian sequence-level mixture over the registry; and
* the original v0.2 production comparator.

The result distinguishes three failure locations:

1. oracle loses held-out prediction -> comparator/generator incompatibility;
2. oracle wins held-out prediction but loses H/I conjunction -> coding gate;
3. oracle passes -> remaining failure is inference.

Development calibration only; not a formal locked test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import production_null_registry as production
import remediation_runtime as remediation

base = remediation.base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logsumexp2(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log2(sum(2.0 ** (value - maximum) for value in values))


def report_dict(report: Any) -> dict[str, float]:
    fields = (
        "canonical_serialization_bits",
        "partition_bits",
        "topology_bits",
        "transition_kt_bits",
        "emission_kt_bits",
        "latent_path_bits",
        "external_model_index_bits",
        "structural_universal_bits",
    )
    return {
        field: float(getattr(report, field))
        for field in fields
        if hasattr(report, field)
    }


def fit_fixed_nulls(train: Sequence[Any], test: Sequence[Any]) -> dict[str, dict[str, Any]]:
    fitted: dict[str, dict[str, Any]] = {}
    for name, state_fn in (
        ("context_iid", production._context_state),
        ("cell_markov", production._markov_state),
        ("context_cell_markov", production._context_markov_state),
    ):
        train_bits, payload = production._fit_rows(train, state_fn)
        fitted[name] = {
            "train_bits": float(train_bits),
            "test_bits": float(production._score_rows(test, payload, state_fn)),
        }
    train_bits, payload = production._fit_repeat_context(train)
    fitted["repeat_context"] = {
        "train_bits": float(train_bits),
        "test_bits": float(production._score_repeat_context(test, payload)),
    }
    return fitted


def mixture_score(fixed: dict[str, dict[str, Any]]) -> dict[str, float]:
    prior_bits = math.log2(len(fixed))
    train_logs = [-(prior_bits + float(row["train_bits"])) for row in fixed.values()]
    joint_logs = [
        -(prior_bits + float(row["train_bits"]) + float(row["test_bits"]))
        for row in fixed.values()
    ]
    log_train = logsumexp2(train_logs)
    log_joint = logsumexp2(joint_logs)
    return {
        "train_bits": float(-log_train),
        "test_bits": float(-(log_joint - log_train)),
    }


def comparison_ledger(
    *,
    cipher_train: float,
    cipher_test: float,
    production_train: float,
    production_test: float,
    n_test: int,
    cipher_full: Any,
    cipher_conditional: Any,
    production_full: Any,
    production_conditional: Any,
    assignment_labels: int,
) -> dict[str, Any]:
    policy_index_bits = math.log2(len(base.POLICIES))
    h_full_cipher = (
        float(cipher_full.canonical_serialization_bits)
        + policy_index_bits
        + cipher_train
        + cipher_test
    )
    h_full_production = (
        float(production_full.canonical_serialization_bits)
        + production_train
        + production_test
    )
    h_cond_cipher = (
        float(cipher_conditional.canonical_serialization_bits)
        + policy_index_bits
        + cipher_train
        + cipher_test
    )
    h_cond_production = (
        float(production_conditional.canonical_serialization_bits)
        + production_train
        + production_test
    )
    extra_partition = float(cipher_conditional.partition_bits) * max(0, assignment_labels - 1)
    i_cipher = (
        float(cipher_conditional.structural_universal_bits)
        + extra_partition
        + policy_index_bits
        + cipher_train
        + cipher_test
    )
    i_production = (
        float(production_conditional.structural_universal_bits)
        + production_train
        + production_test
    )
    differences = {
        "H_full_cipher_minus_production": h_full_cipher - h_full_production,
        "H_conditional_cipher_minus_production": h_cond_cipher - h_cond_production,
        "I_full_cipher_minus_production": i_cipher - i_production,
        "I_conditional_cipher_minus_production": i_cipher - i_production,
        "heldout_predictive_cipher_minus_production": cipher_test - production_test,
    }
    h_i_signs = all(
        differences[key] < 0.0
        for key in (
            "H_full_cipher_minus_production",
            "H_conditional_cipher_minus_production",
            "I_full_cipher_minus_production",
            "I_conditional_cipher_minus_production",
        )
    )
    gain = (production_test - cipher_test) / max(1, n_test)
    legacy_selected = bool(h_i_signs and gain >= -0.025)
    heldout_advantage = bool(differences["heldout_predictive_cipher_minus_production"] < 0.0)
    return {
        "cipher_train_bits": float(cipher_train),
        "cipher_test_bits": float(cipher_test),
        "production_train_bits": float(production_train),
        "production_test_bits": float(production_test),
        "heldout_delta_bits_per_token": float(
            differences["heldout_predictive_cipher_minus_production"] / max(1, n_test)
        ),
        "predictive_gain_bits_per_test_token": float(gain),
        "differences_bits": {key: float(value) for key, value in differences.items()},
        "h_i_signs": bool(h_i_signs),
        "legacy_selected": legacy_selected,
        "heldout_advantage": heldout_advantage,
        "strict_selected": bool(legacy_selected and heldout_advantage),
    }


def run_trial(
    *,
    module: Any,
    registry: Any,
    external: Any,
    index: int,
    seed: int,
    length_profile: str,
    original_production_predictive_nll: Any,
    train_identity_guard: list[Any],
) -> dict[str, Any]:
    scenario = module.scenario_for_index(index)
    events, keys, sizes = module.generate_cipher_trial(
        registry,
        external,
        seed,
        str(scenario["key_scheme"]),
        int(scenario["null_count"]),
        str(scenario["size_profile"]),
        str(scenario["external_profile"]),
        str(scenario["selection_policy"]),
        str(scenario["selector"]),
    )
    events = remediation.safe_subset_events(events, int(base.LENGTHS[length_profile]))
    train = [event for event in events if not event.test]
    test = [event for event in events if event.test]
    # Retain every training-list object so any legacy id-keyed cache cannot see a
    # recycled Python id during this sequential audit.
    train_identity_guard.append(train)

    transition, stationary = module.extend_external_model(
        external, str(scenario["external_profile"]), int(scenario["null_count"])
    )
    external_train, external_test = base.external_nll_by_split(
        module,
        events,
        keys,
        str(scenario["key_scheme"]),
        transition,
        stationary,
    )
    policy_train, policy_test = base.policy_nll(
        module,
        events,
        keys,
        str(scenario["key_scheme"]),
        str(scenario["selection_policy"]),
        registry,
        return_by_split=True,
    )
    selector_train = module.selector_nll(train, registry, str(scenario["selector"]))
    selector_test = module.selector_nll(test, registry, str(scenario["selector"]))
    cipher_train = float(external_train + policy_train + selector_train)
    cipher_test = float(external_test + policy_test + selector_test)

    production_selector, production_selector_scores = module.select_selector(train, registry)
    cipher_full_record = module.cipher_model_record(
        registry,
        keys,
        str(scenario["key_scheme"]),
        sizes,
        int(scenario["null_count"]),
        str(scenario["size_profile"]),
        str(scenario["external_profile"]),
        str(scenario["selector"]),
        train,
        full=True,
    )
    cipher_cond_record = module.cipher_model_record(
        registry,
        keys,
        str(scenario["key_scheme"]),
        sizes,
        int(scenario["null_count"]),
        str(scenario["size_profile"]),
        str(scenario["external_profile"]),
        str(scenario["selector"]),
        train,
        full=False,
    )
    production_full_record = module.production_model_record(
        registry, production_selector, train, full=True
    )
    production_cond_record = module.production_model_record(
        registry, production_selector, train, full=False
    )
    cipher_full = module.cost_model(cipher_full_record)
    cipher_conditional = module.cost_model(cipher_cond_record)
    production_full = module.cost_model(production_full_record)
    production_conditional = module.cost_model(production_cond_record)

    fixed = fit_fixed_nulls(train, test)
    index_bits = math.log2(len(fixed))
    selected_name = min(
        fixed,
        key=lambda name: (float(fixed[name]["train_bits"]) + index_bits, name),
    )
    selected_registry = {
        "train_bits": float(fixed[selected_name]["train_bits"] + index_bits),
        "test_bits": float(fixed[selected_name]["test_bits"]),
        "selected_model": selected_name,
        "registry_index_bits": float(index_bits),
    }
    mixture = mixture_score(fixed)

    original_train = float(
        original_production_predictive_nll(train, train, registry, production_selector)
    )
    original_test = float(
        original_production_predictive_nll(test, train, registry, production_selector)
    )

    comparator_scores: dict[str, dict[str, Any]] = {
        name: {"train_bits": float(row["train_bits"]), "test_bits": float(row["test_bits"])}
        for name, row in fixed.items()
    }
    comparator_scores["selected_registry"] = selected_registry
    comparator_scores["bayesian_registry_mixture"] = mixture
    comparator_scores["v02_original"] = {
        "train_bits": original_train,
        "test_bits": original_test,
    }

    comparisons = {
        name: comparison_ledger(
            cipher_train=cipher_train,
            cipher_test=cipher_test,
            production_train=float(score["train_bits"]),
            production_test=float(score["test_bits"]),
            n_test=len(test),
            cipher_full=cipher_full,
            cipher_conditional=cipher_conditional,
            production_full=production_full,
            production_conditional=production_conditional,
            assignment_labels=len(keys),
        )
        for name, score in comparator_scores.items()
    }
    current = comparisons["selected_registry"]
    if not current["heldout_advantage"]:
        failure_location = "comparator_or_positive_generator"
    elif not current["legacy_selected"]:
        failure_location = "coding_or_conjunction_gate"
    else:
        failure_location = "oracle_passes_inference_remains"

    return {
        "trial_index": int(index),
        "seed": int(seed),
        "length_profile": length_profile,
        "truth": {**scenario, "class_sizes": list(sizes), "length_profile": length_profile},
        "n_train": len(train),
        "n_test": len(test),
        "oracle_cipher": {
            "external_train_bits": float(external_train),
            "external_test_bits": float(external_test),
            "policy_train_bits": float(policy_train),
            "policy_test_bits": float(policy_test),
            "selector_train_bits": float(selector_train),
            "selector_test_bits": float(selector_test),
            "total_train_bits": cipher_train,
            "total_test_bits": cipher_test,
            "mapping_accuracy": 1.0,
            "null_f1": 1.0,
            "policy_correct": True,
            "selector_correct": True,
            "structure_correct": True,
            "latent_unit_error": 0.0,
        },
        "production_selector": production_selector,
        "production_selector_scores": {
            key: float(value) for key, value in production_selector_scores.items()
        },
        "model_cost_ledger": {
            "cipher_full": report_dict(cipher_full),
            "cipher_conditional": report_dict(cipher_conditional),
            "production_full": report_dict(production_full),
            "production_conditional": report_dict(production_conditional),
            "policy_index_bits": float(math.log2(len(base.POLICIES))),
            "assignment_label_count": len(keys),
        },
        "production_scores": comparator_scores,
        "comparisons": comparisons,
        "failure_location": failure_location,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    comparator_names = sorted(results[0]["comparisons"]) if results else []
    comparators: dict[str, Any] = {}
    for name in comparator_names:
        rows = [row["comparisons"][name] for row in results]
        heldout = sum(bool(row["heldout_advantage"]) for row in rows)
        legacy = sum(bool(row["legacy_selected"]) for row in rows)
        strict = sum(bool(row["strict_selected"]) for row in rows)
        comparators[name] = {
            "trials": len(rows),
            "oracle_heldout_wins": heldout,
            "oracle_heldout_win_rate": heldout / max(1, len(rows)),
            "oracle_legacy_selected": legacy,
            "oracle_strict_selected": strict,
            "median_heldout_delta_bits_per_token": statistics.median(
                float(row["heldout_delta_bits_per_token"]) for row in rows
            ),
            "median_predictive_gain_bits_per_test_token": statistics.median(
                float(row["predictive_gain_bits_per_test_token"]) for row in rows
            ),
            "heldout_wilson90": base.wilson(heldout, len(rows)),
            "strict_wilson90": base.wilson(strict, len(rows)),
        }

    locations: defaultdict[str, int] = defaultdict(int)
    for row in results:
        locations[str(row["failure_location"])] += 1

    strata: dict[str, Any] = {}
    for dimension in (
        "selection_policy",
        "key_scheme",
        "null_count",
        "size_profile",
        "external_profile",
        "selector",
        "length_profile",
    ):
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            buckets[str(row["truth"][dimension])].append(row)
        strata[dimension] = {
            value: {
                "trials": len(rows),
                "selected_registry_heldout_wins": sum(
                    bool(item["comparisons"]["selected_registry"]["heldout_advantage"])
                    for item in rows
                ),
                "selected_registry_legacy_selected": sum(
                    bool(item["comparisons"]["selected_registry"]["legacy_selected"])
                    for item in rows
                ),
                "median_selected_registry_delta_bits_per_token": statistics.median(
                    float(
                        item["comparisons"]["selected_registry"][
                            "heldout_delta_bits_per_token"
                        ]
                    )
                    for item in rows
                ),
            }
            for value, rows in sorted(buckets.items())
        }

    return {
        "trials": len(results),
        "comparators": comparators,
        "failure_locations": dict(sorted(locations.items())),
        "selected_registry_strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--positive-start", type=int, required=True)
    parser.add_argument("--positive-end", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3030303)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.positive_start < args.positive_end):
        raise SystemExit("require 0 <= positive-start < positive-end")

    # Load and retain the original v0.2 comparator before installing v0.3.1's
    # production-registry wrapper.  The generated effective source is patched
    # once here, sequentially, so there is no worker race.
    gr, module = base.load_v02(args.repo)
    original_production_predictive_nll = module.production_predictive_nll
    remediation.install()

    record_path, ci_path = module.locate_data(args.repo)
    registry = module.build_surface_registry(pickle.load(record_path.open("rb")))
    external = module.build_external_models(pickle.load(ci_path.open("rb")))
    lengths = tuple(base.LENGTHS)
    train_identity_guard: list[Any] = []

    started = time.time()
    results: list[dict[str, Any]] = []
    for completed, index in enumerate(range(args.positive_start, args.positive_end), 1):
        seed = args.seed + 100000 + index * 7919
        length_profile = lengths[index % len(lengths)]
        result = run_trial(
            module=module,
            registry=registry,
            external=external,
            index=index,
            seed=seed,
            length_profile=length_profile,
            original_production_predictive_nll=original_production_predictive_nll,
            train_identity_guard=train_identity_guard,
        )
        results.append(result)
        current = result["comparisons"]["selected_registry"]
        print(
            f"V032_ORACLE_PROGRESS range={args.positive_start}:{args.positive_end} "
            f"completed={completed}/{args.positive_end-args.positive_start} "
            f"index={index} length={length_profile} "
            f"heldout={int(current['heldout_advantage'])} "
            f"legacy={int(current['legacy_selected'])} "
            f"location={result['failure_location']} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

    observed = [int(row["trial_index"]) for row in results]
    expected = list(range(args.positive_start, args.positive_end))
    if observed != expected:
        raise RuntimeError(f"oracle index mismatch: expected={expected} observed={observed}")

    root = args.repo
    source_paths = [
        root / "experiments/morpholocal_calibration_v0_3/remediation_oracle_null_ablation.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime.py",
        root / "experiments/morpholocal_calibration_v0_3/tournament_runner.py",
        root / "experiments/morpholocal_calibration_v0_3/production_null_registry.py",
        root / "experiments/morpholocal_calibration_v0_2/morpholocal_gate_impl.py",
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "programme": "morpholocal-calibration-v0.3.2-oracle-null-ablation",
        "formal": False,
        "primary_question": "Does the true cipher beat each production comparator?",
        "seed": args.seed,
        "positive_start": args.positive_start,
        "positive_end": args.positive_end,
        "git_commit": commit,
        "scientific_source_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_paths
        },
        "summary": aggregate(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print("V032_ORACLE_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
