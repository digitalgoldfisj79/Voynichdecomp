#!/usr/bin/env python3
"""Morpholocal calibration v0.3 development decoder tournament.

Development-only runner. It reuses the frozen v0.2 generator, registry,
external corpus and production controls, while replacing the inference stack
with policy-aware heuristic, beam and Bayesian mapping solvers.

Formal execution requires a later complete effective-source freeze and does
not use this runtime patching loader.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
POLICIES = ("iid_uniform", "frequency_weighted", "cyclic", "sticky_line_reset")
LENGTHS = {"short": 2000, "medium": 8000, "long": 36000}


def load_v02(repo: Path):
    path = repo / "experiments/morpholocal_calibration_v0_2/gpu_runner.py"
    spec = importlib.util.spec_from_file_location(f"v03_v02_gpu_{os.getpid()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import v0.2 gpu_runner")
    gr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gr
    spec.loader.exec_module(gr)
    impl = gr.load_impl(repo)
    return gr, impl


def log2p(value: float, floor: float = 1e-300) -> float:
    return math.log2(max(floor, float(value)))


def mapping_unit(module, event, assignments: dict[str, Sequence[int]], scheme: str) -> int:
    return int(assignments[module.key_label(event, scheme)][event.cell])


def context_prob(module, registry, candidates: Sequence[int], event) -> dict[int, float]:
    weights = [float(module.context_weight(registry, c, event.section, event.position)) for c in candidates]
    total = sum(weights)
    if total <= 0:
        return {int(c): 1.0 / len(candidates) for c in candidates}
    return {int(c): w / total for c, w in zip(candidates, weights)}


def policy_nll(module, events, assignments, scheme, policy, registry, *, return_by_split=False):
    train_bits = 0.0
    test_bits = 0.0
    cyclic_state = defaultdict(int)
    eps = 1e-9
    for line in module.lines_from_events(events):
        previous_unit = None
        previous_cell = None
        for event in line:
            label = module.key_label(event, scheme)
            assignment = assignments[label]
            unit = int(assignment[event.cell])
            candidates = [c for c, u in enumerate(assignment) if int(u) == unit]
            if not candidates:
                bits = 1e6
            elif policy == "iid_uniform":
                bits = math.log2(len(candidates))
            elif policy == "frequency_weighted":
                probs = context_prob(module, registry, candidates, event)
                bits = -log2p(probs.get(int(event.cell), 0.0))
            elif policy == "cyclic":
                ordered = sorted(candidates)
                expected = ordered[cyclic_state[(label, unit)] % len(ordered)]
                prob = 1.0 - eps if int(event.cell) == expected else eps / max(1, len(ordered) - 1)
                bits = -log2p(prob)
                cyclic_state[(label, unit)] += 1
            elif policy == "sticky_line_reset":
                probs = context_prob(module, registry, candidates, event)
                base = probs.get(int(event.cell), 0.0)
                if previous_unit == unit and previous_cell in candidates:
                    prob = (0.75 if int(event.cell) == int(previous_cell) else 0.0) + 0.25 * base
                else:
                    prob = base
                bits = -log2p(prob)
            else:
                raise ValueError(policy)
            if event.test:
                test_bits += bits
            else:
                train_bits += bits
            previous_unit = unit
            previous_cell = int(event.cell)
    return (train_bits, test_bits) if return_by_split else train_bits + test_bits


def infer_policy(module, train, assignments, scheme, registry):
    scores = {p: float(policy_nll(module, train, assignments, scheme, p, registry)) for p in POLICIES}
    selected = min(scores, key=lambda p: (scores[p], p))
    return selected, scores


def external_nll_by_split(module, events, assignments, scheme, transition, stationary):
    train_bits = 0.0
    test_bits = 0.0
    log_t = np.log2(np.clip(transition, 1e-300, None))
    log_s = np.log2(np.clip(stationary, 1e-300, None))
    for line in module.lines_from_events(events):
        prev = None
        for event in line:
            unit = mapping_unit(module, event, assignments, scheme)
            bits = -float(log_s[unit]) if prev is None else -float(log_t[prev, unit])
            if event.test:
                test_bits += bits
            else:
                train_bits += bits
            prev = unit
    return train_bits, test_bits


def subset_events(events, target):
    if target >= len(events):
        return list(events)
    train = [e for e in events if not e.test]
    test = [e for e in events if e.test]
    n_test = min(len(test), max(1, int(round(target * 0.2))))
    n_train = min(len(train), target - n_test)

    def take_complete_lines(rows, n):
        groups, current, last = [], [], None
        for e in rows:
            key = (e.doc, e.line)
            if last is not None and key != last:
                groups.append(current)
                current = []
            current.append(e)
            last = key
        if current:
            groups.append(current)
        out = []
        for line in groups:
            if out and len(out) + len(line) > n:
                break
            out.extend(line)
            if len(out) >= n:
                break
        return out

    selected = take_complete_lines(train, n_train) + take_complete_lines(test, n_test)
    selected.sort(key=lambda e: (e.doc, e.line, 0 if e.position == "FIRST" else 2 if e.position == "LAST" else 1, e.token_index))
    return selected


def objective(module, pair, uni, mapping, transition, stationary):
    return float(module.objective_assignment(
        pair, uni, mapping,
        np.log2(np.clip(transition, 1e-300, None)),
        np.log2(np.clip(stationary, 1e-300, None)),
    ))


def mapping_counts(module, uni, sizes, null_count, stationary):
    base = list(module.initial_assignment(uni, sizes, null_count, stationary))
    return base, dict(Counter(int(x) for x in base))


def label_events(module, train, scheme, label):
    return [e for e in train if module.key_label(e, scheme) == label]


def label_policy_nll(module, events, mapping, policy, registry, label):
    assignments = {label: tuple(mapping)}
    scheme = "global" if label == "GLOBAL" else "currier"
    return float(policy_nll(module, events, assignments, scheme, policy, registry))


def refine_swaps(module, events, label, pair, uni, mapping, transition, stationary, policy, registry, seed, iterations, temperature=0.0):
    rng = random.Random(seed)
    current = list(mapping)
    ext = objective(module, pair, uni, current, transition, stationary)
    pol = label_policy_nll(module, events, current, policy, registry, label)
    score = ext - pol
    best, best_score = list(current), score
    n = len(current)
    for step in range(iterations):
        a = rng.randrange(n)
        b = rng.randrange(n - 1)
        if b >= a:
            b += 1
        if current[a] == current[b]:
            continue
        current[a], current[b] = current[b], current[a]
        cand_ext = objective(module, pair, uni, current, transition, stationary)
        cand_pol = label_policy_nll(module, events, current, policy, registry, label)
        cand = cand_ext - cand_pol
        delta = cand - score
        if temperature > 0:
            frac = step / max(1, iterations - 1)
            temp = temperature * (0.02 / temperature) ** frac
            accept = delta >= 0 or rng.random() < math.exp(min(0.0, delta / max(temp, 1e-12)))
        else:
            accept = delta >= 0
        if accept:
            ext, pol, score = cand_ext, cand_pol, cand
            if (score, tuple(current)) > (best_score, tuple(best)):
                best, best_score = list(current), score
        else:
            current[a], current[b] = current[b], current[a]
    return tuple(best), float(best_score)


def heuristic_solver(module, events, label, pair, uni, sizes, null_count, transition, stationary, registry, seed, cfg):
    mapping, _ = module.anneal_assignment(pair, uni, sizes, null_count, transition, stationary, seed, int(cfg["steps"]), int(cfg["restarts"]))
    current, policy = tuple(mapping), "iid_uniform"
    for round_index in range(int(cfg.get("alternations", 2))):
        scores = {p: label_policy_nll(module, events, current, p, registry, label) for p in POLICIES}
        policy = min(scores, key=lambda p: (scores[p], p))
        current, _ = refine_swaps(
            module, events, label, pair, uni, current, transition, stationary,
            policy, registry, seed ^ (0xABCDEF + round_index),
            int(cfg.get("refine_iterations", 1200)), float(cfg.get("refine_temperature", 1.0)),
        )
    return current, {"solver": "heuristic", "policy_hint": policy}


def beam_candidates(module, pair, uni, sizes, null_count, transition, stationary, beam_width):
    base, counts = mapping_counts(module, uni, sizes, null_count, stationary)
    n = len(base)
    log_t = np.log2(np.clip(transition, 1e-300, None))
    log_s = np.log2(np.clip(stationary, 1e-300, None))
    degree = uni + pair.sum(axis=0) + pair.sum(axis=1)
    order = [int(x) for x in np.argsort(-degree, kind="stable")]
    units = sorted(counts)
    beam = [(0.0, tuple([-1] * n), tuple(counts[u] for u in units))]
    for depth, cell in enumerate(order):
        expanded = []
        for partial_score, assignment, remaining in beam:
            arr = list(assignment)
            for ui, unit in enumerate(units):
                if remaining[ui] <= 0:
                    continue
                inc = 0.25 * float(uni[cell]) * float(log_s[unit])
                inc += float(pair[cell, cell]) * float(log_t[unit, unit])
                for other in order[:depth]:
                    other_unit = arr[other]
                    if other_unit < 0:
                        continue
                    inc += float(pair[other, cell]) * float(log_t[other_unit, unit])
                    inc += float(pair[cell, other]) * float(log_t[unit, other_unit])
                new_arr = list(arr)
                new_arr[cell] = unit
                new_remaining = list(remaining)
                new_remaining[ui] -= 1
                expanded.append((partial_score + inc, tuple(new_arr), tuple(new_remaining)))
        expanded.sort(key=lambda row: (row[0], row[1]), reverse=True)
        beam = expanded[:beam_width]
        if not beam:
            raise RuntimeError("beam exhausted")
    return [row[1] for row in beam]


def beam_solver(module, events, label, pair, uni, sizes, null_count, transition, stationary, registry, seed, cfg):
    candidates = beam_candidates(module, pair, uni, sizes, null_count, transition, stationary, int(cfg.get("beam_width", 256)))
    best = None
    best_meta = None
    for rank, mapping in enumerate(candidates[: int(cfg.get("policy_rerank", 64))]):
        policy_scores = {p: label_policy_nll(module, events, mapping, p, registry, label) for p in POLICIES}
        policy = min(policy_scores, key=lambda p: (policy_scores[p], p))
        score = objective(module, pair, uni, mapping, transition, stationary) - policy_scores[policy]
        key = (score, tuple(mapping))
        if best is None or key > best:
            best = key
            best_meta = (tuple(mapping), policy, rank)
    if best_meta is None:
        raise RuntimeError("beam produced no mapping")
    mapping, policy, rank = best_meta
    mapping, _ = refine_swaps(
        module, events, label, pair, uni, mapping, transition, stationary,
        policy, registry, seed ^ 0xBEEFBEEF, int(cfg.get("refine_iterations", 600)), 0.0,
    )
    return mapping, {"solver": "beam", "policy_hint": policy, "beam_rank": rank}


def bayes_solver(module, events, label, pair, uni, sizes, null_count, transition, stationary, registry, seed, cfg):
    rng = random.Random(seed)
    base, _ = mapping_counts(module, uni, sizes, null_count, stationary)
    temperatures = tuple(float(x) for x in cfg.get("temperatures", [1.0, 2.0, 4.0, 8.0]))
    chains = []
    for i, temp in enumerate(temperatures):
        current = list(base)
        random.Random(seed ^ ((i + 1) * 0x9E3779B1)).shuffle(current)
        policy_scores = {p: label_policy_nll(module, events, current, p, registry, label) for p in POLICIES}
        policy = min(policy_scores, key=lambda p: (policy_scores[p], p))
        score = objective(module, pair, uni, current, transition, stationary) - policy_scores[policy]
        chains.append({"mapping": current, "policy": policy, "score": score, "temp": temp})
    best = max((c["score"], tuple(c["mapping"]), c["policy"]) for c in chains)
    accepted = 0
    policy_updates = 0
    steps = int(cfg.get("mcmc_steps", 6000))
    for step in range(steps):
        for chain in chains:
            cur = chain["mapping"]
            a = rng.randrange(len(cur))
            b = rng.randrange(len(cur) - 1)
            if b >= a:
                b += 1
            if cur[a] == cur[b]:
                continue
            cur[a], cur[b] = cur[b], cur[a]
            policy = chain["policy"]
            cand = objective(module, pair, uni, cur, transition, stationary) - label_policy_nll(module, events, cur, policy, registry, label)
            delta = cand - chain["score"]
            if delta >= 0 or rng.random() < math.exp(min(0.0, delta / chain["temp"])):
                chain["score"] = cand
                accepted += 1
            else:
                cur[a], cur[b] = cur[b], cur[a]
            if step % int(cfg.get("policy_update_every", 100)) == 0:
                scores = {p: label_policy_nll(module, events, cur, p, registry, label) for p in POLICIES}
                p = min(scores, key=lambda x: (scores[x], x))
                chain["policy"] = p
                chain["score"] = objective(module, pair, uni, cur, transition, stationary) - scores[p]
                policy_updates += 1
            candidate = (chain["score"], tuple(cur), chain["policy"])
            if candidate > best:
                best = candidate
        if len(chains) > 1 and step % 25 == 0:
            for i in range(len(chains) - 1):
                a, b = chains[i], chains[i + 1]
                log_alpha = (1.0 / a["temp"] - 1.0 / b["temp"]) * (b["score"] - a["score"])
                if log_alpha >= 0 or rng.random() < math.exp(max(-700.0, log_alpha)):
                    a["mapping"], b["mapping"] = b["mapping"], a["mapping"]
                    a["policy"], b["policy"] = b["policy"], a["policy"]
                    a["score"], b["score"] = b["score"], a["score"]
    _, mapping, policy = best
    return tuple(mapping), {
        "solver": "bayes", "policy_hint": policy,
        "acceptance_rate": accepted / max(1, steps * len(chains)),
        "policy_updates": policy_updates,
    }


SOLVERS = {"heuristic": heuristic_solver, "beam": beam_solver, "bayes": bayes_solver}


def fit_candidate(module, gr, train, registry, external, seed, solver_name, cfg):
    selected_selector, selector_scores = module.select_selector(train, registry)
    solver = SOLVERS[solver_name]
    best = None
    for scheme in module.KEY_SCHEMES:
        pairs, unis, starts = gr.sufficient_statistics(module, train, scheme, len(registry.cells))
        for null_count in module.NULL_COUNTS:
            for size_profile in module.SIZE_PROFILES:
                sizes = module.class_sizes(size_profile, len(registry.cells) - null_count)
                for external_profile in module.PROFILE_NAMES:
                    transition, stationary = module.extend_external_model(external, external_profile, null_count)
                    assignments, solver_meta = {}, {}
                    for key_index, label in enumerate(sorted(pairs)):
                        mapping, meta = solver(
                            module, label_events(module, train, scheme, label), label,
                            pairs[label], unis[label], sizes, null_count, transition,
                            stationary, registry,
                            seed ^ (key_index * 0x9E3779B1) ^ module.stable_seed(solver_name, scheme, null_count, size_profile, external_profile),
                            cfg,
                        )
                        assignments[label] = tuple(int(x) for x in mapping)
                        solver_meta[label] = meta
                    policy, policy_scores = infer_policy(module, train, assignments, scheme, registry)
                    external_bits, _ = gr.fast_cost_components(module, assignments, pairs, unis, starts, transition, stationary)
                    report = gr.structural_report(module, sizes, null_count, len(assignments), external_profile)
                    selection_score = (
                        report.partition_bits * len(assignments)
                        + report.topology_bits
                        + report.external_model_index_bits
                        + external_bits
                        + policy_scores[policy]
                        + math.log2(len(POLICIES))
                        + module.selector_nll(train, registry, selected_selector)
                    )
                    row = {
                        "scheme": scheme, "null_count": null_count,
                        "size_profile": size_profile, "external_profile": external_profile,
                        "sizes": sizes, "assignments": assignments,
                        "selector": selected_selector, "selector_scores": selector_scores,
                        "policy": policy, "policy_scores": policy_scores,
                        "selection_score": float(selection_score),
                        "solver": solver_name, "solver_meta": solver_meta,
                    }
                    tie = json.dumps(assignments, sort_keys=True)
                    if best is None or (row["selection_score"], tie) < (best["selection_score"], json.dumps(best["assignments"], sort_keys=True)):
                        best = row
    if best is None:
        raise RuntimeError("no candidate fitted")
    return best


def score_trial_v03(module, gr, events, registry, external, seed, solver_name, cfg, truth):
    train = [e for e in events if not e.test]
    test = [e for e in events if e.test]
    fitted = fit_candidate(module, gr, train, registry, external, seed, solver_name, cfg)
    transition, stationary = module.extend_external_model(external, fitted["external_profile"], fitted["null_count"])
    external_train, external_test = external_nll_by_split(module, events, fitted["assignments"], fitted["scheme"], transition, stationary)
    policy_train, policy_test = policy_nll(module, events, fitted["assignments"], fitted["scheme"], fitted["policy"], registry, return_by_split=True)
    selector_train = module.selector_nll(train, registry, fitted["selector"])
    selector_test = module.selector_nll(test, registry, fitted["selector"])
    cipher_train = external_train + policy_train + selector_train
    cipher_test = external_test + policy_test + selector_test

    production_selector, _ = module.select_selector(train, registry)
    production_test = module.production_predictive_nll(test, train, registry, production_selector)
    production_train = module.production_predictive_nll(train, train, registry, production_selector)
    c_full = module.cipher_model_record(registry, fitted["assignments"], fitted["scheme"], fitted["sizes"], fitted["null_count"], fitted["size_profile"], fitted["external_profile"], fitted["selector"], train, full=True)
    c_cond = module.cipher_model_record(registry, fitted["assignments"], fitted["scheme"], fitted["sizes"], fitted["null_count"], fitted["size_profile"], fitted["external_profile"], fitted["selector"], train, full=False)
    p_full = module.production_model_record(registry, production_selector, train, full=True)
    p_cond = module.production_model_record(registry, production_selector, train, full=False)
    cfr, ccr = module.cost_model(c_full), module.cost_model(c_cond)
    pfr, pcr = module.cost_model(p_full), module.cost_model(p_cond)
    policy_index_bits = math.log2(len(POLICIES))
    h_full_c = cfr.canonical_serialization_bits + policy_index_bits + cipher_train + cipher_test
    h_full_p = pfr.canonical_serialization_bits + production_train + production_test
    h_cond_c = ccr.canonical_serialization_bits + policy_index_bits + cipher_train + cipher_test
    h_cond_p = pcr.canonical_serialization_bits + production_train + production_test
    extra_partition = ccr.partition_bits * max(0, len(fitted["assignments"]) - 1)
    i_c = ccr.structural_universal_bits + extra_partition + policy_index_bits + cipher_train + cipher_test
    i_p = pcr.structural_universal_bits + production_train + production_test
    differences = {
        "H_full_cipher_minus_production": h_full_c - h_full_p,
        "H_conditional_cipher_minus_production": h_cond_c - h_cond_p,
        "I_full_cipher_minus_production": i_c - i_p,
        "I_conditional_cipher_minus_production": i_c - i_p,
        "heldout_predictive_cipher_minus_production": cipher_test - production_test,
    }
    signs = all(differences[k] < 0 for k in ("H_full_cipher_minus_production", "H_conditional_cipher_minus_production", "I_full_cipher_minus_production", "I_conditional_cipher_minus_production"))
    gain = (production_test - cipher_test) / max(1, len(test))
    cipher_selected = signs and gain >= -0.025
    result = {
        "n_train": len(train), "n_test": len(test), "solver": solver_name,
        "fitted": {
            "scheme": fitted["scheme"], "null_count": fitted["null_count"],
            "size_profile": fitted["size_profile"], "external_profile": fitted["external_profile"],
            "selector": fitted["selector"], "selection_policy": fitted["policy"],
            "class_sizes": list(fitted["sizes"]),
        },
        "production_selector": production_selector,
        "differences_bits": differences,
        "predictive_gain_bits_per_test_token": gain,
        "cipher_selected": bool(cipher_selected),
        "solver_meta": fitted["solver_meta"], "policy_scores": fitted["policy_scores"],
    }
    if truth is not None:
        accuracy = module.mapping_accuracy(fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"])
        nf1 = module.null_f1(fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"])
        selector_correct = fitted["selector"] == truth["selector"]
        policy_correct = fitted["policy"] == truth["selection_policy"]
        structure_correct = (
            fitted["scheme"] == truth["key_scheme"]
            and fitted["null_count"] == truth["null_count"]
            and fitted["size_profile"] == truth["size_profile"]
            and fitted["external_profile"] == truth["external_profile"]
        )
        test_errors = sum(mapping_unit(module, e, fitted["assignments"], fitted["scheme"]) != int(e.true_unit) for e in test)
        latent_error = test_errors / max(1, len(test))
        threshold = 0.55 if truth["key_scheme"] == "currier" else 0.65
        positive_success = cipher_selected and accuracy >= threshold and nf1 >= 0.50 and policy_correct and latent_error <= 0.35
        result.update({
            "truth": {k: v for k, v in truth.items() if k != "keys"},
            "mapping_accuracy": accuracy, "null_f1": nf1,
            "selector_correct": selector_correct, "policy_correct": policy_correct,
            "structure_correct": structure_correct, "latent_unit_error": latent_error,
            "positive_success": bool(positive_success),
        })
    return result


def wilson(k, n, z=1.6448536269514722):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - r), min(1.0, c + r)]


def aggregate(results):
    positives = [r for r in results if r["trial_type"] == "positive"]
    controls = [r for r in results if r["trial_type"] == "control"]
    successes = sum(bool(r.get("positive_success")) for r in positives)
    fps = sum(bool(r.get("false_positive")) for r in controls)
    summary = {
        "positive": {
            "successes": successes, "trials": len(positives), "wilson90": wilson(successes, len(positives)),
            "median_mapping_accuracy": statistics.median([float(r["mapping_accuracy"]) for r in positives]) if positives else None,
            "median_null_f1": statistics.median([float(r["null_f1"]) for r in positives]) if positives else None,
            "policy_recovery": sum(bool(r["policy_correct"]) for r in positives) / max(1, len(positives)),
            "selector_recovery": sum(bool(r["selector_correct"]) for r in positives) / max(1, len(positives)),
            "structure_recovery": sum(bool(r["structure_correct"]) for r in positives) / max(1, len(positives)),
            "median_latent_unit_error": statistics.median([float(r["latent_unit_error"]) for r in positives]) if positives else None,
        },
        "control": {"false_positives": fps, "trials": len(controls), "wilson90": wilson(fps, len(controls))},
        "positive_strata": {}, "control_strata": {},
    }
    for dim in ("selection_policy", "key_scheme", "null_count", "size_profile", "external_profile", "selector", "length_profile"):
        buckets = defaultdict(list)
        for r in positives:
            buckets[str(r["truth"].get(dim))].append(r)
        summary["positive_strata"][dim] = {
            value: {
                "successes": sum(bool(x["positive_success"]) for x in rows),
                "trials": len(rows),
                "wilson90": wilson(sum(bool(x["positive_success"]) for x in rows), len(rows)),
            }
            for value, rows in sorted(buckets.items())
        }
    buckets = defaultdict(list)
    for r in controls:
        buckets[str(r["control_family"])].append(r)
    summary["control_strata"] = {
        value: {
            "false_positives": sum(bool(x["false_positive"]) for x in rows),
            "trials": len(rows),
            "wilson90": wilson(sum(bool(x["false_positive"]) for x in rows), len(rows)),
        }
        for value, rows in sorted(buckets.items())
    }
    return summary


def run_task(repo_text, solver_name, cfg, task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    repo = Path(repo_text)
    gr, module = load_v02(repo)
    record_path, ci_path = module.locate_data(repo)
    registry = module.build_surface_registry(pickle.load(record_path.open("rb")))
    external = module.build_external_models(pickle.load(ci_path.open("rb")))
    kind, index, seed = task["kind"], int(task["index"]), int(task["seed"])
    if kind == "positive":
        scenario = module.scenario_for_index(index)
        events, keys, sizes = module.generate_cipher_trial(
            registry, external, seed, str(scenario["key_scheme"]), int(scenario["null_count"]),
            str(scenario["size_profile"]), str(scenario["external_profile"]),
            str(scenario["selection_policy"]), str(scenario["selector"]),
        )
        length_profile = task["length_profile"]
        events = subset_events(events, LENGTHS[length_profile])
        truth = {**scenario, "keys": keys, "class_sizes": list(sizes), "length_profile": length_profile}
        result = score_trial_v03(module, gr, events, registry, external, seed ^ 0xC1F3, solver_name, cfg, truth)
        result.update({"trial_type": "positive", "trial_index": index, "seed": seed})
    else:
        family = module.CONTROL_FAMILIES[index % len(module.CONTROL_FAMILIES)]
        selector = module.SELECTORS[(index // len(module.CONTROL_FAMILIES)) % len(module.SELECTORS)]
        events = module.generate_control_trial(registry, external, seed, family, selector)
        length_profile = task["length_profile"]
        events = subset_events(events, LENGTHS[length_profile])
        result = score_trial_v03(module, gr, events, registry, external, seed ^ 0xF00D, solver_name, cfg, None)
        result.update({
            "trial_type": "control", "control_family": family,
            "true_selector": selector, "trial_index": index, "seed": seed,
            "length_profile": length_profile, "false_positive": bool(result["cipher_selected"]),
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--solver", choices=sorted(SOLVERS), required=True)
    ap.add_argument("--positives", type=int, default=48)
    ap.add_argument("--controls", type=int, default=32)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=3030303)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = {
        "steps": 2500, "restarts": 8, "alternations": 2,
        "refine_iterations": 1000, "refine_temperature": 1.0,
        "beam_width": 256, "policy_rerank": 64,
        "mcmc_steps": 5000, "temperatures": [1.0, 2.0, 4.0, 8.0],
        "policy_update_every": 100,
    }
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    lengths = tuple(LENGTHS)
    tasks = [
        {"kind": "positive", "index": i, "seed": args.seed + 100000 + i * 7919, "length_profile": lengths[i % len(lengths)]}
        for i in range(args.positives)
    ] + [
        {"kind": "control", "index": i, "seed": args.seed + 900000 + i * 104729, "length_profile": lengths[i % len(lengths)]}
        for i in range(args.controls)
    ]
    started, results = time.time(), []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(run_task, str(args.repo), args.solver, cfg, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            task = futures[future]
            print(
                f"V03_PROGRESS solver={args.solver} completed={done}/{len(tasks)} "
                f"kind={task['kind']} index={task['index']} length={task['length_profile']} "
                f"elapsed={time.time()-started:.1f}s", flush=True,
            )
    results.sort(key=lambda r: (r["trial_type"], int(r["trial_index"])))
    summary = aggregate(results)
    payload = {
        "programme": "morpholocal-calibration-v0.3-development", "solver": args.solver,
        "seed": args.seed, "config": cfg,
        "parameters": {
            "positives": args.positives, "controls": args.controls, "workers": args.workers,
            "lengths": LENGTHS, "v02_generator_compatibility_track": True,
            "policy_aware_accounting": True, "runtime_v02_patch_loader": True,
            "formal_result": False,
        },
        "summary": summary, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("V03_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
