#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V03 = ROOT / "experiments" / "morpholocal_calibration_v0_3"
V05 = ROOT / "experiments" / "recoverability_frontier_v0_5"
for path in (V03, V05):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generator_disjoint_v034 as gen
import recoverability_v050 as core

ALPHA = 0.5
N_PAYLOAD = 12
BASE_PROFILES = ("loo_general", "loo_dmm", "pooled")
ORDERS = (2, 3)
POSITIVE_FAMILIES = gen.POSITIVE_FAMILIES
CONTROL_FAMILIES = gen.CONTROL_FAMILIES
CORPORA = tuple(sorted(gen.CORPORA))
POLICIES = tuple(gen.base.POLICIES)

DEFAULT_CFG = {
    "beam_width": 256,
    "policy_rerank": 64,
    "refine_iterations": 900,
    "steps": 2500,
    "restarts": 8,
    "alternations": 2,
    "refine_temperature": 1.0,
}

_PROCESS_CACHE: dict[str, Any] = {}


def stable_seed(*parts: object) -> int:
    return core.stable_seed("v070", *parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunked(values: Sequence[int], width: int = 64) -> list[list[int]]:
    return [list(map(int, values[i : i + width])) for i in range(0, len(values), width) if values[i : i + width]]


def classified_sequences(module: Any, words: Iterable[str], width: int = 64) -> list[list[int]]:
    values = [int(module.classify_word(str(word))) for word in words if str(word).strip()]
    return chunked(values, width)


def source_sequences(module: Any, repo: Path) -> tuple[dict[str, list[list[int]]], dict[str, str]]:
    sequences: dict[str, list[list[int]]] = {}
    hashes: dict[str, str] = {}

    for name, relative in sorted(gen.CORPORA.items()):
        path = repo / relative
        payload = pickle.load(path.open("rb"))
        sequences[name] = classified_sequences(module, payload["all_words"])
        hashes[name] = sha256_file(path)

    languages = core.load_languages(
        V05 / "corpus_manifest_v050.json",
        repo / ".cache" / "v070-ud",
    )
    for iso, language in sorted(languages.items()):
        rows: list[list[int]] = []
        for text in language.texts["train"]:
            words = [word for word in text.split() if word]
            if words:
                rows.append([int(module.classify_word(word)) for word in words])
        sequences[f"ud_{iso}"] = [row for row in rows if row]
    hashes["ud_manifest"] = sha256_file(V05 / "corpus_manifest_v050.json")
    return sequences, hashes


def kt_models(rows: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    stationary = np.full(N_PAYLOAD, ALPHA, dtype=np.float64)
    bigram = np.full((N_PAYLOAD, N_PAYLOAD), ALPHA, dtype=np.float64)
    trigram = np.full((N_PAYLOAD, N_PAYLOAD, N_PAYLOAD), ALPHA, dtype=np.float64)
    for row in rows:
        values = [int(x) for x in row if 0 <= int(x) < N_PAYLOAD]
        if not values:
            continue
        stationary[values[0]] += 1.0
        for left, right in zip(values, values[1:]):
            bigram[left, right] += 1.0
        for first, second, third in zip(values, values[1:], values[2:]):
            trigram[first, second, third] += 1.0
    stationary /= stationary.sum()
    bigram /= bigram.sum(axis=1, keepdims=True)
    trigram /= trigram.sum(axis=2, keepdims=True)
    return {"stationary": stationary, "transition": bigram, "trigram": trigram}


def build_source_registry(module: Any, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sequences, hashes = source_sequences(module, repo)
    all_names = sorted(sequences)
    profile_exclusions = {
        "loo_general": {"greek_general"},
        "loo_dmm": {"greek_dmm"},
        "pooled": set(),
    }
    transition: dict[str, np.ndarray] = {}
    stationary: dict[str, np.ndarray] = {}
    trigrams: dict[str, np.ndarray] = {}
    members: dict[str, list[str]] = {}
    for profile, excluded in profile_exclusions.items():
        selected = [name for name in all_names if name not in excluded]
        rows = [row for name in selected for row in sequences[name]]
        model = kt_models(rows)
        transition[profile] = model["transition"]
        stationary[profile] = model["stationary"]
        trigrams[profile] = model["trigram"]
        members[profile] = selected
    source_hash = hashlib.sha256(json.dumps({"hashes": hashes, "members": members}, sort_keys=True).encode()).hexdigest()
    external = {"transition": transition, "stationary": stationary, "source_hash": source_hash}
    meta = {"trigram": trigrams, "members": members, "hashes": hashes, "source_hash": source_hash}
    return external, meta


def get_assets(repo: Path):
    cached = _PROCESS_CACHE.get("assets")
    if cached is not None:
        return cached
    gr, module, registry, _old_external = gen.load_assets(repo)
    source_external, source_meta = build_source_registry(module, repo)
    cached = (gr, module, registry, source_external, source_meta)
    _PROCESS_CACHE["assets"] = cached
    return cached


def decoded_lines(module: Any, events: Sequence[Any], assignments: dict[str, Sequence[int]], scheme: str) -> list[list[int]]:
    lines: list[list[int]] = []
    current: list[int] = []
    previous = None
    for event in events:
        marker = (int(event.doc), int(event.line))
        if previous is not None and marker != previous:
            if current:
                lines.append(current)
            current = []
        current.append(int(gen.base.mapping_unit(module, event, assignments, scheme)))
        previous = marker
    if current:
        lines.append(current)
    return lines


def trigram_bits(
    module: Any,
    events: Sequence[Any],
    assignments: dict[str, Sequence[int]],
    scheme: str,
    transition: np.ndarray,
    stationary: np.ndarray,
    trigram: np.ndarray,
) -> float:
    bits = 0.0
    for line in decoded_lines(module, events, assignments, scheme):
        if not line:
            continue
        first = line[0]
        bits -= math.log2(max(1e-300, float(stationary[first])))
        if len(line) >= 2:
            bits -= math.log2(max(1e-300, float(transition[line[0], line[1]])))
        for a, b, c in zip(line, line[1:], line[2:]):
            if a < N_PAYLOAD and b < N_PAYLOAD and c < N_PAYLOAD:
                probability = float(trigram[a, b, c])
            else:
                probability = float(transition[b, c])
            bits -= math.log2(max(1e-300, probability))
    return float(bits)


def source_bits(
    module: Any,
    events: Sequence[Any],
    fitted: dict[str, Any],
    external: dict[str, Any],
    source_meta: dict[str, Any],
    order: int,
) -> float:
    transition, stationary = module.extend_external_model(external, fitted["external_profile"], fitted["null_count"])
    if order == 2:
        train_bits, test_bits = gen.base.external_nll_by_split(
            module,
            events,
            fitted["assignments"],
            fitted["scheme"],
            transition,
            stationary,
        )
        return float(train_bits + test_bits)
    return trigram_bits(
        module,
        events,
        fitted["assignments"],
        fitted["scheme"],
        transition,
        stationary,
        source_meta["trigram"][fitted["external_profile"]],
    )


def fit_for_profile(
    module: Any,
    gr: Any,
    train: Sequence[Any],
    registry: Any,
    external: dict[str, Any],
    profile: str,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    original_profiles = tuple(module.PROFILE_NAMES)
    try:
        module.PROFILE_NAMES = (profile,)
        fitted = gen.base.fit_candidate(
            module,
            gr,
            list(train),
            registry,
            external,
            int(seed),
            "beam",
            cfg,
        )
    finally:
        module.PROFILE_NAMES = original_profiles
    return fitted


def model_record_bits(
    module: Any,
    registry: Any,
    fitted: dict[str, Any],
    train: Sequence[Any],
    full: bool,
) -> float:
    original_profiles = tuple(module.PROFILE_NAMES)
    try:
        module.PROFILE_NAMES = BASE_PROFILES
        record = module.cipher_model_record(
            registry,
            fitted["assignments"],
            fitted["scheme"],
            fitted["sizes"],
            fitted["null_count"],
            fitted["size_profile"],
            fitted["external_profile"],
            fitted["selector"],
            list(train),
            full=full,
        )
        return float(module.cost_model(record).canonical_serialization_bits)
    finally:
        module.PROFILE_NAMES = original_profiles


def choose_main_fit(
    module: Any,
    gr: Any,
    train: Sequence[Any],
    registry: Any,
    external: dict[str, Any],
    source_meta: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for profile_index, profile in enumerate(BASE_PROFILES):
        fitted = fit_for_profile(
            module, gr, train, registry, external, profile,
            seed ^ stable_seed("profile", profile_index), cfg,
        )
        policy_train = float(gen.base.policy_nll(
            module,
            list(train),
            fitted["assignments"],
            fitted["scheme"],
            fitted["policy"],
            registry,
        ))
        selector_train = float(module.selector_nll(list(train), registry, fitted["selector"]))
        structural = model_record_bits(module, registry, fitted, train, full=False)
        for order in ORDERS:
            src = source_bits(module, train, fitted, external, source_meta, order)
            score = structural + math.log2(len(ORDERS)) + math.log2(len(POLICIES)) + src + policy_train + selector_train
            candidates.append({
                "fitted": fitted,
                "source_order": int(order),
                "selection_score": float(score),
                "train_source_bits": float(src),
                "train_policy_bits": policy_train,
                "train_selector_bits": selector_train,
            })
    candidates.sort(key=lambda row: (row["selection_score"], row["fitted"]["external_profile"], row["source_order"]))
    return candidates[0]


def score_fit(
    module: Any,
    registry: Any,
    external: dict[str, Any],
    source_meta: dict[str, Any],
    train: Sequence[Any],
    test: Sequence[Any],
    chosen: dict[str, Any],
) -> dict[str, float]:
    fitted = chosen["fitted"]
    order = int(chosen["source_order"])
    source_train = source_bits(module, train, fitted, external, source_meta, order)
    source_test = source_bits(module, test, fitted, external, source_meta, order)
    policy_train = float(gen.base.policy_nll(
        module, list(train), fitted["assignments"], fitted["scheme"], fitted["policy"], registry,
    ))
    policy_test = float(gen.base.policy_nll(
        module, list(test), fitted["assignments"], fitted["scheme"], fitted["policy"], registry,
    ))
    selector_train = float(module.selector_nll(list(train), registry, fitted["selector"]))
    selector_test = float(module.selector_nll(list(test), registry, fitted["selector"]))
    cipher_train = source_train + policy_train + selector_train
    cipher_test = source_test + policy_test + selector_test

    production_selector, _ = module.select_selector(list(train), registry)
    production_train = float(module.production_predictive_nll(list(train), list(train), registry, production_selector))
    production_test = float(module.production_predictive_nll(list(test), list(train), registry, production_selector))

    c_full_bits = model_record_bits(module, registry, fitted, train, full=True)
    c_cond_bits = model_record_bits(module, registry, fitted, train, full=False)
    p_full = module.production_model_record(registry, production_selector, list(train), full=True)
    p_cond = module.production_model_record(registry, production_selector, list(train), full=False)
    p_full_bits = float(module.cost_model(p_full).canonical_serialization_bits)
    p_cond_bits = float(module.cost_model(p_cond).canonical_serialization_bits)

    order_bits = math.log2(len(ORDERS))
    policy_bits = math.log2(len(POLICIES))
    h_full_c = c_full_bits + order_bits + policy_bits + cipher_train + cipher_test
    h_cond_c = c_cond_bits + order_bits + policy_bits + cipher_train + cipher_test
    h_full_p = p_full_bits + production_train + production_test
    h_cond_p = p_cond_bits + production_train + production_test
    return {
        "cipher_train_bits": float(cipher_train),
        "cipher_test_bits": float(cipher_test),
        "production_train_bits": float(production_train),
        "production_test_bits": float(production_test),
        "full_difference_bits": float(h_full_c - h_full_p),
        "conditional_difference_bits": float(h_cond_c - h_cond_p),
        "heldout_gain_bits_per_token": float((production_test - cipher_test) / max(1, len(test))),
        "total_gain_bits_per_token": float((h_full_p - h_full_c) / max(1, len(train) + len(test))),
    }


def training_folds(train: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    docs = sorted({int(event.doc) for event in train})
    if len(docs) < 4:
        return list(train), list(train)
    exclude_a = {doc for index, doc in enumerate(docs) if index % 4 == 0}
    exclude_b = {doc for index, doc in enumerate(docs) if index % 4 == 1}
    fold_a = [event for event in train if int(event.doc) not in exclude_a]
    fold_b = [event for event in train if int(event.doc) not in exclude_b]
    return fold_a, fold_b


def aligned_mapping_agreement(left: dict[str, Sequence[int]], right: dict[str, Sequence[int]]) -> float:
    if set(left) != set(right):
        return 0.0
    agreements = []
    for label in sorted(left):
        a = np.asarray(left[label], dtype=np.int64)
        b = np.asarray(right[label], dtype=np.int64)
        n = max(int(a.max(initial=0)), int(b.max(initial=0))) + 1
        contingency = np.zeros((n, n), dtype=np.int64)
        np.add.at(contingency, (a, b), 1)
        rows, cols = linear_sum_assignment(-contingency)
        agreements.append(float(contingency[rows, cols].sum() / max(1, len(a))))
    return float(statistics.fmean(agreements)) if agreements else 0.0


def fold_stability(
    module: Any,
    gr: Any,
    registry: Any,
    external: dict[str, Any],
    source_meta: dict[str, Any],
    train: Sequence[Any],
    test: Sequence[Any],
    selected: dict[str, Any],
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    profile = str(selected["fitted"]["external_profile"])
    order = int(selected["source_order"])
    folds = training_folds(train)
    fits = []
    gains = []
    for index, fold in enumerate(folds):
        fitted = fit_for_profile(
            module, gr, fold, registry, external, profile,
            seed ^ stable_seed("fold", index), cfg,
        )
        chosen = {"fitted": fitted, "source_order": order}
        scored = score_fit(module, registry, external, source_meta, fold, test, chosen)
        fits.append(fitted)
        gains.append(float(scored["heldout_gain_bits_per_token"]))
    structure_same = (
        fits[0]["scheme"] == fits[1]["scheme"]
        and fits[0]["null_count"] == fits[1]["null_count"]
        and fits[0]["size_profile"] == fits[1]["size_profile"]
    )
    agreement = aligned_mapping_agreement(fits[0]["assignments"], fits[1]["assignments"]) if structure_same else 0.0
    return {
        "agreement": float(agreement),
        "structure_same": bool(structure_same),
        "heldout_gains": gains,
        "same_sign_positive": bool(all(value > 0.0 for value in gains)),
    }


def positive_truth_metrics(module: Any, test: Sequence[Any], fitted: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    accuracy = float(module.mapping_accuracy(
        fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"]
    ))
    null_f1 = float(module.null_f1(
        fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"]
    ))
    errors = sum(
        gen.base.mapping_unit(module, event, fitted["assignments"], fitted["scheme"]) != int(event.true_unit)
        for event in test
    )
    latent_error = errors / max(1, len(test))
    structure_correct = fitted["scheme"] == truth["key_scheme"] and fitted["null_count"] == truth["null_count"]
    return {
        "mapping_accuracy": accuracy,
        "null_f1": null_f1,
        "latent_unit_error": float(latent_error),
        "structure_correct": bool(structure_correct),
    }


def generate_trial(repo: Path, module: Any, registry: Any, task: dict[str, Any]):
    seed = int(task["seed"])
    plan = module.document_plan(registry, seed, n_docs=12, tokens_per_doc=180)
    if task["trial_type"] == "positive":
        family = str(task["family"])
        corpus = str(task["corpus"])
        key_scheme, null_count, size_profile, mechanism = gen.family_spec(family)
        words = gen.load_words(repo, corpus)
        latent = gen.corpus_lines(module, words, plan, seed, null_count)
        events, keys, sizes = gen.render_lines(
            module, registry, plan, latent, seed, key_scheme, null_count,
            size_profile, mechanism, True,
        )
        truth = {
            "keys": keys,
            "key_scheme": key_scheme,
            "null_count": null_count,
            "size_profile": size_profile,
            "class_sizes": list(sizes),
        }
        return events, truth

    family = str(task["family"])
    latent = gen.ordered_control_lines(plan, family, seed)
    mechanisms = ("prf", "rotor", "feedback", "line_keyed")
    mechanism = mechanisms[int(task["replicate"]) % len(mechanisms)]
    key_scheme = "global" if int(task["replicate"]) % 2 == 0 else "currier"
    events, _keys, _sizes = gen.render_lines(
        module, registry, plan, latent, seed, key_scheme, 0,
        "unequal" if key_scheme == "global" else "balanced",
        mechanism, False,
    )
    return events, None


def run_task(repo_text: str, task: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    repo = Path(repo_text)
    gr, module, registry, external, source_meta = get_assets(repo)
    events, truth = generate_trial(repo, module, registry, task)
    train = [event for event in events if not event.test]
    test = [event for event in events if event.test]
    if not train or not test:
        raise RuntimeError("empty document split")
    if {int(e.doc) for e in train} & {int(e.doc) for e in test}:
        raise RuntimeError("document leakage")

    seed = int(task["seed"])
    selected = choose_main_fit(module, gr, train, registry, external, source_meta, seed ^ 0x7001, cfg)
    scored = score_fit(module, registry, external, source_meta, train, test, selected)
    stability = fold_stability(
        module, gr, registry, external, source_meta, train, test, selected,
        seed ^ 0x7002, cfg,
    )
    fitted = selected["fitted"]
    target_profile = None
    if task["trial_type"] == "positive":
        target_profile = "loo_general" if task["corpus"] == "greek_general" else "loo_dmm"
    primary_source = target_profile is None or fitted["external_profile"] == target_profile

    decision = (
        scored["total_gain_bits_per_token"] >= 0.05
        and scored["heldout_gain_bits_per_token"] >= 0.02
        and scored["full_difference_bits"] < 0.0
        and scored["conditional_difference_bits"] < 0.0
        and primary_source
        and stability["agreement"] >= 0.70
        and stability["same_sign_positive"]
    )

    result: dict[str, Any] = {
        **task,
        "n_train": len(train),
        "n_test": len(test),
        "source_hash": source_meta["source_hash"],
        "selected": {
            "scheme": fitted["scheme"],
            "null_count": fitted["null_count"],
            "size_profile": fitted["size_profile"],
            "source_profile": fitted["external_profile"],
            "source_order": int(selected["source_order"]),
            "policy": fitted["policy"],
            "selector": fitted["selector"],
        },
        "primary_leave_target_out": bool(primary_source),
        "accounting": scored,
        "stability": stability,
        "cipher_selected": bool(decision),
    }
    if truth is not None:
        truth_metrics = positive_truth_metrics(module, test, fitted, truth)
        success = (
            decision
            and truth_metrics["mapping_accuracy"] >= 0.55
            and truth_metrics["latent_unit_error"] <= 0.35
            and truth_metrics["structure_correct"]
        )
        result["truth_metrics"] = truth_metrics
        result["positive_success"] = bool(success)
    else:
        result["false_positive"] = bool(decision)
    return result


def wilson(k: int, n: int, z: float = 1.6448536269514722) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in results if row["trial_type"] == "positive"]
    controls = [row for row in results if row["trial_type"] == "control"]
    successes = sum(bool(row.get("positive_success")) for row in positives)
    false_positives = sum(bool(row.get("false_positive")) for row in controls)

    def strata(rows, key, outcome):
        buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        return {
            name: {
                "successes": sum(bool(item.get(outcome)) for item in items),
                "trials": len(items),
                "rate": sum(bool(item.get(outcome)) for item in items) / max(1, len(items)),
                "wilson90": wilson(sum(bool(item.get(outcome)) for item in items), len(items)),
            }
            for name, items in sorted(buckets.items())
        }

    positive_gains = [float(row["accounting"]["heldout_gain_bits_per_token"]) for row in positives]
    control_gains = [float(row["accounting"]["heldout_gain_bits_per_token"]) for row in controls]
    family = strata(positives, "family", "positive_success")
    corpus = strata(positives, "corpus", "positive_success")
    control_family = strata(controls, "family", "false_positive")

    gate = (
        successes / max(1, len(positives)) >= 0.70
        and all(row["rate"] >= 0.50 for row in family.values())
        and all(row["rate"] >= 0.60 for row in corpus.values())
        and false_positives / max(1, len(controls)) <= 0.10
        and all(row["rate"] <= 0.20 for row in control_family.values())
        and (statistics.median(positive_gains) if positive_gains else -math.inf) > 0.0
        and (statistics.median(control_gains) if control_gains else math.inf) <= 0.0
    )
    return {
        "positive": {
            "successes": successes,
            "trials": len(positives),
            "rate": successes / max(1, len(positives)),
            "wilson90": wilson(successes, len(positives)),
            "median_heldout_gain": statistics.median(positive_gains) if positive_gains else None,
            "family": family,
            "corpus": corpus,
        },
        "control": {
            "false_positives": false_positives,
            "trials": len(controls),
            "rate": false_positives / max(1, len(controls)),
            "wilson90": wilson(false_positives, len(controls)),
            "median_heldout_gain": statistics.median(control_gains) if control_gains else None,
            "family": control_family,
        },
        "gate_pass": bool(gate),
        "decision": "GO_TO_NEXT_STAGE" if gate else "STOP_V070_STAGE_A_FAILED",
    }


def tasks_for(split: str) -> list[dict[str, Any]]:
    if split == "engineering":
        return [
            {"split": split, "trial_type": "positive", "family": "prf_global", "corpus": "greek_general", "replicate": 0, "seed": stable_seed(split, "positive", 0)},
            {"split": split, "trial_type": "control", "family": "motif_grammar", "replicate": 0, "seed": stable_seed(split, "control", 0)},
        ]
    reps_positive = 4 if split == "dev" else 8
    reps_control = 8 if split == "dev" else 16
    tasks: list[dict[str, Any]] = []
    for family in POSITIVE_FAMILIES:
        for corpus in CORPORA:
            for replicate in range(reps_positive):
                tasks.append({
                    "split": split,
                    "trial_type": "positive",
                    "family": family,
                    "corpus": corpus,
                    "replicate": replicate,
                    "seed": stable_seed(split, "positive", family, corpus, replicate),
                })
    for family in CONTROL_FAMILIES:
        for replicate in range(reps_control):
            tasks.append({
                "split": split,
                "trial_type": "control",
                "family": family,
                "replicate": replicate,
                "seed": stable_seed(split, "control", family, replicate),
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("engineering", "dev", "test"), required=True)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.config:
        cfg.update(json.loads(args.config.read_text(encoding="utf-8")))
    all_tasks = tasks_for(args.split)
    tasks = [task for index, task in enumerate(all_tasks) if index % args.num_shards == args.shard_index]
    if not tasks:
        raise RuntimeError("empty shard")

    started = time.time()
    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(run_task, str(args.repo), task, cfg): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            row = future.result()
            results.append(row)
            print(
                "V070_PROGRESS",
                json.dumps({
                    "completed": completed,
                    "total": len(tasks),
                    "type": task["trial_type"],
                    "family": task["family"],
                    "corpus": task.get("corpus"),
                    "selected": row["cipher_selected"],
                    "success": row.get("positive_success"),
                    "false_positive": row.get("false_positive"),
                    "heldout_gain": row["accounting"]["heldout_gain_bits_per_token"],
                    "elapsed_seconds": time.time() - started,
                }, sort_keys=True),
                flush=True,
            )

    results.sort(key=lambda row: (row["trial_type"], row["family"], row.get("corpus", ""), int(row["replicate"])))
    summary = aggregate(results)
    payload = {
        "programme": "cipher-generated-mdl-v0.7-stage-a",
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "config": cfg,
        "task_count": len(tasks),
        "elapsed_seconds": time.time() - started,
        "results": results,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V070_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V070_SHA256", payload["sha256"], flush=True)
    print("V070_OUTPUT", str(args.output), flush=True)


if __name__ == "__main__":
    main()
