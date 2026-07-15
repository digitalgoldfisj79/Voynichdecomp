#!/usr/bin/env python3
"""Correctness and audit launcher for the frozen v0.3 neural checkpoint.

This applies the v0.3.1 order, cache-isolation and charged production-null
corrections to the neural evaluation path without retraining the model. It is
a development-only secondary comparison, not an independent evidential vote.
"""
from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neural_runner  # noqa: E402
import production_null_registry as production  # noqa: E402

base = neural_runner.base
fast = neural_runner.fast

POLICY_CACHE_LIMIT = 32
LABEL_CACHE_LIMIT = 64
PRODUCTION_CACHE_LIMIT = 16
_POLICY_CACHE: OrderedDict[int, tuple[object, Any]] = OrderedDict()
_LABEL_CACHE: OrderedDict[tuple[int, str, str], tuple[object, list[Any]]] = OrderedDict()
_PRODUCTION_CACHE: OrderedDict[int, tuple[object, production.FittedNull]] = OrderedDict()
_ACTIVE_AUDIT: dict[str, Any] = {}


def _store(cache: OrderedDict, key: Any, value: tuple[object, Any], limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def safe_label_events(module, train, scheme, label):
    key = (id(train), str(scheme), str(label))
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        source, rows = cached
        if source is train:
            _LABEL_CACHE.move_to_end(key)
            return rows
        del _LABEL_CACHE[key]
    rows = [event for event in train if module.key_label(event, scheme) == label]
    _store(_LABEL_CACHE, key, (train, rows), LABEL_CACHE_LIMIT)
    return rows


def safe_policy_prepare(module, events, registry):
    key = id(events)
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        source, value = cached
        if source is events:
            _POLICY_CACHE.move_to_end(key)
            return value
        del _POLICY_CACHE[key]

    contexts = sorted({(event.section, event.position) for event in events})
    context_lookup = {value: index for index, value in enumerate(contexts)}
    cells = np.fromiter((int(event.cell) for event in events), dtype=np.int64, count=len(events))
    context_index = np.fromiter(
        (context_lookup[(event.section, event.position)] for event in events),
        dtype=np.int64,
        count=len(events),
    )
    line_start = np.zeros(len(events), dtype=bool)
    previous = None
    for index, event in enumerate(events):
        marker = (event.doc, event.line)
        line_start[index] = marker != previous
        previous = marker
    weights = np.empty((len(contexts), len(registry.cells)), dtype=np.float64)
    for context_index_value, (section, position) in enumerate(contexts):
        for cell in range(len(registry.cells)):
            weights[context_index_value, cell] = max(
                1e-300,
                float(module.context_weight(registry, cell, section, position)),
            )
    value = fast.PolicyData(
        cells=cells,
        context_index=context_index,
        line_start=line_start,
        weights=weights,
    )
    _store(_POLICY_CACHE, key, (events, value), POLICY_CACHE_LIMIT)
    return value


def safe_subset_events(events: Sequence[Any], target: int) -> list[Any]:
    if target >= len(events):
        return list(events)
    if target <= 0:
        return []
    indexed = list(enumerate(events))
    train = [(index, event) for index, event in indexed if not event.test]
    test = [(index, event) for index, event in indexed if event.test]
    n_test = min(len(test), max(1, int(round(target * 0.2))))
    n_train = min(len(train), max(0, target - n_test))

    def take(rows, limit):
        groups = []
        current = []
        previous = None
        for index, event in rows:
            marker = (event.doc, event.line)
            if previous is not None and marker != previous:
                groups.append(current)
                current = []
            current.append((index, event))
            previous = marker
        if current:
            groups.append(current)
        selected = []
        for line in groups:
            if selected and len(selected) + len(line) > limit:
                break
            selected.extend(index for index, _ in line)
            if len(selected) >= limit:
                break
        return selected

    selected = set(take(train, n_train))
    selected.update(take(test, n_test))
    return [event for index, event in indexed if index in selected]


def safe_production_predictive_nll(data, train, registry, selector) -> float:
    key = id(train)
    fitted = None
    cache_hit = False
    cached = _PRODUCTION_CACHE.get(key)
    if cached is not None:
        source, candidate = cached
        if source is train:
            fitted = candidate
            cache_hit = True
            _PRODUCTION_CACHE.move_to_end(key)
        else:
            del _PRODUCTION_CACHE[key]
    if fitted is None:
        fitted = production.fit_registry(train)
        _store(_PRODUCTION_CACHE, key, (train, fitted), PRODUCTION_CACHE_LIMIT)

    if data is train:
        score = float(fitted.train_bits + math.log2(len(production.MODEL_NAMES)))
        _ACTIVE_AUDIT["production_train_bits"] = score
    else:
        score = float(production.score_fitted(data, fitted))
        _ACTIVE_AUDIT["production_test_bits"] = score
    _ACTIVE_AUDIT.update(
        {
            "production_model": fitted.name,
            "production_model_train_bits_without_index": float(fitted.train_bits),
            "production_registry_index_bits": float(math.log2(len(production.MODEL_NAMES))),
            "production_cache_hit": cache_hit,
        }
    )
    return score


def repo_from_args(arguments: list[str]) -> Path:
    if "--repo" in arguments:
        index = arguments.index("--repo")
        if index + 1 >= len(arguments):
            raise SystemExit("--repo requires a path")
        return Path(arguments[index + 1]).resolve()
    return base.DEFAULT_REPO.resolve()


def main() -> None:
    arguments = sys.argv[1:]
    repo = repo_from_args(arguments)

    fast.prepare = safe_policy_prepare
    base.label_events = safe_label_events
    base.label_policy_nll = fast.fast_label_policy_nll
    base.subset_events = safe_subset_events
    base.SOLVERS["neural"] = neural_runner.neural_solver
    base.fit_candidate = neural_runner.patched_fit_candidate

    original_fit = base.fit_candidate
    original_score = base.score_trial_v03

    def audited_fit(*args, **kwargs):
        fitted = original_fit(*args, **kwargs)
        _ACTIVE_AUDIT.update(
            {
                "cipher_selection_score_train": float(fitted["selection_score"]),
                "selected_scheme": fitted["scheme"],
                "selected_null_count": int(fitted["null_count"]),
                "selected_size_profile": fitted["size_profile"],
                "selected_external_profile": fitted["external_profile"],
                "selected_policy": fitted["policy"],
                "selected_selector": fitted["selector"],
            }
        )
        return fitted

    def audited_score(*args, **kwargs):
        _ACTIVE_AUDIT.clear()
        result = original_score(*args, **kwargs)
        difference = float(
            result["differences_bits"]["heldout_predictive_cipher_minus_production"]
        )
        production_test = _ACTIVE_AUDIT.get("production_test_bits")
        if production_test is not None:
            _ACTIVE_AUDIT["cipher_test_bits"] = float(production_test + difference)
        _ACTIVE_AUDIT.update(
            {
                "heldout_cipher_minus_production_bits": difference,
                "heldout_cipher_minus_production_bits_per_token": difference
                / max(1, int(result["n_test"])),
                "strict_heldout_advantage": difference < 0.0,
                "strict_cipher_selected": bool(result["cipher_selected"] and difference < 0.0),
                "scientific_solver_label": "same_generator_neural_decoder",
            }
        )
        result["remediation_audit"] = dict(_ACTIVE_AUDIT)
        return result

    base.fit_candidate = audited_fit
    base.score_trial_v03 = audited_score

    original_load = base.load_v02

    def corrected_load(path):
        gpu_runner, module = original_load(path)
        module.production_predictive_nll = safe_production_predictive_nll
        return gpu_runner, module

    base.load_v02 = corrected_load
    gpu_runner, module = base.load_v02(repo)
    base.load_v02 = lambda _repo: (gpu_runner, module)
    original_context = base.mp.get_context
    base.mp.get_context = lambda _method=None: original_context("fork")

    sys.argv = [sys.argv[0], *arguments]
    base.main()


if __name__ == "__main__":
    main()
