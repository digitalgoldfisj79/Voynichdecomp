#!/usr/bin/env python3
"""Development-only correctness wrapper for morpholocal calibration v0.3.1.

Corrections applied without changing the frozen v0.2 generator:

1. object-id caches retain the exact source object in bounded LRUs, preventing
   recycled Python ids from returning another trial's data;
2. length subsetting preserves source event order and never sorts by
   ``token_index`` (a vocabulary index, not a sequence position);
3. production-null selection remains training-only;
4. label-specific event caches are identity-safe.

The wrapper also emits trial-level audit fields. Formal use still requires a
static, patch-free effective-source freeze after validation.
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

import tournament_kt as kt  # noqa: E402
import production_null_registry as production  # noqa: E402

base = kt.base

PREPARE_CACHE_LIMIT = 32
PRODUCTION_CACHE_LIMIT = 16
LABEL_EVENTS_CACHE_LIMIT = 64
_PREPARE_CACHE: OrderedDict[int, tuple[object, kt.SequenceData]] = OrderedDict()
_PRODUCTION_CACHE: OrderedDict[int, tuple[object, production.FittedNull]] = OrderedDict()
_LABEL_EVENTS_CACHE: OrderedDict[tuple[int, str, str], tuple[object, list[Any]]] = OrderedDict()
_ACTIVE_AUDIT: dict[str, Any] = {}
_ORIGINAL_FIT_CANDIDATE = base.fit_candidate
_ORIGINAL_SCORE_TRIAL = base.score_trial_v03


def _bounded_store(cache: OrderedDict, key: Any, value: tuple[object, Any], limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def safe_label_events(module, train, scheme, label):
    """Return label events from an identity-safe bounded cache."""
    key = (id(train), str(scheme), str(label))
    cached = _LABEL_EVENTS_CACHE.get(key)
    if cached is not None:
        source, rows = cached
        if source is train:
            _LABEL_EVENTS_CACHE.move_to_end(key)
            return rows
        del _LABEL_EVENTS_CACHE[key]
    rows = [event for event in train if module.key_label(event, scheme) == label]
    _bounded_store(_LABEL_EVENTS_CACHE, key, (train, rows), LABEL_EVENTS_CACHE_LIMIT)
    return rows


def safe_prepare(events: Sequence[Any]) -> kt.SequenceData:
    """Prepare policy arrays with an identity-safe bounded cache."""
    key = id(events)
    cached = _PREPARE_CACHE.get(key)
    if cached is not None:
        source, value = cached
        if source is events:
            _PREPARE_CACHE.move_to_end(key)
            return value
        del _PREPARE_CACHE[key]

    context_values = sorted({(event.section, event.position) for event in events})
    context_index = {value: index for index, value in enumerate(context_values)}
    cells = np.asarray([int(event.cell) for event in events], dtype=np.int64)
    contexts = np.asarray(
        [context_index[(event.section, event.position)] for event in events],
        dtype=np.int64,
    )
    test = np.asarray([bool(event.test) for event in events], dtype=bool)
    line_start = np.zeros(len(events), dtype=bool)
    previous = None
    for index, event in enumerate(events):
        marker = (event.doc, event.line)
        line_start[index] = marker != previous
        previous = marker

    value = kt.SequenceData(cells, contexts, line_start, test, len(context_values))
    _bounded_store(_PREPARE_CACHE, key, (events, value), PREPARE_CACHE_LIMIT)
    return value


def safe_production_predictive_nll(data, train, registry, selector) -> float:
    """Fit/reuse a production null only for the exact training list object."""
    key = id(train)
    cached = _PRODUCTION_CACHE.get(key)
    fitted = None
    cache_hit = False
    if cached is not None:
        source, candidate = cached
        if source is train:
            _PRODUCTION_CACHE.move_to_end(key)
            fitted = candidate
            cache_hit = True
        else:
            del _PRODUCTION_CACHE[key]

    if fitted is None:
        fitted = production.fit_registry(train)
        _bounded_store(_PRODUCTION_CACHE, key, (train, fitted), PRODUCTION_CACHE_LIMIT)

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
            "production_cache_hit": bool(cache_hit),
        }
    )
    return score


def safe_subset_events(events: Sequence[Any], target: int) -> list[Any]:
    """Take complete train/test lines while preserving source sequence order."""
    if target >= len(events):
        return list(events)
    if target <= 0:
        return []

    indexed = list(enumerate(events))
    train = [(index, event) for index, event in indexed if not event.test]
    test = [(index, event) for index, event in indexed if event.test]
    n_test = min(len(test), max(1, int(round(target * 0.2))))
    n_train = min(len(train), max(0, target - n_test))

    def take_complete_lines(rows: list[tuple[int, Any]], limit: int) -> list[int]:
        groups: list[list[tuple[int, Any]]] = []
        current: list[tuple[int, Any]] = []
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

        selected: list[int] = []
        for line in groups:
            if selected and len(selected) + len(line) > limit:
                break
            selected.extend(index for index, _ in line)
            if len(selected) >= limit:
                break
        return selected

    selected_indices = set(take_complete_lines(train, n_train))
    selected_indices.update(take_complete_lines(test, n_test))
    return [event for index, event in indexed if index in selected_indices]


def audited_fit_candidate(*args, **kwargs):
    fitted = _ORIGINAL_FIT_CANDIDATE(*args, **kwargs)
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


def audited_score_trial(*args, **kwargs):
    _ACTIVE_AUDIT.clear()
    result = _ORIGINAL_SCORE_TRIAL(*args, **kwargs)
    difference = float(
        result["differences_bits"]["heldout_predictive_cipher_minus_production"]
    )
    production_test = _ACTIVE_AUDIT.get("production_test_bits")
    if production_test is not None:
        _ACTIVE_AUDIT["cipher_test_bits"] = float(production_test + difference)
    _ACTIVE_AUDIT["heldout_cipher_minus_production_bits"] = difference
    _ACTIVE_AUDIT["heldout_cipher_minus_production_bits_per_token"] = (
        difference / max(1, int(result["n_test"]))
    )
    _ACTIVE_AUDIT["strict_heldout_advantage"] = bool(difference < 0.0)
    _ACTIVE_AUDIT["strict_cipher_selected"] = bool(
        result["cipher_selected"] and difference < 0.0
    )
    _ACTIVE_AUDIT["legacy_solver_label"] = result.get("solver")
    _ACTIVE_AUDIT["scientific_solver_label"] = (
        "parallel_tempering_best_state_optimizer"
        if result.get("solver") == "bayes"
        else result.get("solver")
    )
    result["remediation_audit"] = dict(_ACTIVE_AUDIT)
    return result


def install() -> None:
    kt.prepare = safe_prepare
    production.rich_production_predictive_nll = safe_production_predictive_nll
    base.label_events = safe_label_events
    base.subset_events = safe_subset_events
    base.fit_candidate = audited_fit_candidate
    base.score_trial_v03 = audited_score_trial
    production.install()


def main() -> None:
    install()
    base.main()


if __name__ == "__main__":
    main()
