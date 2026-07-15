#!/usr/bin/env python3
"""Development-only correctness wrapper for morpholocal calibration v0.3.1.

This wrapper fixes three defects without changing the frozen v0.2 generator:

1. object-id caches now retain the exact source object in a bounded LRU, so a
   recycled Python id cannot return data fitted to another trial;
2. length subsetting preserves the original event order exactly and never
   sorts by ``token_index`` (a vocabulary index, not a sequence position);
3. the charged production-null registry remains fitted on training data only.

The wrapper is for remediation/audit runs. Formal use still requires a static,
patch-free effective-source freeze after the corrected behaviour is validated.
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
_PREPARE_CACHE: OrderedDict[int, tuple[object, kt.SequenceData]] = OrderedDict()
_PRODUCTION_CACHE: OrderedDict[int, tuple[object, production.FittedNull]] = OrderedDict()


def _bounded_store(cache: OrderedDict, key: int, value: tuple[object, Any], limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def safe_prepare(events: Sequence[Any]) -> kt.SequenceData:
    """Prepare policy arrays with an identity-safe bounded cache."""
    key = id(events)
    cached = _PREPARE_CACHE.get(key)
    if cached is not None:
        source, value = cached
        if source is events:
            _PREPARE_CACHE.move_to_end(key)
            return value
        # Defensive only: retaining ``source`` should prevent id reuse while
        # the entry exists, but never return a mismatched cached value.
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
    if cached is not None:
        source, candidate = cached
        if source is train:
            _PRODUCTION_CACHE.move_to_end(key)
            fitted = candidate
        else:
            del _PRODUCTION_CACHE[key]

    if fitted is None:
        fitted = production.fit_registry(train)
        _bounded_store(
            _PRODUCTION_CACHE,
            key,
            (train, fitted),
            PRODUCTION_CACHE_LIMIT,
        )

    if data is train:
        return float(fitted.train_bits + math.log2(len(production.MODEL_NAMES)))
    return production.score_fitted(data, fitted)


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
        for row in rows:
            index, event = row
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


def install() -> None:
    kt.prepare = safe_prepare
    production.rich_production_predictive_nll = safe_production_predictive_nll
    base.subset_events = safe_subset_events
    production.install()


def main() -> None:
    install()
    base.main()


if __name__ == "__main__":
    main()
