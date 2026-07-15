#!/usr/bin/env python3
"""v0.3.2 runtime: v0.3.1 correctness fixes plus selector parity.

v0.3.1 replaced the production comparator with a registry of cell-sequence
models, but its replacement predictor ignored the ``selector`` argument.  The
cipher likelihood continued to pay token-selection entropy while production
paid only cell-sequence entropy.  This module restores exact likelihood parity:

    production predictive bits = cell-process bits + token-selector bits.

The v0.3.1 module and branch remain unchanged for provenance.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import production_null_registry as production
import remediation_runtime as v031

base = v031.base
ADJACENT_KERNEL = (16.0, 8.0, 4.0, 2.0, 1.0)


def selector_nll(events: Sequence[Any], registry: Any, selector: str) -> float:
    """Exact production token-selector likelihood used by the v0.2 generator."""
    if selector not in ("none", "adjacent_length"):
        raise ValueError(selector)
    total = 0.0
    for line in production._lines(events):
        previous_length = None
        for event in line:
            candidates = tuple(range(len(registry.token_names[event.cell])))
            weights: list[float] = []
            for index in candidates:
                weight = float(registry.token_weights[event.cell][index]) + 0.5
                if selector == "adjacent_length" and previous_length is not None:
                    difference = min(
                        abs(len(registry.token_names[event.cell][index]) - previous_length),
                        4,
                    )
                    weight *= ADJACENT_KERNEL[difference]
                weights.append(weight)
            probability = weights[int(event.token_index)] / sum(weights)
            total -= math.log2(max(probability, 1e-300))
            previous_length = int(event.length)
    return float(total)


def safe_production_predictive_nll(data, train, registry, selector) -> float:
    """Identity-safe production registry score with token-selector parity."""
    key = id(train)
    cached = v031._PRODUCTION_CACHE.get(key)
    fitted = None
    cache_hit = False
    if cached is not None:
        source, candidate = cached
        if source is train:
            v031._PRODUCTION_CACHE.move_to_end(key)
            fitted = candidate
            cache_hit = True
        else:
            del v031._PRODUCTION_CACHE[key]

    if fitted is None:
        fitted = production.fit_registry(train)
        v031._bounded_store(
            v031._PRODUCTION_CACHE,
            key,
            (train, fitted),
            v031.PRODUCTION_CACHE_LIMIT,
        )

    token_bits = selector_nll(data, registry, selector)
    if data is train:
        cell_bits = float(fitted.train_bits + math.log2(len(production.MODEL_NAMES)))
        v031._ACTIVE_AUDIT["production_train_cell_bits"] = cell_bits
        v031._ACTIVE_AUDIT["production_train_selector_bits"] = token_bits
        v031._ACTIVE_AUDIT["production_train_bits"] = cell_bits + token_bits
    else:
        cell_bits = float(production.score_fitted(data, fitted))
        v031._ACTIVE_AUDIT["production_test_cell_bits"] = cell_bits
        v031._ACTIVE_AUDIT["production_test_selector_bits"] = token_bits
        v031._ACTIVE_AUDIT["production_test_bits"] = cell_bits + token_bits

    v031._ACTIVE_AUDIT.update(
        {
            "production_model": fitted.name,
            "production_model_train_bits_without_index": float(fitted.train_bits),
            "production_registry_index_bits": float(math.log2(len(production.MODEL_NAMES))),
            "production_cache_hit": bool(cache_hit),
            "production_selector": str(selector),
            "selector_parity_enabled": True,
        }
    )
    return float(cell_bits + token_bits)


def install() -> None:
    """Install v0.3.1 fixes while replacing its production scorer by parity."""
    v031.safe_production_predictive_nll = safe_production_predictive_nll
    v031.install()


if __name__ == "__main__":
    install()
    base.main()
