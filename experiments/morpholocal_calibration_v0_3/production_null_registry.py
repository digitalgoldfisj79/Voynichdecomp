#!/usr/bin/env python3
"""Charged production-null registry for morpholocal calibration v0.3.

Development-only wrapper.  It replaces the asymmetric v0.2 production
predictor with a frozen registry of four non-payload surface models while
leaving the synthetic generator, cipher decoders, accounting code and trial
seeds unchanged.

The selected null is chosen on training data only and pays log2(K) bits for
its registry index.  Test probabilities are frozen from training counts.
Formal use requires a later static effective-source freeze.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import tournament_kt as kt

base = kt.base
ALPHA = 0.5
LN2 = math.log(2.0)
N_CELLS = 24
MODEL_NAMES = (
    "context_iid",
    "cell_markov",
    "context_cell_markov",
    "repeat_context",
)


@dataclass(frozen=True)
class FittedNull:
    name: str
    train_bits: float
    payload: Any


_CACHE: dict[tuple[int, int], FittedNull] = {}


def _lines(events: Sequence[Any]) -> list[list[Any]]:
    output: list[list[Any]] = []
    current: list[Any] = []
    previous = None
    for event in events:
        marker = (event.doc, event.line)
        if previous is not None and marker != previous:
            output.append(current)
            current = []
        current.append(event)
        previous = marker
    if current:
        output.append(current)
    return output


def _kt_row_bits(counts: Sequence[int], support: int = N_CELLS) -> float:
    total = int(sum(counts))
    if total == 0:
        return 0.0
    value = (math.lgamma(support * ALPHA + total) - math.lgamma(support * ALPHA)) / LN2
    value -= sum(
        (math.lgamma(ALPHA + int(count)) - math.lgamma(ALPHA)) / LN2
        for count in counts if count
    )
    return float(value)


def _beta_bits(successes: int, failures: int) -> float:
    return -(
        math.lgamma(successes + ALPHA)
        + math.lgamma(failures + ALPHA)
        - math.lgamma(successes + failures + 2 * ALPHA)
        - 2 * math.lgamma(ALPHA)
        + math.lgamma(2 * ALPHA)
    ) / LN2


def _fit_rows(events: Sequence[Any], state_fn) -> tuple[float, dict[Hashable, tuple[int, ...]]]:
    rows: dict[Hashable, list[int]] = defaultdict(lambda: [0] * N_CELLS)
    for line in _lines(events):
        previous_cell: int | None = None
        for event in line:
            state = state_fn(event, previous_cell)
            rows[state][int(event.cell)] += 1
            previous_cell = int(event.cell)
    frozen = {state: tuple(counts) for state, counts in rows.items()}
    return float(sum(_kt_row_bits(counts) for counts in frozen.values())), frozen


def _score_rows(events: Sequence[Any], rows: dict[Hashable, tuple[int, ...]], state_fn) -> float:
    bits = 0.0
    for line in _lines(events):
        previous_cell: int | None = None
        for event in line:
            state = state_fn(event, previous_cell)
            counts = rows.get(state, (0,) * N_CELLS)
            total = sum(counts)
            probability = (counts[int(event.cell)] + ALPHA) / (total + ALPHA * N_CELLS)
            bits -= math.log2(max(1e-300, probability))
            previous_cell = int(event.cell)
    return float(bits)


def _context_state(event, previous_cell):
    return (str(event.section), str(event.position))


def _markov_state(event, previous_cell):
    return -1 if previous_cell is None else int(previous_cell)


def _context_markov_state(event, previous_cell):
    return (str(event.section), str(event.position), -1 if previous_cell is None else int(previous_cell))


def _fit_repeat_context(events: Sequence[Any]):
    successes = 0
    failures = 0
    fallback: dict[Hashable, list[int]] = defaultdict(lambda: [0] * N_CELLS)
    for line in _lines(events):
        previous_cell: int | None = None
        for event in line:
            cell = int(event.cell)
            repeated = previous_cell is not None and cell == previous_cell
            if previous_cell is not None:
                if repeated:
                    successes += 1
                else:
                    failures += 1
            if not repeated:
                fallback[_context_state(event, previous_cell)][cell] += 1
            previous_cell = cell
    frozen = {state: tuple(counts) for state, counts in fallback.items()}
    train_bits = _beta_bits(successes, failures) + sum(_kt_row_bits(row) for row in frozen.values())
    return float(train_bits), (successes, failures, frozen)


def _score_repeat_context(events: Sequence[Any], payload) -> float:
    successes, failures, fallback = payload
    repeat_probability = (successes + ALPHA) / (successes + failures + 2 * ALPHA)
    bits = 0.0
    for line in _lines(events):
        previous_cell: int | None = None
        for event in line:
            cell = int(event.cell)
            state = _context_state(event, previous_cell)
            counts = fallback.get(state, (0,) * N_CELLS)
            total = sum(counts)
            q = (counts[cell] + ALPHA) / (total + ALPHA * N_CELLS)
            if previous_cell is None:
                probability = q
            elif cell == previous_cell:
                probability = repeat_probability + (1.0 - repeat_probability) * q
            else:
                probability = (1.0 - repeat_probability) * q
            bits -= math.log2(max(1e-300, probability))
            previous_cell = cell
    return float(bits)


def fit_registry(train: Sequence[Any]) -> FittedNull:
    candidates: list[FittedNull] = []
    for name, state_fn in (
        ("context_iid", _context_state),
        ("cell_markov", _markov_state),
        ("context_cell_markov", _context_markov_state),
    ):
        train_bits, rows = _fit_rows(train, state_fn)
        candidates.append(FittedNull(name, train_bits, rows))
    train_bits, payload = _fit_repeat_context(train)
    candidates.append(FittedNull("repeat_context", train_bits, payload))
    index_bits = math.log2(len(candidates))
    return min(candidates, key=lambda row: (row.train_bits + index_bits, row.name))


def score_fitted(data: Sequence[Any], fitted: FittedNull) -> float:
    if fitted.name == "context_iid":
        return _score_rows(data, fitted.payload, _context_state)
    if fitted.name == "cell_markov":
        return _score_rows(data, fitted.payload, _markov_state)
    if fitted.name == "context_cell_markov":
        return _score_rows(data, fitted.payload, _context_markov_state)
    if fitted.name == "repeat_context":
        return _score_repeat_context(data, fitted.payload)
    raise ValueError(fitted.name)


def rich_production_predictive_nll(data, train, registry, selector):
    key = (id(train), len(train))
    fitted = _CACHE.get(key)
    if fitted is None:
        fitted = fit_registry(train)
        _CACHE[key] = fitted
    if data is train:
        return float(fitted.train_bits + math.log2(len(MODEL_NAMES)))
    return score_fitted(data, fitted)


def install() -> None:
    original_load = base.load_v02

    def load_v02(repo: Path):
        gr, module = original_load(repo)
        module.production_predictive_nll = rich_production_predictive_nll
        return gr, module

    base.load_v02 = load_v02
    original_context = base.mp.get_context
    base.mp.get_context = lambda _method=None: original_context("fork")


def main() -> None:
    install()
    base.main()


if __name__ == "__main__":
    main()
