#!/usr/bin/env python3
"""KT/universal policy models for morpholocal calibration v0.3 development."""
from __future__ import annotations

import importlib.util
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAST_PATH = HERE / "tournament_fast.py"
spec = importlib.util.spec_from_file_location("v03_fast_kt", FAST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import tournament_fast.py")
fast = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fast
spec.loader.exec_module(fast)
base = fast.base

ALPHA = 0.5
LN2 = math.log(2.0)


@dataclass(frozen=True)
class SequenceData:
    cells: np.ndarray
    contexts: np.ndarray
    line_start: np.ndarray
    test: np.ndarray
    n_contexts: int


_CACHE = {}


def prepare(events):
    key = id(events)
    if key in _CACHE:
        return _CACHE[key]
    context_values = sorted({(e.section, e.position) for e in events})
    context_index = {x: i for i, x in enumerate(context_values)}
    cells = np.asarray([int(e.cell) for e in events], dtype=np.int64)
    contexts = np.asarray([context_index[(e.section, e.position)] for e in events], dtype=np.int64)
    test = np.asarray([bool(e.test) for e in events], dtype=bool)
    line_start = np.zeros(len(events), dtype=bool)
    previous = None
    for i, e in enumerate(events):
        marker = (e.doc, e.line)
        line_start[i] = marker != previous
        previous = marker
    value = SequenceData(cells, contexts, line_start, test, len(context_values))
    _CACHE[key] = value
    return value


def kt_multinomial_bits(counts: np.ndarray, support_sizes: np.ndarray) -> float:
    bits = 0.0
    for row, support in zip(counts, support_sizes):
        n = int(row.sum())
        if n == 0:
            continue
        k = int(support)
        bits += (math.lgamma(k * ALPHA + n) - math.lgamma(k * ALPHA)) / LN2
        nonzero = row[row > 0]
        bits -= sum(
            (math.lgamma(ALPHA + int(c)) - math.lgamma(ALPHA)) / LN2
            for c in nonzero
        )
    return bits


def frequency_counts(data: SequenceData, mapping: np.ndarray, mask: np.ndarray):
    units = mapping[data.cells]
    n_units = int(mapping.max()) + 1
    groups = units * data.n_contexts + data.contexts
    n_groups = n_units * data.n_contexts
    n_cells = len(mapping)
    flat = groups[mask] * n_cells + data.cells[mask]
    counts = np.bincount(flat, minlength=n_groups * n_cells).reshape(n_groups, n_cells)
    class_sizes = np.bincount(mapping, minlength=n_units)
    support = np.repeat(class_sizes, data.n_contexts)
    return counts, groups, units, support


def frequency_train_test(data: SequenceData, mapping: np.ndarray, use_mask_train=None, use_mask_test=None):
    train_mask = ~data.test if use_mask_train is None else ((~data.test) & use_mask_train)
    test_mask = data.test if use_mask_test is None else (data.test & use_mask_test)
    counts, groups, units, support = frequency_counts(data, mapping, train_mask)
    train_bits = kt_multinomial_bits(counts, support)
    totals = counts.sum(axis=1)
    g = groups[test_mask]
    c = data.cells[test_mask]
    denom = totals[g] + ALPHA * support[g]
    numer = counts[g, c] + ALPHA
    test_bits = float((
        -np.log2(np.clip(numer / np.clip(denom, 1e-300, None), 1e-300, None))
    ).sum())
    return train_bits, test_bits


def iid_train_test(data: SequenceData, mapping: np.ndarray):
    units = mapping[data.cells]
    sizes = np.bincount(mapping, minlength=int(mapping.max()) + 1)
    bits = np.log2(sizes[units])
    return float(bits[~data.test].sum()), float(bits[data.test].sum())


def cyclic_indicators(data: SequenceData, mapping: np.ndarray):
    units = mapping[data.cells]
    expected = np.empty(len(data.cells), dtype=np.int64)
    state = defaultdict(int)
    candidates = {
        u: np.flatnonzero(mapping == u) for u in range(int(mapping.max()) + 1)
    }
    for i, unit in enumerate(units):
        cand = candidates[int(unit)]
        expected[i] = cand[state[int(unit)] % len(cand)]
        state[int(unit)] += 1
    return units, expected, data.cells == expected


def beta_bernoulli_bits(successes: int, failures: int):
    return -(
        math.lgamma(successes + ALPHA) + math.lgamma(failures + ALPHA)
        - math.lgamma(successes + failures + 2 * ALPHA)
        - 2 * math.lgamma(ALPHA) + math.lgamma(2 * ALPHA)
    ) / LN2


def cyclic_train_test(data: SequenceData, mapping: np.ndarray):
    units, expected, correct = cyclic_indicators(data, mapping)
    train = ~data.test
    test = data.test
    successes = int(correct[train].sum())
    failures = int(train.sum() - successes)
    train_bits = beta_bernoulli_bits(successes, failures)
    p_correct = (successes + ALPHA) / (successes + failures + 2 * ALPHA)
    sizes = np.bincount(mapping, minlength=int(mapping.max()) + 1)
    probability = np.where(
        correct[test], p_correct,
        (1 - p_correct) / np.maximum(1, sizes[units[test]] - 1),
    )
    test_bits = float((-np.log2(np.clip(probability, 1e-300, None))).sum())
    return train_bits, test_bits


def sticky_train_test(data: SequenceData, mapping: np.ndarray):
    units = mapping[data.cells]
    eligible = np.zeros(len(data.cells), dtype=bool)
    if len(data.cells) > 1:
        eligible[1:] = (~data.line_start[1:]) & (units[1:] == units[:-1])
    sticky = np.zeros(len(data.cells), dtype=bool)
    if len(data.cells) > 1:
        sticky[1:] = eligible[1:] & (data.cells[1:] == data.cells[:-1])
    train = ~data.test
    test = data.test
    successes = int((sticky & train).sum())
    failures = int((eligible & ~sticky & train).sum())
    persistence_bits = beta_bernoulli_bits(successes, failures)
    persistence = (successes + ALPHA) / (successes + failures + 2 * ALPHA)
    fallback_train = train & ~sticky
    frequency_bits, _ = frequency_train_test(
        data, mapping, fallback_train, np.zeros(len(test), dtype=bool)
    )
    counts, groups, _, support = frequency_counts(data, mapping, fallback_train)
    totals = counts.sum(axis=1)
    g = groups[test]
    c = data.cells[test]
    q = (counts[g, c] + ALPHA) / np.clip(
        totals[g] + ALPHA * support[g], 1e-300, None
    )
    previous_cells = np.roll(data.cells, 1)
    sticky_possible = eligible[test] & (data.cells[test] == previous_cells[test])
    probability = (1 - persistence) * q + persistence * sticky_possible.astype(float)
    test_bits = float((-np.log2(np.clip(probability, 1e-300, None))).sum())
    return persistence_bits + frequency_bits, test_bits


def policy_train_test(events, mapping, policy):
    data = prepare(events)
    array = np.asarray(mapping, dtype=np.int64)
    if policy == "iid_uniform":
        return iid_train_test(data, array)
    if policy == "frequency_weighted":
        return frequency_train_test(data, array)
    if policy == "cyclic":
        return cyclic_train_test(data, array)
    if policy == "sticky_line_reset":
        return sticky_train_test(data, array)
    raise ValueError(policy)


def label_policy_nll(module, events, mapping, policy, registry, label):
    train, _ = policy_train_test(events, mapping, policy)
    return float(train)


def policy_nll(module, events, assignments, scheme, policy, registry, *, return_by_split=False):
    train_total = 0.0
    test_total = 0.0
    for label in sorted(assignments):
        rows = [e for e in events if module.key_label(e, scheme) == label]
        train, test = policy_train_test(rows, assignments[label], policy)
        train_total += train
        test_total += test
    return (train_total, test_total) if return_by_split else train_total + test_total


base.label_policy_nll = label_policy_nll
base.policy_nll = policy_nll

if __name__ == "__main__":
    base.main()
