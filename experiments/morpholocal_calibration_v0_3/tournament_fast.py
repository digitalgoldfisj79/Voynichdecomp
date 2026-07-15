#!/usr/bin/env python3
"""Exact vectorised policy-likelihood accelerator for the v0.3 tournament."""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "tournament_runner.py"
spec = importlib.util.spec_from_file_location("v03_tournament_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import tournament_runner.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


@dataclass(frozen=True)
class PolicyData:
    cells: np.ndarray
    context_index: np.ndarray
    line_start: np.ndarray
    weights: np.ndarray


_LABEL_EVENTS_CACHE = {}
_POLICY_DATA_CACHE = {}


def cached_label_events(module, train, scheme, label):
    key = (id(train), scheme, label)
    rows = _LABEL_EVENTS_CACHE.get(key)
    if rows is None:
        rows = [e for e in train if module.key_label(e, scheme) == label]
        _LABEL_EVENTS_CACHE[key] = rows
    return rows


def ordered_subset(events, target):
    """Take complete train/test lines without reordering events inside lines."""
    if target >= len(events):
        return list(events)
    n_test = max(1, int(round(target * 0.2)))
    n_train = target - n_test

    def take(rows, limit):
        out = []
        current = []
        previous = None
        for event in rows:
            marker = (event.doc, event.line)
            if previous is not None and marker != previous:
                if out and len(out) + len(current) > limit:
                    break
                out.extend(current)
                if len(out) >= limit:
                    break
                current = []
            current.append(event)
            previous = marker
        if current and len(out) < limit and (not out or len(out) + len(current) <= limit):
            out.extend(current)
        return out

    train = take([e for e in events if not e.test], n_train)
    test = take([e for e in events if e.test], n_test)
    # The generator places complete training documents before complete test
    # documents; concatenation therefore preserves the original event order.
    return train + test


def prepare(module, events, registry):
    key = id(events)
    cached = _POLICY_DATA_CACHE.get(key)
    if cached is not None:
        return cached
    contexts = sorted({(e.section, e.position) for e in events})
    context_lookup = {value: i for i, value in enumerate(contexts)}
    cells = np.fromiter((int(e.cell) for e in events), dtype=np.int64, count=len(events))
    context_index = np.fromiter(
        (context_lookup[(e.section, e.position)] for e in events),
        dtype=np.int64,
        count=len(events),
    )
    line_start = np.zeros(len(events), dtype=bool)
    previous = None
    for i, e in enumerate(events):
        marker = (e.doc, e.line)
        line_start[i] = marker != previous
        previous = marker
    weights = np.empty((len(contexts), len(registry.cells)), dtype=np.float64)
    for ci, (section, position) in enumerate(contexts):
        for cell in range(len(registry.cells)):
            weights[ci, cell] = max(
                1e-300,
                float(module.context_weight(registry, cell, section, position)),
            )
    data = PolicyData(cells=cells, context_index=context_index, line_start=line_start, weights=weights)
    _POLICY_DATA_CACHE[key] = data
    return data


def fast_label_policy_nll(module, events, mapping, policy, registry, label):
    data = prepare(module, events, registry)
    mapping_array = np.asarray(mapping, dtype=np.int64)
    cells = data.cells
    units = mapping_array[cells]
    n_units = int(mapping_array.max()) + 1
    class_sizes = np.bincount(mapping_array, minlength=n_units).astype(np.float64)

    if policy == "iid_uniform":
        return float(np.log2(class_sizes[units]).sum())

    norm = np.zeros((data.weights.shape[0], n_units), dtype=np.float64)
    for cell, unit in enumerate(mapping_array):
        norm[:, unit] += data.weights[:, cell]
    base_prob = data.weights[data.context_index, cells] / np.clip(
        norm[data.context_index, units], 1e-300, None
    )

    if policy == "frequency_weighted":
        return float((-np.log2(np.clip(base_prob, 1e-300, None))).sum())

    if policy == "sticky_line_reset":
        probability = base_prob.copy()
        if len(cells) > 1:
            comparable = ~data.line_start[1:]
            same_unit = comparable & (units[1:] == units[:-1])
            probability[1:][same_unit] = (
                0.25 * base_prob[1:][same_unit]
                + 0.75 * (cells[1:][same_unit] == cells[:-1][same_unit])
            )
        return float((-np.log2(np.clip(probability, 1e-300, None))).sum())

    if policy == "cyclic":
        eps = 1e-9
        total = 0.0
        for unit in range(n_units):
            candidate_cells = np.flatnonzero(mapping_array == unit)
            selected = cells[units == unit]
            if not len(selected):
                continue
            expected = np.resize(candidate_cells, len(selected))
            mismatch = int(np.count_nonzero(selected != expected))
            match = len(selected) - mismatch
            total += match * -math.log2(1.0 - eps)
            if mismatch:
                total += mismatch * -math.log2(eps / max(1, len(candidate_cells) - 1))
        return float(total)

    raise ValueError(policy)


base.label_events = cached_label_events
base.label_policy_nll = fast_label_policy_nll
base.subset_events = ordered_subset

if __name__ == "__main__":
    base.main()
