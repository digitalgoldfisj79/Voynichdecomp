#!/usr/bin/env python3
"""Deterministic regression tests for the v0.3.1 remediation wrapper."""
from __future__ import annotations

import json
from types import SimpleNamespace

import remediation_runtime as remediation


def event(doc, line, cell, position, token_index, test=False):
    return SimpleNamespace(
        doc=doc,
        line=line,
        cell=cell,
        section="HERBAL_A" if not test else "HERBAL_B",
        position=position,
        token_index=token_index,
        test=test,
    )


def line(doc, line_number, cells, token_indices, test=False):
    rows = []
    for index, (cell, token_index) in enumerate(zip(cells, token_indices)):
        position = "FIRST" if index == 0 else "LAST" if index == len(cells) - 1 else "MID"
        rows.append(event(doc, line_number, cell, position, token_index, test))
    return rows


def test_subset_preserves_order():
    events = []
    events += line("D1", 1, [0, 7, 8, 9], [50, 30, 10, 40], False)
    events += line("D1", 2, [1, 4, 5, 6], [80, 60, 20, 70], False)
    events += line("D2", 1, [2, 10, 11, 12], [90, 35, 15, 45], True)
    events += line("D2", 2, [3, 13, 14, 15], [95, 55, 25, 65], True)

    selected = remediation.safe_subset_events(events, 8)
    expected = events[:4] + events[8:12]
    assert selected == expected, (
        [row.cell for row in selected],
        [row.cell for row in expected],
    )
    assert [row.cell for row in selected[:4]] == [0, 7, 8, 9]


def test_prepare_cache_isolation():
    first = line("A", 1, [0, 0, 0], [1, 2, 3], False)
    second = line("B", 1, [1, 1, 1], [1, 2, 3], False)
    first_data = remediation.safe_prepare(first)
    second_data = remediation.safe_prepare(second)
    assert first_data.cells.tolist() == [0, 0, 0]
    assert second_data.cells.tolist() == [1, 1, 1]
    assert remediation.safe_prepare(first) is first_data
    assert remediation.safe_prepare(second) is second_data


def test_production_cache_isolation():
    train_zero = line("A", 1, [0, 0, 0, 0], [1, 2, 3, 4], False)
    train_one = line("B", 1, [1, 1, 1, 1], [1, 2, 3, 4], False)
    test_zero = line("T", 1, [0, 0], [1, 2], True)
    zero_score = remediation.safe_production_predictive_nll(test_zero, train_zero, None, None)
    one_score = remediation.safe_production_predictive_nll(test_zero, train_one, None, None)
    assert zero_score < one_score, (zero_score, one_score)


def test_cache_bounds():
    for index in range(remediation.PREPARE_CACHE_LIMIT + 10):
        rows = line(f"P{index}", 1, [index % 24], [index], False)
        remediation.safe_prepare(rows)
    assert len(remediation._PREPARE_CACHE) <= remediation.PREPARE_CACHE_LIMIT

    for index in range(remediation.PRODUCTION_CACHE_LIMIT + 10):
        rows = line(f"R{index}", 1, [index % 24, index % 24], [1, 2], False)
        remediation.safe_production_predictive_nll(rows, rows, None, None)
    assert len(remediation._PRODUCTION_CACHE) <= remediation.PRODUCTION_CACHE_LIMIT


def main():
    tests = [
        test_subset_preserves_order,
        test_prepare_cache_isolation,
        test_production_cache_isolation,
        test_cache_bounds,
    ]
    completed = []
    for test in tests:
        test()
        completed.append(test.__name__)
    print(json.dumps({"status": "PASS", "tests": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
