#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import remediation_runtime_v033 as runtime


def event(doc, line, cell):
    return SimpleNamespace(doc=doc, line=line, cell=cell)


def test_transition_bits() -> None:
    log_t = np.log2(np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float))
    bits, count = runtime._transition_bits([[0, 0, 1], [1, 1]], log_t)
    expected = -np.log2(0.9) - np.log2(0.1) - np.log2(0.8)
    assert abs(bits - expected) < 1e-12
    assert count == 3


def test_latent_lines_preserve_order() -> None:
    module = SimpleNamespace()
    original = runtime.base.mapping_unit
    try:
        runtime.base.mapping_unit = lambda module, row, assignments, scheme: assignments["GLOBAL"][row.cell]
        rows = [event(0, 0, 0), event(0, 0, 1), event(0, 1, 1), event(0, 1, 0)]
        lines = runtime._latent_lines(module, rows, {"GLOBAL": (1, 0)}, "global")
        assert lines == [[1, 0], [0, 1]]
    finally:
        runtime.base.mapping_unit = original


def test_deterministic_randomization() -> None:
    original = runtime._latent_lines
    try:
        runtime._latent_lines = lambda *args, **kwargs: [[0, 1, 0, 1, 1], [1, 0, 1, 0]]
        transition = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=float)
        fitted = {"assignments": {}, "scheme": "global"}
        left = runtime.sequence_randomization_audit(None, [], fitted, transition, 1234, 31)
        right = runtime.sequence_randomization_audit(None, [], fitted, transition, 1234, 31)
        assert left == right
        assert left["randomizations"] == 31
        assert left["transitions"] == 7
        assert 1 / 32 <= left["p_value"] <= 1.0
    finally:
        runtime._latent_lines = original


def test_ordered_sequence_beats_randomization() -> None:
    original = runtime._latent_lines
    try:
        # Alternation is strongly preferred by this transition matrix.
        runtime._latent_lines = lambda *args, **kwargs: [[0, 1] * 20]
        transition = np.asarray([[0.01, 0.99], [0.99, 0.01]], dtype=float)
        fitted = {"assignments": {}, "scheme": "global"}
        audit = runtime.sequence_randomization_audit(None, [], fitted, transition, 9876, 199)
        assert audit["advantage_bits"] > 0
        assert audit["p_value"] <= 0.05
        assert audit["pass_0_05"]
    finally:
        runtime._latent_lines = original


def main() -> None:
    tests = [
        test_transition_bits,
        test_latent_lines_preserve_order,
        test_deterministic_randomization,
        test_ordered_sequence_beats_randomization,
    ]
    for test in tests:
        test()
    print({"status": "PASS", "tests": [test.__name__ for test in tests]})


if __name__ == "__main__":
    main()
