#!/usr/bin/env python3
"""Contract tests for the published TScore German-lute-tablature data model.

This is not a test of the unreleased official TScore implementation. It tests a
small, independently written ingestion contract derived from the public 2024
paper: duration semantics, carry semantics, and symbol-to-string/fret mapping.
The fixture is synthetic and contains no copied musical passage.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import unittest

DURATION_BASE = {
    "I": Fraction(1, 4),
    "T": Fraction(1, 8),
    "F": Fraction(1, 16),
    "E": Fraction(1, 32),
    ".": Fraction(1, 2),
    "..": Fraction(3, 4),
    "...": Fraction(1, 1),
}

# Synthetic table with the same documented row/column semantics but invented
# symbols, avoiding redistribution of a historical musical example.
SYMBOL_TABLE = (
    ("a", "b", "c", "d", "e", "f"),
    ("g", "h", "i", "j", "k", "l"),
    ("m", "n", "o", "p", "q", "r"),
    ("s", "t", "u", "v", "w", "x"),
    ("y", "z", "aa", "bb", "cc", "dd"),
)


@dataclass(frozen=True)
class Column:
    index: int
    source_duration: str
    duration: Fraction
    onset: Fraction
    sounds: tuple[tuple[str, int, int], ...]  # source, fret, string


def decode_duration(token: str, previous: Fraction | None, carry: bool) -> Fraction:
    if token == "-":
        if not carry or previous is None:
            raise ValueError("carry token requires duration inheritance and a previous duration")
        return previous
    cleaned = token.replace("_", "")
    dotted = cleaned.endswith(".") and cleaned not in {".", "..", "..."}
    if dotted:
        cleaned = cleaned[:-1]
    if cleaned not in DURATION_BASE:
        raise ValueError(f"unknown duration token: {token}")
    value = DURATION_BASE[cleaned]
    return value * Fraction(3, 2) if dotted else value


def symbol_map(table=SYMBOL_TABLE) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for string_index, row in enumerate(table):
        for fret_index, symbol in enumerate(row, start=1):
            if symbol in out:
                raise ValueError(f"duplicate tablature symbol: {symbol}")
            out[symbol] = (fret_index, string_index)
    return out


def ingest(duration_tokens: list[str], voices: list[list[str]], *, carry: bool) -> list[Column]:
    if not duration_tokens:
        return []
    if any(len(v) != len(duration_tokens) for v in voices):
        raise ValueError("all VOX rows must align with the T row")
    mapping = symbol_map()
    previous = None
    onset = Fraction(0)
    columns: list[Column] = []
    for idx, source_duration in enumerate(duration_tokens):
        duration = decode_duration(source_duration, previous, carry)
        sounds = []
        for voice in voices:
            symbol = voice[idx]
            if symbol == ".":
                continue
            if symbol not in mapping:
                raise ValueError(f"unknown tablature symbol: {symbol}")
            fret, string = mapping[symbol]
            sounds.append((symbol, fret, string))
        columns.append(Column(idx, source_duration, duration, onset, tuple(sounds)))
        onset += duration
        previous = duration
    return columns


class TScoreDocumentedContractTests(unittest.TestCase):
    def test_duration_and_beam_semantics(self):
        tokens = ["I", "T", "F", "E", "E.", "E_", "_E", ".", "..", "..."]
        got = [decode_duration(t, None, False) for t in tokens]
        expected = [
            Fraction(1, 4), Fraction(1, 8), Fraction(1, 16), Fraction(1, 32),
            Fraction(3, 64), Fraction(1, 32), Fraction(1, 32),
            Fraction(1, 2), Fraction(3, 4), Fraction(1, 1),
        ]
        self.assertEqual(got, expected)

    def test_carry_semantics(self):
        self.assertEqual(decode_duration("-", Fraction(1, 8), True), Fraction(1, 8))
        with self.assertRaises(ValueError):
            decode_duration("-", Fraction(1, 8), False)

    def test_symbol_mapping_coordinates(self):
        mapping = symbol_map()
        self.assertEqual(mapping["a"], (1, 0))
        self.assertEqual(mapping["f"], (6, 0))
        self.assertEqual(mapping["y"], (1, 4))
        self.assertEqual(mapping["dd"], (6, 4))

    def test_aligned_ingestion_and_onsets(self):
        durations = ["I", "T", "-", "E."]
        voices = [
            ["a", "h", ".", "dd"],
            ["y", ".", "c", "m"],
        ]
        cols = ingest(durations, voices, carry=True)
        self.assertEqual([c.duration for c in cols], [
            Fraction(1, 4), Fraction(1, 8), Fraction(1, 8), Fraction(3, 64)
        ])
        self.assertEqual([c.onset for c in cols], [
            Fraction(0), Fraction(1, 4), Fraction(3, 8), Fraction(1, 2)
        ])
        self.assertEqual(cols[0].sounds, (("a", 1, 0), ("y", 1, 4)))
        self.assertEqual(cols[2].sounds, (("c", 3, 0),))

    def test_misaligned_voice_rejected(self):
        with self.assertRaises(ValueError):
            ingest(["I", "T"], [["a"]], carry=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
