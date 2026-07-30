#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from representations import chunk_text, encode_representation


class RepresentationTests(unittest.TestCase):
    def test_fixed_width_codepoints(self) -> None:
        data = encode_representation("a β", "codepoint_u32_ws")
        self.assertEqual(len(data), 3 * 4)

    def test_token_recurrence_is_label_invariant(self) -> None:
        self.assertEqual(
            encode_representation("foo bar foo baz", "token_recurrence_u32"),
            encode_representation("x y x z", "token_recurrence_u32"),
        )

    def test_chunking_deterministic(self) -> None:
        text = "abcdef" * 100
        self.assertEqual(chunk_text(text, "surface_utf8", 50, 25), chunk_text(text, "surface_utf8", 50, 25))


if __name__ == "__main__":
    unittest.main()
