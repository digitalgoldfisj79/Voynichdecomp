#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from compression_metrics import conditional_bits_per_byte, directional_excess_bits_per_byte, normalized_compression_distance
from representations import encode_representation


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = encode_representation("abracadabra " * 1000, "surface_utf8")
        self.b = encode_representation("xyzzy plugh " * 1000, "surface_utf8")
        self.probe_a = encode_representation("abracadabra " * 100, "surface_utf8")

    def test_same_source_conditional_is_better(self) -> None:
        for compressor in ("zlib9", "bz2_9", "lzma9e"):
            self.assertLess(conditional_bits_per_byte(self.a, self.probe_a, compressor), conditional_bits_per_byte(self.b, self.probe_a, compressor))

    def test_directional_excess_positive_for_wrong_source(self) -> None:
        for compressor in ("zlib9", "bz2_9", "lzma9e"):
            self.assertGreater(directional_excess_bits_per_byte(self.b, self.a, self.probe_a, compressor), 0)

    def test_symmetric_ncd_is_mean_of_orders(self) -> None:
        for compressor in ("zlib9", "bz2_9", "lzma9e"):
            forward, reverse, symmetric = normalized_compression_distance(self.a, self.b, compressor)
            self.assertAlmostEqual(symmetric, (forward + reverse) / 2.0, places=12)
            self.assertGreaterEqual(forward, 0)
            self.assertGreaterEqual(reverse, 0)


if __name__ == "__main__":
    unittest.main()
