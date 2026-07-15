#!/usr/bin/env python3
"""Exact accelerated launcher for the v0.5.0 channel oracle."""
from __future__ import annotations

import runpy

from rapidfuzz.distance import Levenshtein

import recoverability_v050 as core


def fast_edit_distance(a, b):
    return int(Levenshtein.distance(a, b))


core.edit_distance = fast_edit_distance

if __name__ == "__main__":
    runpy.run_module("recoverability_v050", run_name="__main__")
