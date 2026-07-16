#!/usr/bin/env python3
"""Execution-only minimal smoke wrapper for the final Family P coordinate solver."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.argv = [
    "v060_family_p_coordinate_final.py",
    "--repo", str(repo),
    "--output", "/tmp/p2-smoke.json",
    "--iso", "en",
    "--split", "dev",
    "--length", "96",
    "--replicates", "1",
    "--seed-count", "2",
    "--screen-cycles", "1",
    "--screen-mono-iterations", "100",
    "--screen-mono-restarts", "1",
    "--screen-shift-iterations", "100",
    "--screen-shift-restarts", "1",
    "--top-refine", "1",
    "--refine-cycles", "1",
    "--refine-mono-iterations", "100",
    "--refine-mono-restarts", "1",
    "--refine-shift-iterations", "100",
    "--refine-shift-restarts", "1",
    "--final-mono-iterations", "100",
    "--final-mono-restarts", "1",
    "--final-shift-iterations", "100",
    "--final-shift-restarts", "1",
    "--workers", "2",
]
runpy.run_path(str(Path(__file__).with_name("v060_family_p_coordinate_final.py")), run_name="__main__")
