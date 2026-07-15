#!/usr/bin/env python3
"""Linux fork launcher for the development neural decoder evaluation.

The tournament's generic development runner uses spawn. The neural solver is
registered dynamically, so spawned workers would re-import the unmodified
solver registry. Fork is used here only for development evaluation so workers
inherit the frozen checkpoint and registered neural solver. The formal runner
will use a static, patch-free decoder registry.
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import neural_runner


def main() -> None:
    original = neural_runner.base.mp.get_context
    neural_runner.base.mp.get_context = lambda _method=None: original("fork")
    neural_runner.run_main(sys.argv[1:])


if __name__ == "__main__":
    main()
