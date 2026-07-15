#!/usr/bin/env python3
"""Linux fork launcher for the development neural decoder evaluation.

The tournament's generic development runner uses spawn. The neural solver is
registered dynamically, so spawned workers would re-import the unmodified
solver registry. This launcher reconstructs the compatibility implementation
once in the parent, caches it read-only, and then forks workers. The formal
runner will use a static, patch-free decoder registry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import neural_runner


def repo_from_args(arguments: list[str]) -> Path:
    if "--repo" in arguments:
        index = arguments.index("--repo")
        if index + 1 >= len(arguments):
            raise SystemExit("--repo requires a path")
        return Path(arguments[index + 1]).resolve()
    return neural_runner.base.DEFAULT_REPO.resolve()


def main() -> None:
    arguments = sys.argv[1:]
    repo = repo_from_args(arguments)
    gpu_runner, module = neural_runner.base.load_v02(repo)
    neural_runner.base.load_v02 = lambda _repo: (gpu_runner, module)
    original = neural_runner.base.mp.get_context
    neural_runner.base.mp.get_context = lambda _method=None: original("fork")
    neural_runner.run_main(arguments)


if __name__ == "__main__":
    main()
