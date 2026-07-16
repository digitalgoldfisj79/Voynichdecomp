#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v070_source_transfer_mdl as programme

_original_context = programme.mp.get_context
_original_build = programme.build_source_registry


def corrected_get_assets(repo: Path):
    cached = programme._PROCESS_CACHE.get("assets")
    if cached is not None:
        return cached
    gr, module, registry, old_external = programme.gen.load_assets(repo)
    raw_external, source_meta = _original_build(module, repo)
    source_external = type(old_external)(
        transition=raw_external["transition"],
        stationary=raw_external["stationary"],
        source_hash=raw_external["source_hash"],
    )
    cached = (gr, module, registry, source_external, source_meta)
    programme._PROCESS_CACHE["assets"] = cached
    return cached


programme.get_assets = corrected_get_assets
programme.mp.get_context = lambda _method=None: _original_context("fork")

if __name__ == "__main__":
    programme.main()
