#!/usr/bin/env python3
"""Parser-only correction wrapper for VOYNICH-LEGACY-CHAIN-REPLAY-v0.1.

The frozen scientific protocol, target, metrics, legacy arms, parameters and
adjudication are unchanged.  This wrapper replaces only load_timm_file so that
the pinned generator's leading #Properties/#Statistics metadata block is not
mistaken for generated text.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
BASE_RUNNER = HERE / 'run_legacy_chain_replay.py'

spec = importlib.util.spec_from_file_location('legacy_chain_base', BASE_RUNNER)
base = importlib.util.module_from_spec(spec)
sys.modules['legacy_chain_base'] = base
assert spec.loader is not None
spec.loader.exec_module(base)


def load_timm_file_corrected(path):
    lines = []
    for raw in Path(path).read_text(errors='ignore').splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        toks = base.clean_words(raw)
        if toks:
            lines.append(toks)
    # Pinned Timm configuration: text.lines_to_create=1200.  Fail closed on
    # format drift rather than silently admitting metadata or dropping text.
    if len(lines) != 1200:
        raise ValueError(f'Timm generated-line count {len(lines)} != 1200 for {path}')
    if sum(map(len, lines)) < 5000:
        raise ValueError(f'Timm parse too small {path}')
    return lines


base.load_timm_file = load_timm_file_corrected

if __name__ == '__main__':
    base.main()
