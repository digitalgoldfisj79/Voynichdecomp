#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V05 = ROOT / "experiments" / "recoverability_frontier_v0_5"
sys.path.insert(0, str(V05))
import recoverability_v050 as core

languages = core.load_languages(
    V05 / "corpus_manifest_v050.json",
    ROOT / ".cache" / "v060-p-alphabet-preflight",
)
print("V060_P_LANGUAGE_ALPHABETS", json.dumps({
    iso: {"size": len(lang.alphabet), "alphabet": "".join(lang.alphabet)}
    for iso, lang in sorted(languages.items())
}, ensure_ascii=False, sort_keys=True), flush=True)
