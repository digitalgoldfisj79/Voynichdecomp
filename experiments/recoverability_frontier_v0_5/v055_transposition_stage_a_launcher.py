#!/usr/bin/env python3
"""Launch v0.5.5 Stage A with the mono search bound to the quadgram scorer."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v055_transposition_stage_a.py")
source = path.read_text(encoding="utf-8")
import_needle = (
    "from homophonic_confirm_v052_quadgram import build_quadgram_model\n"
)
import_replacement = (
    "from homophonic_confirm_v052_quadgram import "
    "build_quadgram_model, quadgram_score_key\n"
)
if source.count(import_needle) != 1:
    raise RuntimeError("quadgram import site mismatch")
patched = source.replace(import_needle, import_replacement)
main_needle = "def main() -> None:\n"
main_replacement = (
    "def main() -> None:\n"
    "    mono.score_key = quadgram_score_key\n"
)
if patched.count(main_needle) != 1:
    raise RuntimeError("main site mismatch")
patched = patched.replace(main_needle, main_replacement)
print(
    "V055_STAGE_A_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "objective": "train-only quadgram plus unigram",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
