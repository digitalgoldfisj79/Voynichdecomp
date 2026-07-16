#!/usr/bin/env python3
"""Execution-only launcher creating per-candidate SentencePiece directories."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v060_family_s_stage_s3.py")
source = path.read_text(encoding="utf-8")
needle = '''def train_segmentation(
    text: str, model_type: str, vocab_size: int, working: Path
) -> list[str]:
    input_path = working / "cipher.txt"
'''
replacement = '''def train_segmentation(
    text: str, model_type: str, vocab_size: int, working: Path
) -> list[str]:
    working.mkdir(parents=True, exist_ok=True)
    input_path = working / "cipher.txt"
'''
if source.count(needle) != 1:
    raise RuntimeError("S3 directory patch site mismatch")
patched = source.replace(needle, replacement)
print(
    "V060_S3_LAUNCHER_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "scientific_search_changed": False,
        "reason": "create temporary SentencePiece directories",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
