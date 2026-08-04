#!/usr/bin/env python3
"""Apply the frozen scale-only manuscript allocation correction.

The v0.2 pilot source used a 5:1:1 train/dev/test manuscript cycle. That
cannot fill 1,000-line held-out sets with a 40-line-per-manuscript cap on the
eligible CATMuS stream. This patch changes only the allocation ratio to 2:1:1;
all data filters, caps, model code and scientific gates remain unchanged.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXPECTED_INPUT_SHA256 = "3ace252ecbf58f63b4144a2f61586e57763adb1a484fcc783c85cc1fc8baf7ec"
OLD = '''def split_cycle() -> tuple[str, ...]:
    # 5:1:1 manuscript assignment; caps force many held-out shelfmarks.
    return ("train", "train", "dev", "train", "train", "test", "train")
'''
NEW = '''def split_cycle() -> tuple[str, ...]:
    # 2:1:1 assignment fills 1,000-line dev/test sets while preserving
    # strict shelfmark separation and the 40-line held-out cap.
    return ("train", "train", "dev", "test")
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_scale_split.py TRAIN_V02_PY")
    path = Path(sys.argv[1])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"unexpected input SHA-256: {digest}")
    text = data.decode("utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError("scale split patch site mismatch")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("SCALE_SPLIT_PATCH_OK", hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
