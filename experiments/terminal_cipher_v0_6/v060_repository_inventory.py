#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "experiments" / "terminal_cipher_v0_6"

for path in sorted(TARGET.rglob("*")):
    if path.is_file():
        print(path.relative_to(ROOT))

print("--- DATA CANDIDATES ---")
terms = (
    "voynich", "zandbergen", "landini", "takaha", "evola",
    "interlinear", "eva", "ivtff", "transcription",
)
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and any(term in str(path).lower() for term in terms):
        print(path.relative_to(ROOT))
