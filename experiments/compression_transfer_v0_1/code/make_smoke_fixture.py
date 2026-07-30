#!/usr/bin/env python3
"""Generate deterministic synthetic corpora for smoke testing."""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "smoke"

SOURCES = {
    "alpha": ("abcde ", {"a": "aaab", "b": "bc", "c": "cde", "d": "dea", "e": "eab", " ": "abcde"}),
    "beta": ("klmno ", {"k": "kkkl", "l": "lmn", "m": "mmo", "n": "nok", "o": "okl", " ": "klmno"}),
    "gamma": ("pqrst ", {"p": "pppq", "q": "qrs", "r": "rrt", "s": "stp", "t": "tpq", " ": "pqrst"}),
    "delta": ("uvwxy ", {"u": "uuuv", "v": "vwx", "w": "wwy", "x": "xyu", "y": "yuv", " ": "uvwxy"}),
}


def make_text(name: str, seed: int, n: int = 18000) -> str:
    alphabet, transitions = SOURCES[name]
    rng = random.Random(seed)
    ch = alphabet[0]
    out = []
    for i in range(n):
        if i and i % 37 == 0:
            out.append(" ")
            ch = rng.choice(alphabet[:-1])
            continue
        out.append(ch)
        ch = rng.choice(transitions.get(ch, alphabet))
    text = "".join(out)
    return "\n".join(text[i:i + 120] for i in range(0, len(text), 120)) + "\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_i, source in enumerate(SOURCES):
        for split_i, split in enumerate(("train", "dev", "test")):
            for replicate in range(2):
                doc_id = f"{source}_{split}_{replicate}"
                path = FIX / f"{doc_id}.txt"
                path.write_text(make_text(source, seed=10000 * source_i + 100 * split_i + replicate), encoding="utf-8")
                rows.append({
                    "corpus_id": source,
                    "document_id": doc_id,
                    "split": split,
                    "class_label": source,
                    "language": source,
                    "family": "synthetic_markov",
                    "path": path.name,
                    "sha256": sha(path),
                    "encoding": "utf-8",
                    "license": "CC0 synthetic fixture",
                    "author_id": f"generator_{source}",
                    "work_id": doc_id,
                    "date_band": "synthetic",
                    "notes": "deterministic smoke fixture",
                })
    manifest = FIX / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(manifest)


if __name__ == "__main__":
    main()
