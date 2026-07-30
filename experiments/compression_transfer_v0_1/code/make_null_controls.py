#!/usr/bin/env python3
"""Generate deterministic frequency- and order-destroying control corpora."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffle_nonspace(text: str, rng: random.Random) -> str:
    chars = [c for c in text if not c.isspace()]
    rng.shuffle(chars)
    iterator = iter(chars)
    return "".join(c if c.isspace() else next(iterator) for c in text)


def shuffle_tokens(text: str, rng: random.Random) -> str:
    tokens = text.split()
    rng.shuffle(tokens)
    return " ".join(tokens) + "\n"


def shuffle_blocks(text: str, rng: random.Random, block_size: int) -> str:
    compact = " ".join(text.split())
    blocks = [compact[i:i + block_size] for i in range(0, len(compact), block_size)]
    rng.shuffle(blocks)
    return "".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    out_rows = []
    for row_index, row in enumerate(source_rows):
        source_path = Path(row["path"])
        if not source_path.is_absolute():
            source_path = (args.manifest.parent / source_path).resolve()
        text = source_path.read_text(encoding=row.get("encoding") or "utf-8")
        for control_i, control in enumerate(("char_shuffle", "token_shuffle", "block_shuffle")):
            rng = random.Random(args.seed + 100000 * row_index + control_i)
            if control == "char_shuffle":
                transformed = shuffle_nonspace(text, rng)
            elif control == "token_shuffle":
                transformed = shuffle_tokens(text, rng)
            else:
                transformed = shuffle_blocks(text, rng, args.block_size)
            out_name = f"{row['document_id']}__{control}.txt"
            out_path = args.output_dir / out_name
            out_path.write_text(transformed, encoding="utf-8")
            new = dict(row)
            new.update({
                "corpus_id": f"{row['corpus_id']}__{control}",
                "document_id": f"{row['document_id']}__{control}",
                "class_label": control,
                "family": "null_control",
                "path": out_name,
                "sha256": sha(out_path),
                "encoding": "utf-8",
                "license": row.get("license", "") + "; derived control",
                "notes": f"deterministic {control}; source={row['document_id']}",
            })
            out_rows.append(new)
    manifest_out = args.output_dir / "manifest_controls.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(manifest_out)


if __name__ == "__main__":
    main()
