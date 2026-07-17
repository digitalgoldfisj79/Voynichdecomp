#!/usr/bin/env python3
"""Native Hugging Face UV entry point for the preregistered v1.2 smoke run.

The scientific source is reconstructed from the immutable v1.1 source parts,
derived by the preregistered substitutions, and verified by exact digests before
execution. This file alters only execution and result transport.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import urllib.request

PARENT_ROOT = (
    "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
    "8f5cd23cd1f8415c21c3a9367c3e280eafe58bbd/"
    "experiments/blind_stroke_palaeography_v1/"
)
PARENT_LENGTH = 37401
PARENT_SHA256 = "f93fc90c0527266d71d876962050923b9f7e4020c77dc8c7fad83019b80ac883"
DERIVED_LENGTH = 37929
DERIVED_SHA256 = "edb7fe7b3405e2c41678ab035c267f563657b8e7a379425e9b26a89675b73607"


def reconstruct() -> str:
    chunks: list[str] = []
    for i in range(5):
        url = PARENT_ROOT + f"source_bundle/external_calibration.py.v4.gz.b64.part{i:02d}"
        with urllib.request.urlopen(url) as response:
            chunks.append(response.read().decode("ascii").strip())
    raw = gzip.decompress(base64.b64decode("".join(chunks), validate=True))
    if len(raw) != PARENT_LENGTH or hashlib.sha256(raw).hexdigest() != PARENT_SHA256:
        raise RuntimeError("parent source digest mismatch")
    return raw.decode("utf-8")


def derive(source: str) -> str:
    old_head = '''def evaluate_foldlocal(base: dict[str, np.ndarray], tile_features: dict[str, list[np.ndarray]],
                       writers: list[str], pages: list[str], seed: int) -> dict[str, Any]:
    folds = grouped_folds(writers, pages, 5, seed)'''
    new_head = '''def evaluate_foldlocal(base: dict[str, np.ndarray], tile_features: dict[str, list[np.ndarray]],
                       writers: list[str], pages: list[str], seed: int) -> dict[str, Any]:
    # v1.2: use the largest valid grouped leave-one-page-out fold count up to five.
    # Historical-WI contains exactly three physical pages per writer after
    # colour/binary derivative canonicalisation, so n_folds=3 there.
    page_counts = [len({p for ww, p in zip(writers, pages) if ww == w}) for w in sorted(set(writers))]
    n_folds = min(5, min(page_counts)) if page_counts else 0
    if n_folds < 2:
        raise RuntimeError(f"insufficient physical pages for grouped evaluation: {page_counts[:20]}")
    folds = grouped_folds(writers, pages, n_folds, seed)'''
    substitutions = [
        (old_head, new_head),
        ("for f in range(5):", "for f in range(n_folds):"),
        ('ap.add_argument("--pages-per-writer", type=int, default=5)',
         'ap.add_argument("--pages-per-writer", type=int, default=3)'),
        ('"schema": "blind-pal-external-calibration-v1"',
         '"schema": "blind-pal-external-calibration-v1.2"'),
    ]
    for old, new in substitutions:
        if source.count(old) != 1:
            raise RuntimeError(f"unexpected substitution count for {old[:80]!r}")
        source = source.replace(old, new, 1)
    raw = source.encode("utf-8")
    if len(raw) != DERIVED_LENGTH or hashlib.sha256(raw).hexdigest() != DERIVED_SHA256:
        raise RuntimeError("derived source digest mismatch")
    return source


def load_module():
    source = derive(reconstruct())
    destination = pathlib.Path("/tmp/external_calibration_v1_2.py")
    destination.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("external_calibration_v1_2", destination)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create v1.2 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main() -> int:
    module = load_module()
    module.upload_directory = lambda path, repo, token, path_in_repo: {
        "transport": "job_log",
        "path": path_in_repo,
    }
    work = "/tmp/blindpal_smoke_v12_uv"
    sys.argv = [
        "external_calibration_v1_2.py",
        "--corpus", "historical_wi",
        "--work", work,
        "--output-repo", "Digitalgoldfish79/blind-scribal-hands-v1",
        "--max-writers", "20",
        "--pages-per-writer", "3",
        "--fragments-per-page", "1",
        "--max-tiles", "2",
        "--workers", "32",
        "--batch-size", "128",
        "--permutations", "3",
        "--panel-seed", "20260717",
    ]
    rc = int(module.main())
    result = pathlib.Path(work) / "historical_wi" / "output" / "calibration_result.json"
    print("V12_UV_SMOKE_RESULT " + result.read_text(encoding="utf-8"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
