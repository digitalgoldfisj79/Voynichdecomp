#!/usr/bin/env python3
"""Corpus-manifest loading and validation."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {
    "corpus_id", "document_id", "split", "class_label", "language",
    "family", "path", "sha256", "encoding", "license",
}
ALLOWED_SPLITS = {"train", "dev", "test", "sealed"}


@dataclass(frozen=True)
class DocumentRecord:
    corpus_id: str
    document_id: str
    split: str
    class_label: str
    language: str
    family: str
    path: Path
    sha256: str
    encoding: str
    license: str
    author_id: str = ""
    work_id: str = ""
    date_band: str = ""
    notes: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path, verify_hashes: bool = True) -> list[DocumentRecord]:
    base = path.parent
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows: list[DocumentRecord] = []
        for i, row in enumerate(reader, start=2):
            split = row["split"].strip()
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"line {i}: invalid split {split!r}")
            file_path = Path(row["path"].strip())
            if not file_path.is_absolute():
                file_path = (base / file_path).resolve()
            if not file_path.is_file():
                raise FileNotFoundError(f"line {i}: missing file {file_path}")
            expected = row["sha256"].strip().lower()
            if verify_hashes and expected:
                actual = sha256_file(file_path)
                if actual != expected:
                    raise ValueError(f"line {i}: hash mismatch for {file_path}: {actual} != {expected}")
            rows.append(DocumentRecord(
                corpus_id=row["corpus_id"].strip(),
                document_id=row["document_id"].strip(),
                split=split,
                class_label=row["class_label"].strip(),
                language=row["language"].strip(),
                family=row["family"].strip(),
                path=file_path,
                sha256=expected,
                encoding=row["encoding"].strip() or "utf-8",
                license=row["license"].strip(),
                author_id=row.get("author_id", "").strip(),
                work_id=row.get("work_id", "").strip(),
                date_band=row.get("date_band", "").strip(),
                notes=row.get("notes", "").strip(),
            ))
    validate_manifest(rows)
    return rows


def validate_manifest(rows: Iterable[DocumentRecord]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("manifest is empty")
    seen_doc: dict[str, str] = {}
    corpora: dict[str, set[str]] = {}
    for row in rows:
        if not row.corpus_id or not row.document_id:
            raise ValueError("corpus_id and document_id must be non-empty")
        prior = seen_doc.get(row.document_id)
        if prior is not None and prior != row.split:
            raise ValueError(f"document leakage: {row.document_id} occurs in {prior} and {row.split}")
        seen_doc[row.document_id] = row.split
        corpora.setdefault(row.corpus_id, set()).add(row.split)
    for corpus, splits in sorted(corpora.items()):
        if "train" not in splits:
            raise ValueError(f"corpus {corpus!r} has no train document")
        if not ({"dev", "test"} & splits):
            raise ValueError(f"corpus {corpus!r} has neither dev nor test documents")
