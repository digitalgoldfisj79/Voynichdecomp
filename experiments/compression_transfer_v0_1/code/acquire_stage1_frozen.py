#!/usr/bin/env python3
"""Acquire the frozen Stage-1 period-tolerant corpus panel.

The source selection is data, not code: pass a committed JSON file containing
exact source identifiers, URLs, labels, authors, works, dates and splits.
This program performs only deterministic acquisition, normalization, hashing,
manifest construction and duplicate screening. It computes no compression
metric and never opens Voynich data.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "CompressionTransferResearch/0.1 (github digitalgoldfisj79/Voynichdecomp)"
WS_RE = re.compile(r"\s+")
PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
MANIFEST_FIELDS = [
    "corpus_id", "document_id", "split", "class_label", "language", "family",
    "path", "sha256", "encoding", "license", "author_id", "work_id",
    "date_band", "notes",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return WS_RE.sub(" ", text).strip() + "\n"


def strip_gutenberg(text: str) -> str:
    start = PG_START.search(text)
    if start:
        text = text[start.end():]
    end = PG_END.search(text)
    if end:
        text = text[:end.start()]
    return normalize_text(text)


def request_bytes(session: requests.Session, url: str, expected_raw_sha256: str = "") -> bytes:
    last: Exception | None = None
    for _attempt in range(6):
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=(30, 180))
            response.raise_for_status()
            data = response.content
            if expected_raw_sha256 and sha256_bytes(data) != expected_raw_sha256:
                raise ValueError(f"raw hash mismatch for {url}")
            return data
        except Exception as exc:
            last = exc
    raise RuntimeError(f"failed to acquire {url}: {last}")


def extract_wikisource_pages(dump_path: Path, wanted_titles: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with bz2.open(dump_path, "rb") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if not elem.tag.endswith("page"):
                continue
            title = next((child.text or "" for child in elem if child.tag.endswith("title")), "")
            namespace = next((child.text or "" for child in elem if child.tag.endswith("ns")), "")
            if namespace == "0" and title in wanted_titles:
                text = next((child.text or "" for child in elem.iter() if child.tag.endswith("text")), "")
                found[title] = text
            elem.clear()
            if len(found) == len(wanted_titles):
                break
    missing = sorted(wanted_titles - set(found))
    if missing:
        raise KeyError(f"missing Wikisource titles: {missing}")
    return found


def strip_wikitext(text: str) -> str:
    import mwparserfromhell

    parsed = mwparserfromhell.parse(text)
    plain = parsed.strip_code(normalize=True, collapse=True)
    return normalize_text(plain)


def load_hf_exact(selection: list[dict[str, Any]]) -> dict[str, str]:
    from datasets import load_dataset

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selection:
        by_dataset[record["dataset"]].append(record)
    output: dict[str, str] = {}
    for dataset, records in sorted(by_dataset.items()):
        field = records[0]["match_field"]
        wanted = {str(record["match_value"]): record for record in records}
        found_here: set[str] = set()
        for row in load_dataset(dataset, split=records[0].get("dataset_split", "train"), streaming=True):
            cursor: Any = row
            for part in field.split("."):
                cursor = cursor[part]
            key = str(cursor)
            if key in wanted:
                document_id = wanted[key]["document_id"]
                output[document_id] = normalize_text(str(row[records[0].get("text_field", "text")]))
                found_here.add(document_id)
                if len(found_here) == len(records):
                    break
        missing = [record["document_id"] for record in records if record["document_id"] not in output]
        if missing:
            raise KeyError(f"missing HF records in {dataset}: {missing}")
    return output


def sampled_shingles(text: str, width: int = 5, cap: int = 10000) -> set[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < width:
        return {tuple(tokens)}
    n = len(tokens) - width + 1
    stride = max(1, n // cap)
    return {tuple(tokens[i:i + width]) for i in range(0, n, stride)}


def duplicate_screen(documents: dict[str, str], records: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, list[str]] = defaultdict(list)
    for doc_id, text in documents.items():
        exact[sha256_bytes(text.encode("utf-8"))].append(doc_id)
    exact_groups = [sorted(group) for group in exact.values() if len(group) > 1]

    by_corpus: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_corpus[record["corpus_id"]].append(record["document_id"])
    signatures = {doc_id: sampled_shingles(documents[doc_id]) for doc_id in documents}
    near: list[dict[str, Any]] = []
    for corpus_id, ids in sorted(by_corpus.items()):
        ids = sorted(ids)
        for i, left in enumerate(ids):
            a = signatures[left]
            for right in ids[i + 1:]:
                b = signatures[right]
                union = len(a | b)
                score = len(a & b) / union if union else 1.0
                if score >= 0.50:
                    near.append({"corpus_id": corpus_id, "left": left, "right": right, "sampled_token_5gram_jaccard": score})
    return {"method": "exact full-text SHA-256 plus within-class evenly sampled token 5-gram Jaccard", "exact_duplicate_groups": exact_groups, "near_duplicate_pairs_ge_0_50": near}


def validate_panel(records: list[dict[str, Any]], documents: dict[str, str], duplicate_report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter(record["corpus_id"] for record in records)
    for corpus_id, count in sorted(counts.items()):
        if count < 12:
            errors.append(f"{corpus_id}: only {count} documents")
        splits = {record["split"] for record in records if record["corpus_id"] == corpus_id}
        if not {"train", "dev", "test"}.issubset(splits):
            errors.append(f"{corpus_id}: missing required split; has {sorted(splits)}")
        authors = {record["author_id"] for record in records if record["corpus_id"] == corpus_id and record["author_id"]}
        works = {record["work_id"] for record in records if record["corpus_id"] == corpus_id and record["work_id"]}
        if len(authors) < 2:
            errors.append(f"{corpus_id}: fewer than two attributable authors/entities")
        if len(works) < 2:
            errors.append(f"{corpus_id}: fewer than two works")
    for record in records:
        if not record.get("license") or not record.get("source_url"):
            errors.append(f"{record['document_id']}: blank licence or source URL")
        text = documents[record["document_id"]]
        if len(text) < 4096:
            errors.append(f"{record['document_id']}: fewer than 4096 normalized characters")
    if duplicate_report["exact_duplicate_groups"]:
        errors.append("exact duplicate documents detected")
    if duplicate_report["near_duplicate_pairs_ge_0_50"]:
        errors.append("near-duplicate candidate pairs at Jaccard >= 0.50")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "documents_per_corpus": dict(sorted(counts.items())),
        "n_documents": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    specification = json.loads(selection_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = specification["records"]
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "documents"
    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    documents: dict[str, str] = {}
    provenance: dict[str, Any] = {}

    for record in records:
        if record["source_type"] != "gutenberg":
            continue
        raw = request_bytes(session, record["source_url"], record.get("raw_sha256", ""))
        text = strip_gutenberg(raw.decode(record.get("source_encoding", "utf-8"), errors="replace"))
        documents[record["document_id"]] = text
        provenance[record["document_id"]] = {"raw_sha256": sha256_bytes(raw)}

    ws_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["source_type"] == "wikisource_dump":
            ws_groups[record["dump_url"]].append(record)
    downloads_dir = output_dir / "source_archives"
    downloads_dir.mkdir(exist_ok=True)
    for dump_url, group in sorted(ws_groups.items()):
        raw = request_bytes(session, dump_url, group[0].get("dump_sha256", ""))
        dump_path = downloads_dir / (hashlib.sha256(dump_url.encode()).hexdigest()[:16] + ".xml.bz2")
        dump_path.write_bytes(raw)
        pages = extract_wikisource_pages(dump_path, {record["source_title"] for record in group})
        for record in group:
            documents[record["document_id"]] = strip_wikitext(pages[record["source_title"]])
            provenance[record["document_id"]] = {"dump_sha256": sha256_bytes(raw), "source_title": record["source_title"]}

    hf_records = [record for record in records if record["source_type"] == "huggingface"]
    if hf_records:
        documents.update(load_hf_exact(hf_records))
        for record in hf_records:
            provenance[record["document_id"]] = {
                "dataset": record["dataset"],
                "match_field": record["match_field"],
                "match_value": record["match_value"],
            }

    missing = sorted({record["document_id"] for record in records} - set(documents))
    if missing:
        raise RuntimeError(f"acquisition produced no text for: {missing}")

    manifest_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, Any]] = []
    for record in records:
        doc_id = record["document_id"]
        text = documents[doc_id]
        encoded = text.encode("utf-8")
        out_path = data_dir / f"{doc_id}.txt"
        out_path.write_bytes(encoded)
        normalized_sha = sha256_bytes(encoded)
        expected = record.get("normalized_sha256", "")
        if expected and normalized_sha != expected:
            raise ValueError(f"normalized hash mismatch for {doc_id}: {normalized_sha} != {expected}")
        manifest_rows.append({
            "corpus_id": record["corpus_id"],
            "document_id": doc_id,
            "split": record["split"],
            "class_label": record["class_label"],
            "language": record["language"],
            "family": "plaintext",
            "path": f"documents/{doc_id}.txt",
            "sha256": normalized_sha,
            "encoding": "utf-8",
            "license": record["license"],
            "author_id": record["author_id"],
            "work_id": record["work_id"],
            "date_band": record["date_band"],
            "notes": f"source={record['source_url']}; title={record['title']}",
        })
        metadata_rows.append({
            **record,
            **provenance.get(doc_id, {}),
            "normalized_sha256": normalized_sha,
            "normalized_characters": len(text),
            "normalized_bytes": len(encoded),
        })

    duplicate_report = duplicate_screen(documents, records)
    qualification = validate_panel(records, documents, duplicate_report)

    manifest_path = output_dir / "formal_stage1_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "source_metadata.json").write_text(json.dumps(metadata_rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "duplicate_screen.json").write_text(json.dumps(duplicate_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "acquisition_qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    freeze = {
        "programme": specification["programme"],
        "panel": specification["panel"],
        "voynich_status": "SEALED",
        "selection_sha256": sha256_bytes(selection_path.read_bytes()),
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "source_metadata_sha256": sha256_bytes((output_dir / "source_metadata.json").read_bytes()),
        "duplicate_screen_sha256": sha256_bytes((output_dir / "duplicate_screen.json").read_bytes()),
        "qualification": qualification,
        "scientific_results_computed": False,
    }
    freeze["freeze_payload_sha256"] = sha256_bytes(canonical_json(freeze))
    (output_dir / "STAGE1_ACQUISITION_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))

    if qualification["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
