#!/usr/bin/env python3
"""Acquire a dated Finnish historical panel from Kielipankki NLFCL VRT.

The exact public release ZIP and published MD5 are frozen. Texts are admitted
only when the VRT supplies an attributable work boundary, author metadata, a
publication year at or before the registered cutoff, and at least 4,096
normalized characters. No missing metadata is inferred.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ZIP_URL = "https://www.kielipankki.fi/download/nlfcl/fi/vrt/nlfcl-fi-vrt.zip"
MD5_URL = ZIP_URL + ".md5"
USER_AGENT = "VoynichCompressionTransfer/0.1 (NLFCL acquisition; github.com/digitalgoldfisj79/Voynichdecomp)"
TARGET = 12
MIN_UNITS = 4096
YEAR_CUTOFF = 1800


def fetch(url: str, attempts: int = 6) -> bytes:
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(min(90, 2 ** attempt))
    raise RuntimeError(f"fetch failed {url}: {last}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).replace("\u00a0", " ")).strip()


def parse_attributes(line: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, quote, value in re.findall(r"([:\w.-]+)\s*=\s*(['\"])(.*?)\2", line):
        attrs[key.casefold()] = html.unescape(value).strip()
    return attrs


def first(attrs: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = attrs.get(name, "").strip()
        if value:
            return value
    return ""


def publication_year(attrs: dict[str, str]) -> int | None:
    candidates = []
    for key, value in attrs.items():
        if any(term in key for term in ("year", "date", "vuosi", "julkais")):
            candidates.extend(int(match) for match in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", value))
    return min(candidates) if candidates else None


def iter_vrt_documents(archive: zipfile.ZipFile):
    for member in sorted(name for name in archive.namelist() if name.lower().endswith((".vrt", ".txt")) and not name.endswith("/")):
        with archive.open(member) as raw:
            attrs = None
            tokens: list[str] = []
            ordinal = 0
            for binary_line in raw:
                line = binary_line.decode("utf-8", errors="strict").rstrip("\r\n")
                stripped = line.strip()
                if stripped.startswith("<text"):
                    if attrs is not None:
                        yield member, ordinal, attrs, normalize(" ".join(tokens))
                        ordinal += 1
                    attrs = parse_attributes(stripped)
                    tokens = []
                elif stripped.startswith("</text"):
                    if attrs is not None:
                        yield member, ordinal, attrs, normalize(" ".join(tokens))
                        ordinal += 1
                    attrs = None
                    tokens = []
                elif attrs is not None and stripped and not stripped.startswith("<"):
                    token = line.split("\t", 1)[0].strip()
                    if token:
                        tokens.append(token)
            if attrs is not None:
                yield member, ordinal, attrs, normalize(" ".join(tokens))


def split(index: int) -> str:
    return "train" if index < 8 else ("dev" if index < 10 else "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/nlfcl_fi")
    parser.add_argument("--zip-url", default=ZIP_URL)
    parser.add_argument("--md5-url", default=MD5_URL)
    parser.add_argument("--year-cutoff", type=int, default=YEAR_CUTOFF)
    parser.add_argument("--target-docs", type=int, default=TARGET)
    parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    args = parser.parse_args()

    output = Path(args.output)
    normalized_dir = output / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    archive_bytes = fetch(args.zip_url)
    md5_bytes = fetch(args.md5_url)
    (output / "nlfcl-fi-vrt.zip").write_bytes(archive_bytes)
    (output / "nlfcl-fi-vrt.zip.md5").write_bytes(md5_bytes)
    expected_md5_match = re.search(rb"\b([0-9a-fA-F]{32})\b", md5_bytes)
    if not expected_md5_match:
        raise ValueError("published MD5 file contains no digest")
    expected_md5 = expected_md5_match.group(1).decode().lower()
    actual_md5 = hashlib.md5(archive_bytes).hexdigest()
    if actual_md5 != expected_md5:
        raise ValueError(f"release MD5 mismatch: {actual_md5} != {expected_md5}")

    archive_path = output / "nlfcl-fi-vrt.zip"
    candidates = []
    rejected = Counter()
    attribute_keys = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for member, ordinal, attrs, text in iter_vrt_documents(archive):
            attribute_keys.update(attrs.keys())
            author = first(attrs, ("author", "authors", "creator", "kirjoittaja", "writer"))
            title = first(attrs, ("title", "work", "name", "teos", "nimi"))
            work_id = first(attrs, ("id", "doc_id", "document_id", "urn", "filename", "file"))
            year = publication_year(attrs)
            if not author:
                rejected["missing_author"] += 1
                continue
            if not title:
                rejected["missing_title"] += 1
                continue
            if year is None:
                rejected["missing_year"] += 1
                continue
            if year > args.year_cutoff:
                rejected["after_cutoff"] += 1
                continue
            if len(text) < args.min_units:
                rejected["short"] += 1
                continue
            identity = work_id or f"{member}#{ordinal}"
            candidates.append({"member": member, "ordinal": ordinal, "attrs": attrs, "author": author, "title": title, "work_id": identity, "year": year, "text": text})

    candidates.sort(key=lambda row: (row["year"], row["author"].casefold(), row["title"].casefold(), row["work_id"]))
    selected = []
    seen_work_ids = set()
    for candidate in candidates:
        if candidate["work_id"] in seen_work_ids:
            rejected["duplicate_work_id"] += 1
            continue
        remaining = args.target_docs - len(selected)
        authors = {row["author"] for row in selected}
        if remaining == 1 and len(authors) < 2 and candidate["author"] in authors:
            rejected["reserved_for_second_author"] += 1
            continue
        selected.append(candidate)
        seen_work_ids.add(candidate["work_id"])
        if len(selected) >= args.target_docs:
            break

    manifest = []
    accepted_log = []
    for index, item in enumerate(selected):
        stem = f"{index:02d}_fi_{sha256(item['work_id'].encode())[:16]}"
        path = normalized_dir / f"{stem}.txt"
        data = item["text"].encode("utf-8")
        path.write_bytes(data)
        author_id = "nlfcl-author:" + sha256(item["author"].encode())[:16]
        row = {
            "corpus_id": "nlfcl_historical_fi",
            "document_id": f"nlfcl-fi-{sha256(item['work_id'].encode())[:20]}",
            "split": split(index), "class_label": "fi", "language": "fi", "family": "historical_plaintext",
            "path": f"normalized/{path.name}", "sha256": sha256(data), "encoding": "utf-8",
            "license": "Public Domain source works; Kielipankki public VRT release", "author_id": author_id,
            "work_id": item["work_id"], "date_band": f"publication_year={item['year']};not_later_than_{args.year_cutoff}",
            "notes": f"title={item['title']}; author={item['author']}; archive_member={item['member']}; text_ordinal={item['ordinal']}; source={args.zip_url}; release_md5={actual_md5}; normalized_units={len(item['text'])}",
        }
        manifest.append(row)
        accepted_log.append({**row, "attributes": item["attrs"]})

    duplicates = []
    for index, first_row in enumerate(manifest):
        first_text = (output / first_row["path"]).read_text(encoding="utf-8")
        for second_row in manifest[index + 1:]:
            second_text = (output / second_row["path"]).read_text(encoding="utf-8")
            ratio = SequenceMatcher(None, first_text[:20000], second_text[:20000], autojunk=False).ratio()
            if first_row["sha256"] == second_row["sha256"] or ratio >= 0.85:
                duplicates.append({"a": first_row["document_id"], "b": second_row["document_id"], "ratio": ratio})

    fields = ["corpus_id", "document_id", "split", "class_label", "language", "family", "path", "sha256", "encoding", "license", "author_id", "work_id", "date_band", "notes"]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (output / "accepted.json").write_text(json.dumps(accepted_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "duplicate_screen.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "attribute_inventory.json").write_text(json.dumps(dict(attribute_keys.most_common()), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    author_count = len({row["author_id"] for row in manifest})
    status = "ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" if len(manifest) >= args.target_docs and author_count >= 2 and not duplicates else ("BLOCKED_DUPLICATES" if duplicates else "BLOCKED_INSUFFICIENT_ELIGIBLE_WORKS")
    summary = {
        "programme": "compression-transfer-v0.1", "panel": "nlfcl_finnish_historical_fallback", "status": status,
        "voynich_accessed": False, "source_url": args.zip_url, "source_sha256": sha256(archive_bytes),
        "source_md5": actual_md5, "published_md5_url": args.md5_url, "year_cutoff": args.year_cutoff,
        "minimum_units": args.min_units, "target_documents": args.target_docs, "accepted_documents": len(manifest),
        "accepted_authors": author_count, "candidate_documents": len(candidates), "rejection_counts": dict(rejected),
        "duplicate_findings_count": len(duplicates),
        "rights_boundary": "The corpus provider describes the source works as Public Domain. This acquisition freezes the processed VRT release and does not infer missing author, title or date metadata.",
    }
    summary["scientific_payload_sha256"] = sha256(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
