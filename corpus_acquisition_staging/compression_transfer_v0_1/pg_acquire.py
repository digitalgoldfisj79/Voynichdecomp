#!/usr/bin/env python3
"""Acquire a public-domain multilingual panel from Project Gutenberg.

Eligibility is determined from the official machine-readable catalog. A work is
admitted only if: language is registered; every listed contributor has one
explicit death year at or before the conservative cutoff; a plain-text file is
available; normalized text is at least the registered length; and the work is
not an exact or near duplicate. Exact catalog and text bytes are hashed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

UA = "VoynichCompressionTransfer/0.1 (Project Gutenberg acquisition; github.com/digitalgoldfisj79/Voynichdecomp)"
CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
LANGS = ["en", "de", "fi", "tr", "el", "he", "la"]
TARGET = 12
MIN_UNITS = 4096
DEATH_CUTOFF = 1900


def fetch(url: str, tries: int = 6) -> bytes:
    last = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"fetch failed {url}: {last}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).replace("\u00a0", " ")).strip()


def contributor_death_years(author_field: str) -> list[int] | None:
    """Return one explicit death year per listed contributor, else None."""
    contributors = [value.strip() for value in (author_field or "").split(";") if value.strip()]
    if not contributors:
        return None
    years: list[int] = []
    for contributor in contributors:
        matches = re.findall(r"(?<!\d)(?:\d{3,4})-(\d{3,4})(?!\d)", contributor)
        if len(matches) != 1:
            return None
        years.append(int(matches[0]))
    return years


def language_codes(field: str) -> set[str]:
    return {value.strip().lower() for value in re.split(r"[;,\s]+", field or "") if value.strip()}


def ebook_id(row: dict[str, str]) -> int:
    for key in ("Text#", "Text #", "EBook-No.", "ebook_id"):
        if row.get(key, "").strip().isdigit():
            return int(row[key].strip())
    raise ValueError(f"no ebook id fields={list(row)}")


def text_urls(identifier: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt",
        f"https://www.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt.utf-8",
        f"https://www.gutenberg.org/files/{identifier}/{identifier}-0.txt",
        f"https://www.gutenberg.org/files/{identifier}/{identifier}.txt",
    ]


def strip_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return normalize(text)


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "iso-8859-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", raw, 0, 1, "no registered decoding succeeded")


def split(index: int) -> str:
    return "train" if index < 8 else ("dev" if index < 10 else "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/stage1_gutenberg")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--death-cutoff", type=int, default=DEATH_CUTOFF)
    parser.add_argument("--target-docs", type=int, default=TARGET)
    parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    args = parser.parse_args()

    output = Path(args.output)
    raw_directory = output / "source_raw"
    normalized_directory = output / "normalized"
    raw_directory.mkdir(parents=True, exist_ok=True)
    normalized_directory.mkdir(parents=True, exist_ok=True)

    catalog = fetch(args.catalog)
    (output / "pg_catalog.csv.gz").write_bytes(catalog)
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(catalog)), encoding="utf-8-sig", newline="")))
    selected = {language: [] for language in LANGS}
    rejected: dict[str, list[dict]] = defaultdict(list)

    parsed = []
    for row in rows:
        try:
            identifier = ebook_id(row)
        except Exception:
            continue
        parsed.append((identifier, row))

    for identifier, row in sorted(parsed):
        if all(len(selected[language]) >= args.target_docs for language in LANGS):
            break
        codes = language_codes(row.get("Language", ""))
        relevant = [language for language in LANGS if language in codes and len(selected[language]) < args.target_docs]
        if not relevant:
            continue
        authors = row.get("Authors", "") or row.get("Author", "") or ""
        deaths = contributor_death_years(authors)
        if not deaths or max(deaths) > args.death_cutoff:
            for language in relevant:
                rejected[language].append({"ebook_id": identifier, "reason": "contributors_not_fully_date_eligible", "authors": authors})
            continue
        title = row.get("Title", "").strip()
        acquired = None
        errors = []
        for url in text_urls(identifier):
            try:
                raw = fetch(url, tries=2)
                text = strip_boilerplate(decode(raw))
                if len(text) >= args.min_units:
                    acquired = (url, raw, text)
                    break
                errors.append({"url": url, "reason": "short", "units": len(text)})
            except Exception as exc:
                errors.append({"url": url, "reason": "fetch_or_decode", "error": repr(exc)})
        if not acquired:
            for language in relevant:
                rejected[language].append({"ebook_id": identifier, "reason": "no_eligible_plain_text", "details": errors})
            continue
        url, raw, text = acquired
        for language in relevant:
            selected[language].append({
                "ebook_id": identifier,
                "title": title,
                "authors": authors,
                "death_years": deaths,
                "url": url,
                "raw": raw,
                "text": text,
                "subjects": row.get("Subjects", ""),
                "bookshelves": row.get("Bookshelves", ""),
                "issued": row.get("Issued", ""),
            })

    manifest = []
    accepted_log = []
    for language in LANGS:
        for index, item in enumerate(selected[language]):
            stem = f"{index:02d}_{language}_pg{item['ebook_id']}"
            raw_path = raw_directory / f"{stem}.txt"
            normalized_path = normalized_directory / f"{stem}.txt"
            normalized_bytes = item["text"].encode("utf-8")
            raw_path.write_bytes(item["raw"])
            normalized_path.write_bytes(normalized_bytes)
            author_id = "pg-author:" + hashlib.sha256(item["authors"].encode()).hexdigest()[:16]
            row = {
                "corpus_id": f"gutenberg_historical_{language}",
                "document_id": f"pg-{language}-{item['ebook_id']}",
                "split": split(index),
                "class_label": language,
                "language": language,
                "family": "historical_plaintext",
                "path": f"normalized/{stem}.txt",
                "sha256": sha256(normalized_bytes),
                "encoding": "utf-8",
                "license": "Project Gutenberg public-domain ebook; additionally screened by contributor death dates",
                "author_id": author_id,
                "work_id": f"pg:{item['ebook_id']}",
                "date_band": f"all_contributor_deaths_not_later_than_{args.death_cutoff};death_years={'|'.join(map(str, item['death_years']))}",
                "notes": f"title={item['title']}; authors={item['authors']}; issued={item['issued']}; source={item['url']}; catalog_source={args.catalog}; raw_sha256={sha256(item['raw'])}; normalized_units={len(item['text'])}",
            }
            manifest.append(row)
            accepted_log.append({**row, "title": item["title"], "authors": item["authors"], "subjects": item["subjects"], "bookshelves": item["bookshelves"]})

    duplicates = []
    for language in LANGS:
        documents = [row for row in accepted_log if row["language"] == language]
        texts = {row["document_id"]: (normalized_directory / Path(row["path"]).name).read_text() for row in documents}
        for index, first in enumerate(documents):
            for second in documents[index + 1:]:
                ratio = SequenceMatcher(None, texts[first["document_id"]][:20000], texts[second["document_id"]][:20000], autojunk=False).ratio()
                if first["sha256"] == second["sha256"] or ratio >= 0.85:
                    duplicates.append({"language": language, "a": first["document_id"], "b": second["document_id"], "ratio": ratio})

    fields = ["corpus_id", "document_id", "split", "class_label", "language", "family", "path", "sha256", "encoding", "license", "author_id", "work_id", "date_band", "notes"]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (output / "accepted.json").write_text(json.dumps(accepted_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (output / "rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (output / "duplicate_screen.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    counts = Counter(row["language"] for row in manifest)
    author_counts = {language: len({row["author_id"] for row in manifest if row["language"] == language}) for language in LANGS}
    eligible = [language for language in LANGS if counts[language] >= args.target_docs and author_counts[language] >= 2]
    status = "ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" if len(eligible) == len(LANGS) and not duplicates else ("BLOCKED_DUPLICATES" if duplicates else "BLOCKED_INSUFFICIENT_ELIGIBLE_WORKS")
    summary = {
        "programme": "compression-transfer-v0.1",
        "panel": "stage1_gutenberg_historical",
        "status": status,
        "voynich_accessed": False,
        "catalog_url": args.catalog,
        "catalog_sha256": sha256(catalog),
        "catalog_rows": len(rows),
        "contributor_death_cutoff": args.death_cutoff,
        "target_documents_per_language": args.target_docs,
        "minimum_units": args.min_units,
        "counts": dict(counts),
        "author_counts": author_counts,
        "eligible_languages": eligible,
        "duplicate_findings_count": len(duplicates),
        "rights_boundary": "Project Gutenberg US public-domain status is not treated as sufficient alone; every listed contributor must have exactly one explicit catalogued death year at or before the registered cutoff. Editorial, translation and edition rights remain a residual limitation recorded for specialist review.",
    }
    summary["scientific_payload_sha256"] = sha256(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
