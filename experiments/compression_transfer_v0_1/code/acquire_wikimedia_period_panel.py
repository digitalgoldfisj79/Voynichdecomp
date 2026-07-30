#!/usr/bin/env python3
"""Acquire and freeze a topic-matched Wikimedia period-tolerant panel.

This acquisition is deterministic and eligibility-only. It resolves a fixed ordered
list of English Wikipedia topics through Wikidata sitelinks, fetches exact page
revisions in eight language Wikipedias, and admits the first twelve topics for
which every language has at least the registered 4,096 normalized codepoint units.

It does not analyse compression distances and does not access Voynich data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

USER_AGENT = (
    "VoynichCompressionTransfer/0.1 "
    "(research corpus acquisition; contact via github.com/digitalgoldfisj79/Voynichdecomp)"
)
LICENSE_NAME = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MIN_UNITS = 4096
TARGET_DOCS = 12
LANGUAGES = [
    ("en", "enwiki", "English"), ("de", "dewiki", "German"),
    ("fi", "fiwiki", "Finnish"), ("tr", "trwiki", "Turkish"),
    ("el", "elwiki", "Greek"), ("he", "hewiki", "Hebrew"),
    ("ar", "arwiki", "Arabic"), ("la", "lawiki", "Latin"),
]
CANDIDATE_EN_TITLES = [
    "History", "Philosophy", "Science", "Mathematics", "Physics", "Chemistry",
    "Biology", "Astronomy", "Medicine", "Geography", "Literature", "Music",
    "Architecture", "Religion", "Law", "Economics", "Agriculture", "Technology",
    "Education", "Language", "Writing", "Human", "Society", "Culture", "Art",
    "Earth", "Europe", "Asia", "Africa", "Ancient Rome", "Middle Ages",
    "Renaissance", "University", "Book", "Library", "City", "Nature", "Time",
    "Space", "Knowledge", "Logic", "Government", "Democracy", "Civilization",
    "Writing system", "Natural science", "Social science", "History of medicine",
]


def http_json(url: str, attempts: int = 6, pause: float = 0.25) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"failed after {attempts} attempts: {url}: {last}")


def api_url(host: str, params: dict[str, Any]) -> str:
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    return f"https://{host}/w/api.php?{urllib.parse.urlencode(params, doseq=True)}"


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    terminal = {
        "references", "external links", "see also", "bibliography", "notes",
        "einzelnachweise", "weblinks", "siehe auch", "literatur",
        "lähteet", "aiheesta muualla", "katso myös", "kirjallisuutta",
        "kaynakça", "dış bağlantılar", "ayrıca bakınız",
        "παραπομπές", "βιβλιογραφία", "εξωτερικοί σύνδεσμοι", "δείτε επίσης",
        "הערות שוליים", "קישורים חיצוניים", "ראו גם", "לקריאה נוספת",
        "المراجع", "وصلات خارجية", "انظر أيضًا", "انظر أيضا",
        "notae", "bibliographia", "vide etiam", "nexus externi",
    }
    lines = []
    for raw in text.splitlines():
        if raw.strip().casefold() in terminal:
            break
        lines.append(raw)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_qid(english_title: str) -> tuple[str, int, str]:
    payload = http_json(api_url("en.wikipedia.org", {
        "action": "query", "redirects": 1, "prop": "pageprops|info",
        "inprop": "url", "titles": english_title,
    }))
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing"):
        raise ValueError(f"English page missing: {english_title}")
    page = pages[0]
    qid = page.get("pageprops", {}).get("wikibase_item")
    if not qid:
        raise ValueError(f"English page lacks Wikidata item: {english_title}")
    return qid, int(page["pageid"]), str(page["title"])


def get_sitelinks(qid: str) -> dict[str, str]:
    sites = "|".join(site for _, site, _ in LANGUAGES)
    payload = http_json(api_url("www.wikidata.org", {
        "action": "wbgetentities", "ids": qid, "props": "sitelinks", "sitefilter": sites,
    }))
    links = payload.get("entities", {}).get(qid, {}).get("sitelinks", {})
    return {site: row["title"] for site, row in links.items() if "title" in row}


def fetch_page(lang: str, title: str) -> dict[str, Any]:
    host = f"{lang}.wikipedia.org"
    payload = http_json(api_url(host, {
        "action": "query", "redirects": 1, "prop": "extracts|revisions|info",
        "explaintext": 1, "exsectionformat": "plain", "rvprop": "ids|timestamp",
        "inprop": "url", "titles": title,
    }))
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing"):
        raise ValueError(f"missing page {host}:{title}")
    page = pages[0]
    revisions = page.get("revisions") or []
    if not revisions:
        raise ValueError(f"no revision metadata {host}:{title}")
    raw = str(page.get("extract", ""))
    normalized = normalize_surface(raw)
    return {
        "language": lang, "host": host, "pageid": int(page["pageid"]),
        "title": str(page["title"]), "canonicalurl": str(page.get("canonicalurl") or ""),
        "revid": int(revisions[0]["revid"]), "parentid": int(revisions[0].get("parentid", 0)),
        "timestamp": str(revisions[0]["timestamp"]), "raw_extract": raw,
        "normalized": normalized, "normalized_units": len(normalized),
    }


def split_for_index(index: int) -> str:
    return "train" if index < 8 else ("dev" if index < 10 else "test")


def screen_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_lang.setdefault(row["language"], []).append(row)
    for lang, docs in by_lang.items():
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                exact = a["sha256"] == b["sha256"]
                ratio = SequenceMatcher(None, a["normalized"][:20000], b["normalized"][:20000], autojunk=False).ratio()
                if exact or ratio >= 0.85:
                    findings.append({"language": lang, "document_a": a["document_id"],
                        "document_b": b["document_id"], "exact_sha_duplicate": exact,
                        "sequence_ratio_capped_20000": ratio})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/stage1_period_wikimedia")
    parser.add_argument("--target-docs", type=int, default=TARGET_DOCS)
    parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    args = parser.parse_args()
    out = Path(args.output); raw_dir = out / "source_raw"; norm_dir = out / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True); norm_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []; rejections: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for candidate_index, english_title in enumerate(CANDIDATE_EN_TITLES):
        if len(selected) >= args.target_docs:
            break
        candidate: dict[str, Any] = {"candidate_index": candidate_index, "english_title": english_title}
        try:
            qid, en_pageid, resolved_en_title = resolve_qid(english_title)
            sitelinks = get_sitelinks(qid)
            missing = [site for _, site, _ in LANGUAGES if site not in sitelinks]
            if missing:
                rejections.append({**candidate, "qid": qid, "reason": "missing_sitelinks", "details": missing}); continue
            pages = []; short = []
            for lang, site, label in LANGUAGES:
                page = fetch_page(lang, sitelinks[site])
                page.update({"site": site, "language_label": label, "qid": qid, "topic_en": resolved_en_title})
                pages.append(page)
                if page["normalized_units"] < args.min_units:
                    short.append({"language": lang, "units": page["normalized_units"], "title": page["title"]})
            if short:
                rejections.append({**candidate, "qid": qid, "reason": "short_in_one_or_more_languages", "details": short}); continue
            selected.append({"qid": qid, "english_title": resolved_en_title, "en_pageid": en_pageid, "pages": pages})
        except Exception as exc:
            rejections.append({**candidate, "reason": "acquisition_error", "details": repr(exc)})
    rows: list[dict[str, Any]] = []
    status = "ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" if len(selected) >= args.target_docs else "BLOCKED_INSUFFICIENT_ELIGIBLE_TOPICS"
    for topic_index, topic in enumerate(selected):
        split = split_for_index(topic_index)
        for page in topic["pages"]:
            lang = page["language"]; qid = topic["qid"]
            stem = f"{topic_index:02d}_{lang}_{qid}_{page['revid']}"
            raw_path = raw_dir / f"{stem}.txt"; norm_path = norm_dir / f"{stem}.txt"
            raw_bytes = page["raw_extract"].encode("utf-8"); norm_bytes = page["normalized"].encode("utf-8")
            raw_path.write_bytes(raw_bytes); norm_path.write_bytes(norm_bytes)
            doc_id = f"wikimedia-{lang}-{qid}-r{page['revid']}"
            revision_url = page["canonicalurl"] + ("&" if "?" in page["canonicalurl"] else "?") + f"oldid={page['revid']}"
            row = {
                "corpus_id": "wikimedia_parallel_concepts_20260730", "document_id": doc_id,
                "split": split, "class_label": lang, "language": lang,
                "family": "plaintext_encyclopedic", "path": norm_path.as_posix(),
                "sha256": sha256_bytes(norm_bytes), "encoding": "utf-8", "license": LICENSE_NAME,
                "author_id": f"collective:{lang}wiki-contributors", "work_id": qid,
                "date_band": "contemporary_dynamic_source_frozen_revision",
                "notes": f"topic={topic['english_title']}; page={page['title']}; pageid={page['pageid']}; revid={page['revid']}; revision_timestamp={page['timestamp']}; source={revision_url}; license_url={LICENSE_URL}; raw_sha256={sha256_bytes(raw_bytes)}; normalized_units={page['normalized_units']}; deterministic MediaWiki plaintext extract",
                "normalized": page["normalized"], "normalized_units": page["normalized_units"],
                "raw_path": raw_path.as_posix(), "raw_sha256": sha256_bytes(raw_bytes),
                "source_url": revision_url, "page_title": page["title"], "pageid": page["pageid"],
                "revid": page["revid"], "revision_timestamp": page["timestamp"],
                "topic_index": topic_index, "topic_en": topic["english_title"],
            }
            rows.append(row); events.append({k: v for k, v in row.items() if k != "normalized"})
    duplicate_findings = screen_duplicates(rows)
    if duplicate_findings: status = "BLOCKED_DUPLICATE_OR_NEAR_DUPLICATE"
    fields = ["corpus_id", "document_id", "split", "class_label", "language", "family", "path", "sha256", "encoding", "license", "author_id", "work_id", "date_band", "notes"]
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: row[key] for key in fields})
    (out / "rejections.json").write_text(json.dumps(rejections, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "acquired_pages.json").write_text(json.dumps(events, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "duplicate_screen.json").write_text(json.dumps(duplicate_findings, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {lang: 0 for lang, _, _ in LANGUAGES}; split_counts = {lang: {"train": 0, "dev": 0, "test": 0} for lang, _, _ in LANGUAGES}
    for row in rows: counts[row["language"]] += 1; split_counts[row["language"]][row["split"]] += 1
    summary = {
        "programme": "compression-transfer-v0.1", "panel": "stage1_period_tolerant_wikimedia_parallel_concepts",
        "status": status, "voynich_accessed": False,
        "selection_rule": "first 12 frozen candidate English topics with sitelinks and >=4096 normalized units in every registered language",
        "candidate_titles_sha256": sha256_bytes(json.dumps(CANDIDATE_EN_TITLES, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        "languages": [lang for lang, _, _ in LANGUAGES], "target_documents_per_language": args.target_docs,
        "minimum_units": args.min_units, "selected_topics": [{"index": i, "qid": t["qid"], "english_title": t["english_title"]} for i, t in enumerate(selected)],
        "counts": counts, "split_counts": split_counts, "rejected_candidate_count": len(rejections),
        "duplicate_findings_count": len(duplicate_findings), "licence": LICENSE_NAME, "licence_url": LICENSE_URL,
        "author_holdout_available": False, "work_holdout_available": True,
        "scientific_limitations": [
            "Wikipedia articles are collective-authorship documents; author holdout is unavailable.",
            "Latin Wikipedia is modern Latin and does not satisfy the separately required medieval/early-modern Latin panel.",
            "This acquisition is period-tolerant engineering material only until the historical-domain panel is sourced and frozen.",
        ],
    }
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["scientific_payload_sha256"] = sha256_bytes(payload)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
