#!/usr/bin/env python3
"""Discover and freeze a historical-domain Wikisource panel.

The script uses Wikidata only to locate dated, author-attributed works with a
sitelink to each registered language Wikisource. It retrieves exact revisions
and admits only works with at least 4,096 normalized units. No text is
synthesized, translated, or silently substituted.
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
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

USER_AGENT = "VoynichCompressionTransfer/0.1 (historical corpus discovery; github.com/digitalgoldfisj79/Voynichdecomp)"
LICENSE_NAME = "Wikisource CC BY-SA 4.0; underlying work public-domain candidate by registered date cutoff"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
TARGET_DOCS = 12
MIN_UNITS = 4096
DATE_CUTOFF_YEAR = 1800
LANGUAGES = [
    ("en", "enwikisource", "English"), ("de", "dewikisource", "German"),
    ("fi", "fiwikisource", "Finnish"), ("tr", "trwikisource", "Turkish"),
    ("el", "elwikisource", "Greek"), ("he", "hewikisource", "Hebrew"),
    ("ar", "arwikisource", "Arabic"), ("la", "lawikisource", "Latin"),
]


def request_json(url: str, attempts: int = 6, pause: float = 0.4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json, application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc; time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def api_url(host: str, params: dict[str, Any]) -> str:
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    return f"https://{host}/w/api.php?{urllib.parse.urlencode(params, doseq=True)}"


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).replace("\u00a0", " ")).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entity_id(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def discover_candidates(lang: str, cutoff_year: int, limit: int = 600) -> list[dict[str, Any]]:
    site = f"https://{lang}.wikisource.org/"
    query = f'''SELECT DISTINCT ?work ?article ?pageTitle ?author ?authorLabel ?date WHERE {{
  ?article schema:about ?work ; schema:isPartOf <{site}> ; schema:name ?pageTitle .
  ?work wdt:P50 ?author .
  {{ ?work wdt:P577 ?date . }} UNION {{ ?work wdt:P571 ?date . }}
  FILTER(?date < "{cutoff_year + 1}-01-01T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}} ORDER BY ?work ?article ?author LIMIT {int(limit)}'''
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    payload = request_json(url)
    out: list[dict[str, Any]] = []
    for b in payload.get("results", {}).get("bindings", []):
        try:
            out.append({"work_id": entity_id(b["work"]["value"]), "article_url": b["article"]["value"],
                "page_title": b["pageTitle"]["value"], "author_id": entity_id(b["author"]["value"]),
                "author_label": b.get("authorLabel", {}).get("value", entity_id(b["author"]["value"])),
                "date": b["date"]["value"]})
        except KeyError:
            continue
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in out: unique.setdefault((row["work_id"], row["page_title"]), row)
    return sorted(unique.values(), key=lambda r: (r["work_id"], r["page_title"], r["author_id"]))


def fetch_extract(host: str, title: str) -> dict[str, Any]:
    payload = request_json(api_url(host, {"action": "query", "redirects": 1,
        "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "plain",
        "rvprop": "ids|timestamp", "inprop": "url", "titles": title}))
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing"): raise ValueError(f"missing {host}:{title}")
    page = pages[0]; revisions = page.get("revisions") or []
    if not revisions: raise ValueError(f"no revisions {host}:{title}")
    return {"pageid": int(page["pageid"]), "title": str(page["title"]),
        "canonicalurl": str(page.get("canonicalurl") or ""), "revid": int(revisions[0]["revid"]),
        "timestamp": str(revisions[0]["timestamp"]), "extract": str(page.get("extract", ""))}


def fetch_subpage_titles(host: str, root_title: str, limit: int = 50) -> list[str]:
    titles: list[str] = []; continuation: str | None = None
    while len(titles) < limit:
        params: dict[str, Any] = {"action": "query", "list": "allpages", "apnamespace": 0,
            "apprefix": root_title.rstrip("/") + "/", "aplimit": min(50, limit - len(titles))}
        if continuation: params["apcontinue"] = continuation
        payload = request_json(api_url(host, params))
        titles.extend(row["title"] for row in payload.get("query", {}).get("allpages", []))
        continuation = payload.get("continue", {}).get("apcontinue")
        if not continuation: break
    return titles


def fetch_work_text(lang: str, root_title: str, min_units: int) -> dict[str, Any]:
    host = f"{lang}.wikisource.org"; root = fetch_extract(host, root_title); components = [root]
    normalized = normalize_surface(root["extract"])
    if len(normalized) < min_units:
        for subtitle in fetch_subpage_titles(host, root["title"], 50):
            try: components.append(fetch_extract(host, subtitle))
            except Exception: continue
            normalized = normalize_surface("\n".join(c["extract"] for c in components))
            if len(normalized) >= min_units: break
    return {"host": host, "components": components, "normalized": normalized, "units": len(normalized)}


def split_for_index(index: int) -> str:
    return "train" if index < 8 else ("dev" if index < 10 else "test")


def near_duplicate_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []; grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["language"]].append(row)
    for lang, docs in grouped.items():
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                ratio = SequenceMatcher(None, a["normalized"][:20000], b["normalized"][:20000], autojunk=False).ratio()
                if a["sha256"] == b["sha256"] or ratio >= 0.85:
                    findings.append({"language": lang, "document_a": a["document_id"],
                        "document_b": b["document_id"], "exact_sha_duplicate": a["sha256"] == b["sha256"],
                        "sequence_ratio_capped_20000": ratio})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/stage1_historical_wikisource")
    parser.add_argument("--target-docs", type=int, default=TARGET_DOCS); parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    parser.add_argument("--cutoff-year", type=int, default=DATE_CUTOFF_YEAR); args = parser.parse_args()
    out = Path(args.output); raw_dir = out / "source_raw"; norm_dir = out / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True); norm_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []; language_status: dict[str, Any] = {}; discovery_log: dict[str, Any] = {}
    for lang, site, label in LANGUAGES:
        accepted: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
        try: candidates = discover_candidates(lang, args.cutoff_year)
        except Exception as exc:
            language_status[lang] = {"status": "BLOCKED_DISCOVERY_ERROR", "error": repr(exc), "accepted": 0}
            discovery_log[lang] = {"candidates": [], "rejected": []}; continue
        for candidate in candidates:
            if len(accepted) >= args.target_docs: break
            try:
                work = fetch_work_text(lang, candidate["page_title"], args.min_units)
                if work["units"] < args.min_units:
                    rejected.append({**candidate, "reason": "short", "units": work["units"]}); continue
                authors = {r["author_id"] for r in accepted}; remaining = args.target_docs - len(accepted)
                if remaining == 1 and len(authors) < 2 and candidate["author_id"] in authors:
                    rejected.append({**candidate, "reason": "reserved_for_second_author", "units": work["units"]}); continue
                accepted.append({**candidate, **work})
            except Exception as exc: rejected.append({**candidate, "reason": "fetch_error", "error": repr(exc)})
        authors = sorted({row["author_id"] for row in accepted})
        state = "ELIGIBLE" if len(accepted) >= args.target_docs and len(authors) >= 2 else "BLOCKED_INSUFFICIENT_ELIGIBLE_WORKS"
        language_status[lang] = {"status": state, "accepted": len(accepted), "authors": authors,
            "candidate_count": len(candidates), "rejected_count": len(rejected)}
        discovery_log[lang] = {"accepted": [], "rejected": rejected}
        for idx, item in enumerate(accepted):
            components = item["components"]; raw_text = "\n\n".join(c["extract"] for c in components); normalized = item["normalized"]
            component_signature = sha256(json.dumps([{"pageid": c["pageid"], "title": c["title"], "revid": c["revid"], "timestamp": c["timestamp"]} for c in components], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            stem = f"{idx:02d}_{lang}_{item['work_id']}_{component_signature[:12]}"; raw_path = raw_dir / f"{stem}.txt"; norm_path = norm_dir / f"{stem}.txt"
            raw_bytes = raw_text.encode("utf-8"); norm_bytes = normalized.encode("utf-8"); raw_path.write_bytes(raw_bytes); norm_path.write_bytes(norm_bytes)
            root = components[0]; revision_url = root["canonicalurl"] + ("&" if "?" in root["canonicalurl"] else "?") + f"oldid={root['revid']}"
            row = {"corpus_id": "wikisource_historical_domain_20260730", "document_id": f"wikisource-{lang}-{item['work_id']}-{component_signature[:12]}",
                "split": split_for_index(idx), "class_label": lang, "language": lang, "family": "historical_plaintext",
                "path": norm_path.as_posix(), "sha256": sha256(norm_bytes), "encoding": "utf-8", "license": LICENSE_NAME,
                "author_id": item["author_id"], "work_id": item["work_id"],
                "date_band": f"dated_not_later_than_{args.cutoff_year};wikidata_date={item['date']}",
                "notes": f"author_label={item['author_label']}; root_page={root['title']}; source={revision_url}; component_count={len(components)}; component_signature={component_signature}; license_url={LICENSE_URL}; raw_sha256={sha256(raw_bytes)}; normalized_units={item['units']}; selection based only on Wikidata date/author/sitelink and registered length threshold",
                "normalized": normalized, "normalized_units": item["units"], "root_title": root["title"],
                "root_revision_url": revision_url, "components": [{"pageid": c["pageid"], "title": c["title"], "revid": c["revid"], "timestamp": c["timestamp"]} for c in components]}
            rows.append(row); discovery_log[lang]["accepted"].append({k: v for k, v in row.items() if k != "normalized"})
    duplicates = near_duplicate_findings(rows); eligible = [lang for lang, state in language_status.items() if state["status"] == "ELIGIBLE"]
    overall = "ACQUIRED_NOT_SCIENTIFICALLY_EVALUATED" if len(eligible) == len(LANGUAGES) and not duplicates else "BLOCKED"
    if duplicates: overall = "BLOCKED_DUPLICATE_OR_NEAR_DUPLICATE"
    fields = ["corpus_id", "document_id", "split", "class_label", "language", "family", "path", "sha256", "encoding", "license", "author_id", "work_id", "date_band", "notes"]
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: row[key] for key in fields})
    (out / "discovery_log.json").write_text(json.dumps(discovery_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "duplicate_screen.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"programme": "compression-transfer-v0.1", "panel": "stage1_historical_domain_wikisource",
        "status": overall, "voynich_accessed": False, "registered_cutoff_year": args.cutoff_year,
        "minimum_units": args.min_units, "target_documents_per_language": args.target_docs,
        "language_status": language_status, "eligible_languages": eligible,
        "duplicate_findings_count": len(duplicates),
        "selection_rule": "Wikidata dated and author-attributed Wikisource sitelinks, deterministic QID order, first 12 works >=4096 units with >=2 authors",
        "scientific_boundary": "A blocked language remains blocked; no synthetic substitution or post-distance source addition is permitted."}
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"); summary["scientific_payload_sha256"] = sha256(payload)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
