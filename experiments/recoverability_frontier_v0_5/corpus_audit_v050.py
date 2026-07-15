#!/usr/bin/env python3
"""Stage-0 corpus audit for recoverability frontier v0.5.0.

Downloads only files pinned by immutable Git commit, computes hashes, extracts
UD sentence text, and reports corpus/partition sizes and licence provenance.
No scientific cipher experiment is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CORPORA = [
    {
        "language": "English",
        "iso": "en",
        "typology": "fusional, mostly analytic; Latin script",
        "repo": "UniversalDependencies/UD_English-EWT",
        "commit": "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b",
        "prefix": "en_ewt",
    },
    {
        "language": "German",
        "iso": "de",
        "typology": "fusional; compounding; Latin script",
        "repo": "UniversalDependencies/UD_German-GSD",
        "commit": "ce54dbe9c6a5640c93e9952f069f582f6cd1f9fc",
        "prefix": "de_gsd",
    },
    {
        "language": "Finnish",
        "iso": "fi",
        "typology": "agglutinative; rich case; Latin script",
        "repo": "UniversalDependencies/UD_Finnish-TDT",
        "commit": "bfaae13719f249573d940edda6a0d7aa8eec620f",
        "prefix": "fi_tdt",
    },
    {
        "language": "Turkish",
        "iso": "tr",
        "typology": "agglutinative; vowel harmony; Latin script",
        "repo": "UniversalDependencies/UD_Turkish-IMST",
        "commit": "0c939115d8277ecfb39e1bbc3f066b1852ab5ddc",
        "prefix": "tr_imst",
    },
    {
        "language": "Hebrew",
        "iso": "he",
        "typology": "Semitic root-pattern morphology; abjad",
        "repo": "UniversalDependencies/UD_Hebrew-HTB",
        "commit": "dd6d2133e6b9373e7e4888a1b33724df38e2e549",
        "prefix": "he_htb",
    },
    {
        "language": "Arabic",
        "iso": "ar",
        "typology": "Semitic root-pattern morphology; abjad",
        "repo": "UniversalDependencies/UD_Arabic-PADT",
        "commit": "dfb6b4c547f1fe10f1857b39e44de3f86c47a2fe",
        "prefix": "ar_padt",
    },
]

PARTITIONS = ("train", "dev", "test")
META_FILES = ("README.md", "LICENSE.txt", "LICENSE", "LICENSE.md")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Voynichdecomp-v050-audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def maybe_fetch(url: str) -> bytes | None:
    try:
        return fetch(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def sentence_texts(raw: bytes) -> list[str]:
    text = raw.decode("utf-8")
    found = []
    for line in text.splitlines():
        if line.startswith("# text = "):
            value = line[len("# text = "):].strip()
            if value:
                found.append(value)
    return found


def token_count_from_conllu(raw: bytes) -> int:
    total = 0
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        ident = line.split("\t", 1)[0]
        if ident.isdigit():
            total += 1
    return total


def licence_hint(text: str) -> str | None:
    patterns = [
        r"CC BY-NC-SA [0-9.]+",
        r"CC BY-SA [0-9.]+",
        r"CC BY [0-9.]+",
        r"Creative Commons Attribution[^.\n]*",
        r"GNU (?:Lesser )?General Public License[^.\n]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ud-v050"))
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for corpus in CORPORA:
        owner_repo = corpus["repo"]
        commit = corpus["commit"]
        base = f"https://raw.githubusercontent.com/{owner_repo}/{commit}"
        record: dict[str, Any] = dict(corpus)
        record["files"] = {}
        record["metadata"] = {}

        for name in META_FILES:
            data = maybe_fetch(f"{base}/{name}")
            if data is None:
                continue
            local = args.cache_dir / f"{corpus['iso']}-{name.replace('/', '_')}"
            local.write_bytes(data)
            decoded = data.decode("utf-8", errors="replace")
            record["metadata"][name] = {
                "sha256": sha256(data),
                "bytes": len(data),
                "licence_hint": licence_hint(decoded),
            }

        for split in PARTITIONS:
            filename = f"{corpus['prefix']}-ud-{split}.conllu"
            url = f"{base}/{filename}"
            data = fetch(url)
            local = args.cache_dir / filename
            local.write_bytes(data)
            texts = sentence_texts(data)
            record["files"][split] = {
                "filename": filename,
                "url": url,
                "sha256": sha256(data),
                "bytes": len(data),
                "sentences": len(texts),
                "tokens": token_count_from_conllu(data),
                "characters_in_text_fields": sum(len(x) for x in texts),
            }

        record["total_sentences"] = sum(x["sentences"] for x in record["files"].values())
        record["total_tokens"] = sum(x["tokens"] for x in record["files"].values())
        record["total_characters"] = sum(
            x["characters_in_text_fields"] for x in record["files"].values()
        )
        records.append(record)
        print(
            "V050_CORPUS",
            corpus["iso"],
            record["total_sentences"],
            record["total_tokens"],
            record["total_characters"],
            flush=True,
        )

    payload = {
        "programme": "recoverability-frontier-v0.5.0-stage0-corpus-audit",
        "corpora": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    digest = sha256(args.output.read_bytes())
    print("V050_CORPUS_AUDIT_SHA256", digest, flush=True)


if __name__ == "__main__":
    main()
