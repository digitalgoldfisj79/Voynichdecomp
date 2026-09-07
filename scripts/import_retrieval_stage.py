#!/usr/bin/env python3
"""Import an encrypted retrieval corpus into Supabase using GitHub OIDC.

The staging object is ciphertext only. Its URL and AES-GCM key are returned by
Supabase only after GitHub OIDC validation. Plaintext exists only in runner
memory and is never committed or uploaded as an artifact.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import sys
from urllib.parse import quote

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

EDGE_URL = os.environ.get(
    "CLAUDE_RETRIEVAL_EDGE_URL",
    "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/claude-voynich-search-v01",
)
AUDIENCE = "claude-voynich-index"


def github_oidc_token() -> str:
    req_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    req_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not req_url or not req_token:
        raise RuntimeError("GitHub OIDC environment unavailable; permissions.id-token must be write")
    sep = "&" if "?" in req_url else "?"
    r = requests.get(
        f"{req_url}{sep}audience={quote(AUDIENCE)}",
        headers={"Authorization": f"Bearer {req_token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["value"]


def edge_post(payload: dict, timeout: int = 120) -> dict:
    token = github_oidc_token()
    r = requests.post(
        EDGE_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"Edge {r.status_code}: {r.text[:1200]}")
    return r.json()


def b64url_decode(value: str) -> bytes:
    value = value.strip()
    value += "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value.encode())


def download_cipher(url: str, expected_bytes: int, expected_sha256: str) -> bytes:
    candidates = [url]
    if "download=1" not in url:
        candidates.insert(0, url + ("&" if "?" in url else "?") + "download=1")
    last_error = None
    for candidate in candidates:
        try:
            h = hashlib.sha256()
            chunks = []
            size = 0
            with requests.get(candidate, stream=True, allow_redirects=True, timeout=(30, 240)) as r:
                r.raise_for_status()
                for block in r.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    chunks.append(block)
                    h.update(block)
                    size += len(block)
            digest = h.hexdigest()
            if size == expected_bytes and digest == expected_sha256:
                return b"".join(chunks)
            last_error = RuntimeError(
                f"stage verification failed: bytes={size}/{expected_bytes} sha256_match={digest == expected_sha256}"
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not download verified staging object: {last_error}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=400)
    args = ap.parse_args()
    batch_size = max(1, min(args.batch_size, 1000))

    info = edge_post({"op": "stage_info"})
    stage = info.get("stage") or {}
    required = ["download_url", "key_b64", "aad", "cipher_bytes", "cipher_sha256", "gzip_bytes", "gzip_sha256"]
    missing = [k for k in required if not stage.get(k)]
    if missing:
        raise RuntimeError(f"stage metadata missing fields: {missing}")

    cipher = download_cipher(
        str(stage["download_url"]),
        int(stage["cipher_bytes"]),
        str(stage["cipher_sha256"]),
    )
    key = b64url_decode(str(stage["key_b64"]))
    aad = str(stage["aad"]).encode()
    if len(cipher) < 29:
        raise RuntimeError("ciphertext is too short")
    gzip_bytes = AESGCM(key).decrypt(cipher[:12], cipher[12:], aad)
    if len(gzip_bytes) != int(stage["gzip_bytes"]):
        raise RuntimeError("decrypted gzip byte count mismatch")
    if hashlib.sha256(gzip_bytes).hexdigest() != str(stage["gzip_sha256"]):
        raise RuntimeError("decrypted gzip SHA-256 mismatch")

    total = 0
    batches = 0
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(gzip_bytes), mode="rb") as gz:
        for raw in gz:
            if not raw.strip():
                continue
            rows.append(json.loads(raw))
            if len(rows) >= batch_size:
                result = edge_post({"op": "ingest_chunks", "rows": rows}, timeout=180)
                upserted = int(result.get("upserted") or 0)
                if upserted != len(rows):
                    raise RuntimeError(f"upsert mismatch: sent={len(rows)} upserted={upserted}")
                total += upserted
                batches += 1
                if batches == 1 or batches % 10 == 0:
                    print(f"ingest batch={batches} total={total}")
                    sys.stdout.flush()
                rows = []
        if rows:
            result = edge_post({"op": "ingest_chunks", "rows": rows}, timeout=180)
            upserted = int(result.get("upserted") or 0)
            if upserted != len(rows):
                raise RuntimeError(f"upsert mismatch: sent={len(rows)} upserted={upserted}")
            total += upserted
            batches += 1

    df = edge_post({"op": "refresh_chunk_df"}, timeout=240)
    status = edge_post({"op": "retrieval_status"})
    print(f"complete ingested={total} batches={batches} token_df={df.get('tokens')} status={status.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
