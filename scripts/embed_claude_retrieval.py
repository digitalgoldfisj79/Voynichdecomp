#!/usr/bin/env python3
"""Embed staged Claude retrieval chunks via Supabase Edge + GitHub OIDC.

No repository secret is required. The Edge function validates the GitHub Actions
OIDC issuer, repository, and main-branch ref. Source text is fetched only after
OIDC authentication and is never written to the repository or Actions artifacts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from urllib.parse import quote

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EDGE_URL = os.environ.get(
    "CLAUDE_RETRIEVAL_EDGE_URL",
    "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/claude-voynich-search-v01",
)
AUDIENCE = "claude-voynich-index"
MODEL_NAME = "Supabase/gte-small"


def github_oidc_token() -> str:
    req_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    req_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not req_url or not req_token:
        raise RuntimeError("GitHub OIDC environment is unavailable; permissions.id-token must be write")
    sep = "&" if "?" in req_url else "?"
    r = requests.get(
        f"{req_url}{sep}audience={quote(AUDIENCE)}",
        headers={"Authorization": f"Bearer {req_token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["value"]


def edge_post(payload: dict, timeout: int = 90) -> dict:
    # Get a fresh short-lived OIDC token for every Edge call so long jobs do not
    # fail because a previously-issued JWT expired.
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


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-size", type=int, default=128)
    ap.add_argument("--encode-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=100000)
    args = ap.parse_args()

    fetch_size = max(1, min(args.fetch_size, 256))
    encode_size = max(1, min(args.encode_size, 128))

    torch.set_num_threads(max(1, os.cpu_count() or 2))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embedded_total = 0
    for batch_no in range(1, args.max_batches + 1):
        payload = edge_post({"op": "fetch_embedding_batch", "limit": fetch_size})
        rows = payload.get("rows") or []
        if not rows:
            df = edge_post({"op": "refresh_chunk_df"})
            status = edge_post({"op": "retrieval_status"})
            print(f"complete embedded_total={embedded_total} token_df={df.get('tokens')} status={status.get('status')}")
            return 0

        vectors: list[dict] = []
        for start in range(0, len(rows), encode_size):
            part = rows[start : start + encode_size]
            texts = [str(r.get("text") or "") for r in part]
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                out = model(**enc)
                pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
                normed = F.normalize(pooled, p=2, dim=1)
            arr = normed.cpu().numpy()
            for row, vec in zip(part, arr):
                # PostgreSQL pgvector accepts the canonical '[v1,v2,...]' text form.
                vector_text = "[" + ",".join(f"{float(v):.9g}" for v in vec) + "]"
                vectors.append({"id": row["id"], "embedding": vector_text})

        result = edge_post(
            {"op": "set_embedding_batch", "rows": vectors, "model": "gte-small"},
            timeout=120,
        )
        updated = int(result.get("updated") or 0)
        if updated != len(rows):
            raise RuntimeError(f"setter mismatch: fetched={len(rows)} updated={updated}")
        embedded_total += updated

        if batch_no == 1 or batch_no % 10 == 0:
            status = edge_post({"op": "retrieval_status"})
            print(
                f"batch={batch_no} embedded_total={embedded_total} "
                f"pending={status.get('status', {}).get('pending_embeddings')}"
            )
        else:
            print(f"batch={batch_no} embedded_total={embedded_total}")
        sys.stdout.flush()
        time.sleep(0.05)

    print(f"stopped at max_batches={args.max_batches}; embedded_total={embedded_total}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
