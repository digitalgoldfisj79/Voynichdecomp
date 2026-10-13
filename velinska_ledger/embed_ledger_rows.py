#!/usr/bin/env python3
"""Embed velinska_ledger/ledger_passages.jsonl with Supabase/gte-small.

Same model, mean pooling and L2 normalisation as scripts/embed_claude_retrieval.py
and the ingest-passages edge function (Supabase.ai gte-small, mean_pool, normalize).
Also embeds calibration.json (passages already in source_passages) so the
vectors can be checked against the stored ones before anything is inserted.
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Supabase/gte-small"
D = sys.argv[1] if len(sys.argv) > 1 else "velinska_ledger"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()


def embed(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        out += F.normalize(mean_pool(h, enc["attention_mask"]), p=2, dim=1).tolist()
    return [[round(x, 6) for x in v] for v in out]


rows = [json.loads(l) for l in open(f"{D}/ledger_passages.jsonl")]
for r, v in zip(rows, embed([r["text"] for r in rows])):
    r["embedding"] = v
with open(f"{D}/ledger_passages_embedded.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

cal = json.load(open(f"{D}/calibration.json"))
for c, v in zip(cal, embed([c["text"] for c in cal])):
    c["embedding"] = v
json.dump(cal, open(f"{D}/calibration_embedded.json", "w"), ensure_ascii=False)
print(f"embedded {len(rows)} passages + {len(cal)} calibration texts, dim={len(rows[0]['embedding'])}")
