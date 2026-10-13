#!/usr/bin/env python3
"""Turn velinska posts.jsonl/comments.jsonl into citation-ledger rows.

Mirrors the existing dulov_ (Blogger) conventions in public.sources /
public.source_passages. Writes ledger_sources.jsonl and ledger_passages.jsonl
(no embeddings; those are added by embed step).
"""
import hashlib, json, re, sys
from datetime import datetime

src_dir, out_dir = sys.argv[1], sys.argv[2]
PREFIX = "velinska_"
VENUE = "Ellie Velinska — ellievelinska.blogspot.com"
AUTHOR = "Ellie Velinska"
LICENSE = "fair use; research/citation only; original copyright retained by author"
MAXLEN = 1500
FETCHED_AT = "2026-10-13T17:24:24+00:00"   # Wayback backfill run 38434621959


def source_id(p):
    s = re.sub(r"[^a-z0-9]+", "_", p["slug"].lower()).strip("_")
    return f"{PREFIX}{p['year']}_{p['month']:02d}_{s}_html"


def chunks(text):
    """Greedy pack of paragraphs into <=MAXLEN chars; long paragraphs split at sentences."""
    paras = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n+", text)]
    pieces = []
    for para in filter(None, paras):
        if len(para) <= MAXLEN:
            pieces.append(para)
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > MAXLEN:            # pathological: no sentence breaks
                if cur: pieces.append(cur); cur = ""
                pieces.append(sent[:MAXLEN]); sent = sent[MAXLEN:]
            if len(cur) + len(sent) + 1 > MAXLEN:
                pieces.append(cur); cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur: pieces.append(cur)
    out, cur = [], ""
    for pc in pieces:
        if cur and len(cur) + len(pc) + 1 > MAXLEN:
            out.append(cur); cur = pc
        else:
            cur = f"{cur} {pc}".strip()
    if cur: out.append(cur)
    return out


def year_of(s):
    m = re.search(r"\b(20\d\d)\b", s or "")
    return int(m[1]) if m else None


posts = [json.loads(l) for l in open(f"{src_dir}/posts.jsonl")]
comments, seen = [], set()
for l in open(f"{src_dir}/comments.jsonl"):
    c = json.loads(l)
    k = (c["post_url"], c["author"], c["text"])
    if k in seen:          # double-posted comments (12 in run 38434621959)
        continue
    seen.add(k); comments.append(c)
by_url = {}
for c in comments:
    by_url.setdefault(c["post_url"], []).append(c)

srows, prows, skipped = [], [], []
for p in sorted(posts, key=lambda p: (p["year"], p["month"], p["slug"])):
    url = p["url"].replace("http://", "https://", 1)
    body = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in p["body_text"].split("\n")).strip()
    body = re.sub(r"\n{2,}", "\n", body)
    sid = source_id(p)
    pc = chunks(body)
    cm = by_url.get(p["url"], [])
    if "voynich" not in (p["title"] + " " + body).lower():
        skipped.append({"source_id": sid, "title": p["title"], "words": p["word_count"],
                        "images": len(p["images"]), "reason": "not Voynich-related (user scope decision)"})
        continue
    if not pc and not cm:
        skipped.append({"source_id": sid, "title": p["title"], "words": p["word_count"],
                        "images": len(p["images"]), "reason": "no text (image-only)"})
        continue
    wb = f"https://web.archive.org/web/{p['capture_timestamp']}/{p['url']}"
    srows.append({
        "source_id": sid, "title": p["title"], "source_url": url,
        "source_citation": f"Velinska, E. ({p['year']}). {p['title']}. {VENUE} ({url}). "
                           f"Blog offline; text recovered from Internet Archive capture {wb} "
                           f"(via {p['source'].replace('_', ' ')}).",
        "source_type": "independent_blog", "fetched_at": FETCHED_AT,
        "sha256": hashlib.sha256(body.encode()).hexdigest(), "license": LICENSE,
        "body": body, "word_count": len(body.split()), "publication_year": p["year"],
        "venue": VENUE, "peer_review_status": "independent_blog",
    })
    i = 0
    for t in pc:
        prows.append({"source_id": sid, "paragraph_index": i, "text": t,
                      "passage_author": AUTHOR, "passage_year": p["year"], "passage_url": url})
        i += 1
    for c in cm:
        for t in chunks(c["text"]):
            if len(t) < 5:
                continue
            prows.append({"source_id": sid, "paragraph_index": i, "text": t,
                          "passage_author": c["author"], "passage_year": year_of(c["date"]) or p["year"],
                          "passage_url": url})
            i += 1

with open(f"{out_dir}/ledger_sources.jsonl", "w") as f:
    for r in srows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{out_dir}/ledger_passages.jsonl", "w") as f:
    for r in prows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{out_dir}/ledger_skipped.json", "w") as f:
    json.dump(skipped, f, indent=1, ensure_ascii=False)
print(f"sources={len(srows)} passages={len(prows)} "
      f"(post={sum(r['passage_author']==AUTHOR for r in prows)}, "
      f"comment={sum(r['passage_author']!=AUTHOR for r in prows)}) skipped={len(skipped)}")
