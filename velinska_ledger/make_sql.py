#!/usr/bin/env python3
"""Emit batched INSERT statements for the velinska ledger rows.

sources: ON CONFLICT DO NOTHING; passages: upsert on (source_id, paragraph_index).
Text is dollar-quoted; text_tsv is filled by trg_passage_tsv_update.
Rollback: DELETE FROM sources WHERE source_id LIKE 'velinska\\_%' (passages cascade).
"""
import json, os, sys

D, out, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12
os.makedirs(out, exist_ok=True)
TAG = "$vlk$"


def q(v):
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    s = str(v)
    assert TAG not in s
    return f"{TAG}{s}{TAG}"


S = [json.loads(l) for l in open(f"{D}/ledger_sources.jsonl")]
cols = ["source_id", "title", "source_url", "source_citation", "source_type", "fetched_at", "sha256",
        "license", "body", "word_count", "publication_year", "venue", "peer_review_status"]
files = []
for i in range(0, len(S), 25):
    vals = ",\n".join("(" + ",".join(q(r[c]) for c in cols) + ")" for r in S[i:i + 25])
    p = f"{out}/s{i // 25:03d}.sql"
    open(p, "w").write(f"insert into sources ({','.join(cols)}) values\n{vals}\non conflict (source_id) do nothing;\n")
    files.append(p)

P = [json.loads(l) for l in open(f"{D}/ledger_passages_embedded.jsonl")]
pc = ["source_id", "paragraph_index", "text", "passage_author", "passage_year", "passage_url"]
for i in range(0, len(P), batch):
    vals = ",\n".join("(" + ",".join(q(r[c]) for c in pc) + ",'[" + ",".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in r["embedding"]) + "]'::vector)"
                      for r in P[i:i + batch])
    p = f"{out}/p{i // batch:03d}.sql"
    open(p, "w").write(f"insert into source_passages ({','.join(pc)},embedding) values\n{vals}\n"
                       "on conflict (source_id, paragraph_index) do update set text=excluded.text, "
                       "passage_author=excluded.passage_author, passage_year=excluded.passage_year, "
                       "passage_url=excluded.passage_url, embedding=excluded.embedding;\n")
    files.append(p)
print(len(files), "files;", "max bytes", max(os.path.getsize(f) for f in files))
