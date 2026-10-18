#!/usr/bin/env python3
"""Rebuild npcompl33t's Clementine Vulgate token stream (his stated tokenisation) and check it against his
reported corpus size (612,195 tokens; 46,476 types). usage: python3 -I s36_build_vulgate.py SRC_DIR OUT_JSON"""
import json, re, sys, unicodedata
from collections import Counter
src, out = sys.argv[1], sys.argv[2]
books = [l.split('/')[0] for l in open(f"{src}/data.txt", encoding="cp1252") if l.strip() and not l.startswith('#')]
toks, spans = [], []
for b in books:
    start = len(toks)
    for line in open(f"{src}/{b}.lat", encoding="cp1252"):
        line = re.sub(r"^\s*\d+:\d+\s*", "", line)
        line = unicodedata.normalize("NFC", line).lower()
        toks += re.findall(r"[^\W\d_]+", line)
    spans.append([b, start, len(toks)])
c = Counter(toks)
print(len(books), "books", len(toks), "tokens", len(c), "types", sum(1 for v in c.values() if v == 1), "hapax")
json.dump({"books": spans, "tokens": toks}, open(out, "w"))
