#!/usr/bin/env python3
"""Stage 14: shared features for T1b-T5 -> data/s14_features.pkl."""
import os, pickle, re, sys
from collections import Counter
import numpy as np, pandas as pd
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data")
F = pd.read_csv(os.path.join(ROOT, "data", "folio_bifolium_canonical.csv"), index_col=0)
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"].copy()
for c in ("folio", "word"): meta[c] = meta[c].astype(str)
meta["line_num"] = pd.to_numeric(meta.line_num, errors="coerce"); meta["word_index"] = meta.word_index.astype(int)
folios = list(F.index)
counts, strings = [], []
for f in folios:
    sub = meta[meta.folio == f].sort_values(["line_num", "word_index"])
    c = Counter(); lines = []
    for _, ln in sub.groupby("line_num", sort=True):
        ws = list(ln.word); lines.append(" ".join(ws))
        for w in ws:
            s = "^" + w + "$"; c.update(("w", a, b) for a, b in zip(s, s[1:]))
        s = "".join(ws); c.update(("l", a, b) for a, b in zip(s, s[1:]))
    counts.append(c); strings.append("\n".join(lines))
vocab = sorted(set().union(*counts)); vi = {v: i for i, v in enumerate(vocab)}
M = np.zeros((len(folios), len(vocab)))
for i, c in enumerate(counts):
    for k, v in c.items(): M[i, vi[k]] = v
wmask = np.array([v[0] == "w" for v in vocab])
def key(f):
    m = re.match(r"f(\d+)([rv])(\d*)", f); return (int(m.group(1)), 0 if m.group(2) == "r" else 1, int(m.group(3) or 0))
order = sorted(folios, key=key); pos = {f: i for i, f in enumerate(order)}
num = {f: key(f)[0] for f in folios}; side = {f: key(f)[1] for f in folios}
nums = sorted(set(num.values()))
nextnum = {n: (nums[i + 1] if i + 1 < len(nums) else None) for i, n in enumerate(nums)}
facing = np.zeros((len(folios), len(folios)), bool)
for i, a in enumerate(folios):
    for j, b in enumerate(folios):
        if side[a] == 1 and side[b] == 0 and nextnum[num[a]] == num[b]:
            facing[i, j] = facing[j, i] = True
F["scribe"] = F.scribe.fillna("unknown").astype(str)
F["quire_l"] = F.sheet.str.split(":").str[0]
F["leaf_id"] = F.sheet + "|" + F.leaf.astype(str)
F["stratum"] = F.quire_l + "|" + F.scribe.astype(str)
F["book_pos"] = [pos[f] for f in folios]
F["n_tokens"] = M[:, wmask].sum(1) - (meta.groupby("folio").size().reindex(folios).values)  # word-block transitions minus words = chars
F["n_words"] = meta.groupby("folio").size().reindex(folios).values
pickle.dump(dict(F=F, M=M, wmask=wmask, vocab=vocab, strings=strings, facing=facing),
            open(os.path.join(D, "s14_features.pkl"), "wb"))
print("pages", len(folios), "| features", len(vocab), "| facing pairs", int(facing.sum() // 2),
      "| leaves", F.leaf_id.nunique(), "| sheets", F.sheet.nunique(), "| strata", F.stratum.nunique())
