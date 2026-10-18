#!/usr/bin/env python3
"""Stage 10 (EXPLORATORY bounding of PREREG 3 / B3a).
(a) per-sheet sign test vs stratum-matched non-conjugate pairs
(b) drop central bifolia (conjugate leaves adjacent in reading order)
(c) fixed-size text profiles: every page downsampled to N words (5 replicates), distance re-computed
All nulls: permute sheet labels within quire x scribe, 2,000 draws.
"""
import json, os, pickle, re, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import binomtest

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
folios = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[1]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"].copy()
for c in ("folio", "word", "scribe_davis"): meta[c] = meta[c].astype(str)
meta["line_num"] = pd.to_numeric(meta.line_num, errors="coerce"); meta["word_index"] = meta.word_index.astype(int)
F["scribe"] = meta.groupby("folio").scribe_davis.agg(lambda x: x.mode().iat[0]).reindex(F.index).values
F["quire_l"] = F.sheet.str.split(":").str[0]
st = (F.quire_l + "|" + F.scribe.astype(str)).values
sheet, leaf = F.sheet.values, F.leaf.astype(str).values
n = len(F); iu = np.triu_indices(n, 1)
gi = [np.where(st == g)[0] for g in np.unique(st)]


def conj(sh): return (sh[iu[0]] == sh[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])


def null_test(Dm, keep=None, draws=2000):
    keep = np.ones(n, bool) if keep is None else keep
    pm = keep[iu[0]] & keep[iu[1]]
    d = Dm[iu]; obs = d[conj(sheet) & pm].mean()
    null = []
    for _ in range(draws):
        s2 = sheet.copy()
        for ix in gi: s2[ix] = sheet[rng.permutation(ix)]
        m = conj(s2) & pm
        if m.any(): null.append(d[m].mean())
    null = np.array(null)
    return dict(n_pairs=int((conj(sheet) & pm).sum()), obs=round(float(obs), 5),
                null_mean=round(float(null.mean()), 5), null_sd=round(float(null.std()), 5),
                z=round(float((null.mean() - obs) / null.std()), 2),
                pct_lower=round(float(100 * (null.mean() - obs) / null.mean()), 2))


res = {}
# (a) per sheet: mean conjugate distance vs mean distance to same-stratum pages on OTHER sheets
wins, tot = 0, 0; per = []
for s in np.unique(sheet):
    ix = np.where(sheet == s)[0]
    cj = [(i, j) for a, i in enumerate(ix) for j in ix[a + 1:] if leaf[i] != leaf[j]]
    if not cj: continue
    ctrl = [Dt[i, k] for i in ix for k in range(n) if st[k] == st[i] and sheet[k] != s]
    if not ctrl: continue
    c = np.mean([Dt[i, j] for i, j in cj]); o = np.mean(ctrl)
    per.append((s, round(c, 4), round(o, 4))); tot += 1; wins += c < o
res["a_sheets_tested"] = tot; res["a_sheets_conjugate_closer"] = int(wins)
res["a_sign_test_p"] = float(binomtest(int(wins), tot, 0.5, alternative="greater").pvalue)

# (b) drop central bifolia (leaf numbers consecutive, e.g. A:4-5)
def central(s):
    m = re.match(r".*:(\d+)-(\d+)$", s)
    return bool(m) and abs(int(m.group(1)) - int(m.group(2))) == 1
keep_b = np.array([not central(s) for s in sheet])
res["b_central_sheets_dropped"] = sorted({s for s in sheet if central(s)})
res["b_no_central"] = null_test(Dt, keep_b)

# (c) fixed-size text profiles
def profile(words_by_line):
    c = Counter()
    for ln in words_by_line:
        for w in ln:
            s = "^" + w + "$"; c.update(("w", a, b) for a, b in zip(s, s[1:]))
        s = "".join(ln); c.update(("l", a, b) for a, b in zip(s, s[1:]))
    return c
lines = {f: [list(g.word) for _, g in sub.sort_values(["line_num", "word_index"]).groupby("line_num")]
         for f, sub in meta.groupby("folio")}
for N in (60, 100):
    ok = np.array([sum(len(l) for l in lines[f]) >= N for f in folios])
    Ds = []
    for rep in range(5):
        profs = []
        for f in folios:
            if not ok[folios.index(f)]: profs.append(Counter()); continue
            flat = [(li, w) for li, l in enumerate(lines[f]) for w in l]
            pick = sorted(rng.choice(len(flat), N, replace=False))
            byl = {}
            for k in pick: byl.setdefault(flat[k][0], []).append(flat[k][1])
            profs.append(profile(list(byl.values())))
        vocab = sorted(set().union(*profs)); vi = {v: i for i, v in enumerate(vocab)}
        M = np.zeros((n, len(vocab)))
        for i, p in enumerate(profs):
            for k, v in p.items(): M[i, vi[k]] = v
        wm = np.array([v[0] == "w" for v in vocab]); P = np.zeros_like(M)
        with np.errstate(invalid="ignore", divide="ignore"):
            P[:, wm] = M[:, wm] / M[:, wm].sum(1, keepdims=True); P[:, ~wm] = M[:, ~wm] / M[:, ~wm].sum(1, keepdims=True)
        P = np.nan_to_num(P) + 1e-9; P /= P.sum(1, keepdims=True)
        Ds.append(squareform(pdist(P, lambda a, b: jensenshannon(a, b, base=2))))
    res[f"c_fixed_{N}_words_pages"] = int(ok.sum())
    res[f"c_fixed_{N}_words"] = null_test(np.mean(Ds, 0), ok)
    res[f"c_fixed_{N}_words_no_central"] = null_test(np.mean(Ds, 0), ok & keep_b)

json.dump(res, open(os.path.join(ROOT, "results_scribe_bounding.json"), "w"), indent=1)
pd.DataFrame(per, columns=["sheet", "conjugate_mean_JS", "same_stratum_other_sheet_JS"]).to_csv(
    os.path.join(ROOT, "per_sheet_conjugate.csv"), index=False)
print(json.dumps(res, indent=1))
