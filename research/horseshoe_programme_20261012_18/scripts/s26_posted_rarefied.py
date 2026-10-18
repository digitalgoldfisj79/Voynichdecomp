#!/usr/bin/env python3
"""PREREG 11: size-equalised rerun of the posted bifolium tests, with a within-stratum word-shuffle negative control.
Stages pickled in data/s26_*.pkl; results -> results_posted_rarefied.json"""
import json, os, pickle, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261018)
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb"))
F, M, wmask, facing, vocab = S["F"], S["M"], S["wmask"], S["facing"], S["vocab"]
folios = list(F.index); n = len(F)
OUT = os.path.join(ROOT, "results_posted_rarefied.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)
def stage(name, fn):
    p = os.path.join(D, f"s26_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r = fn(); pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r

# ---- negative control counts: shuffle words within quire x scribe stratum, keep line lengths
def build_shuf():
    meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"].copy()
    meta["folio"] = meta.folio.astype(str); meta["word"] = meta.word.astype(str)
    meta["line_num"] = pd.to_numeric(meta.line_num, errors="coerce"); meta["word_index"] = meta.word_index.astype(int)
    lines = {}
    for f in folios:
        sub = meta[meta.folio == f].sort_values(["line_num", "word_index"])
        lines[f] = [list(l.word) for _, l in sub.groupby("line_num", sort=True)]
    vi = {v: i for i, v in enumerate(vocab)}; Ms = np.zeros_like(M)
    for st, grp in F.groupby("stratum"):
        pool = [w for f in grp.index for l in lines[f] for w in l]; rng.shuffle(pool); pos = 0
        for f in grp.index:
            c = Counter()
            for l in lines[f]:
                ws = pool[pos:pos + len(l)]; pos += len(l)
                for w in ws:
                    s = "^" + w + "$"; c.update(("w", a, b) for a, b in zip(s, s[1:]))
                s = "".join(ws); c.update(("l", a, b) for a, b in zip(s, s[1:]))
            r = folios.index(f)
            for k, v in c.items():
                if k in vi: Ms[r, vi[k]] += v
                # unseen transitions (new across-word joins) are dropped; logged below
    return Ms
Mshuf = stage("Mshuf", build_shuf)

def rarefied(Mx, keep, Nw, Nl, draws=5):
    idx = np.where(keep)[0]; out = np.zeros((len(idx), len(idx)))
    for _ in range(draws):
        X = np.zeros((len(idx), Mx.shape[1]))
        for r, i in enumerate(idx):
            for blk, N in ((wmask, Nw), (~wmask, Nl)):
                cols = np.where(blk)[0]; cnt = Mx[i, cols].astype(int)
                arr = np.repeat(cols, cnt); np.add.at(X[r], rng.choice(arr, N, replace=False), 1)
        out += L.js_matrix(L.profiles(X, wmask))
    return out / draws, idx

def tests(Dm, idx, tag, do_perm=True, do_mrm=False):
    Fi = F.iloc[idx]; sheet, leaf_id, strat = Fi.sheet.values, Fi.leaf_id.values, Fi.stratum.values
    leafnum = Fi.leaf.astype(str).values; m = len(idx); iu = np.triu_indices(m, 1); d = Dm[iu]
    nulls = [L.leaf_permute(sheet, leaf_id, strat, rng) for _ in range(2000)]
    def mc(sh):
        mm = (sh[iu[0]] == sh[iu[1]]) & (leafnum[iu[0]] != leafnum[iu[1]]); return d[mm].mean()
    obs = mc(sheet); nl = np.array([mc(s) for s in nulls])
    r = dict(n_pages=int(m), T_conj=dict(obs=obs, null_mean=nl.mean(), null_sd=nl.std(), z=(nl.mean() - obs) / nl.std(),
             pct_closer=100 * (nl.mean() - obs) / nl.mean()))
    # per-sheet: conjugate mean vs same-stratum other-sheet mean
    closer = tot = 0
    for s in np.unique(sheet):
        a = np.where(sheet == s)[0]
        conj = [Dm[i, j] for i in a for j in a if i < j and leafnum[i] != leafnum[j]]
        oth = [Dm[i, j] for i in a for j in range(m) if sheet[j] != s and strat[j] == strat[i]]
        if conj and oth: tot += 1; closer += np.mean(conj) < np.mean(oth)
    r["T_conj"]["sheets_closer"] = f"{closer}/{tot}"
    if do_perm:
        G = L.gower(Dm)
        base = [("currier", Fi.currier.values), ("scribe", Fi.scribe.astype(str).values),
                ("section", Fi.csec.values), ("quire", Fi.quire_l.values)]
        o = L.permanova_seq(G, base + [("sheet", sheet), ("leaf", leaf_id)]); ob = o["sheet"][0]
        nn = np.array([L.permanova_seq(G, base + [("sheet", s)])["sheet"][0] for s in nulls[:1000]])
        r["T1b"] = dict(SS_sheet=ob, null_mean=nn.mean(), null_sd=nn.std(), z=(ob - nn.mean()) / nn.std(),
                        R2_sheet_of_total=ob / o["total"][0])
    if do_mrm:
        fac = facing[np.ix_(idx, idx)]; bp = Fi.book_pos.values; nw = Fi.n_words.values.astype(float)
        def X_for(sh):
            a, b = iu
            cols = [(sh[a] == sh[b]) & (leafnum[a] != leafnum[b]), leaf_id[a] == leaf_id[b], fac[a, b],
                    np.log1p(np.abs(bp[a] - bp[b])), Fi.quire_l.values[a] == Fi.quire_l.values[b],
                    Fi.scribe.astype(str).values[a] == Fi.scribe.astype(str).values[b],
                    Fi.csec.values[a] == Fi.csec.values[b], Fi.currier.values[a] == Fi.currier.values[b],
                    np.abs(np.log(nw[a]) - np.log(nw[b])), np.log(np.minimum(nw[a], nw[b]))]
            return np.c_[[c.astype(float) for c in cols]].T
        X = X_for(sheet); b, _ = L.mrm_fit(d, X); jk = []
        for s in np.unique(sheet):
            mm = (sheet[iu[0]] != s) & (sheet[iu[1]] != s); bj, _ = L.mrm_fit(d[mm], X[mm]); jk.append((bj[0], bj[2]))
        jk = np.array(jk); k = len(jk); se = np.sqrt((k - 1) / k * ((jk - jk.mean(0)) ** 2).sum(0))
        nullb = np.array([L.mrm_fit(d, X_for(s))[0][0] for s in nulls[:1000]])
        r["T2"] = dict(conj_beta=b[0], ci_conj=[b[0] - 1.96 * se[0], b[0] + 1.96 * se[0]],
                       facing_beta=b[2], ci_facing=[b[2] - 1.96 * se[1], b[2] + 1.96 * se[1]],
                       conj_z=(nullb.mean() - b[0]) / nullb.std(), n_facing=int(X[:, 2].sum()))
    # face test: non-central standard bifolia with all four pages present
    pos = {folios[i]: k for k, i in enumerate(idx)}; dd = []
    for sh, sub in Fi.groupby("sheet"):
        mt = re.match(r".*:(\d+)-(\d+)$", sh)
        if not mt: continue
        a_, b_ = sorted((int(mt.group(1)), int(mt.group(2))))
        if b_ - a_ <= 1: continue
        want = [f"f{a_}r", f"f{a_}v", f"f{b_}r", f"f{b_}v"]
        if not all(w in pos for w in want): continue
        ar, av, br, bv = [pos[w] for w in want]
        dd.append((Dm[ar, br] + Dm[av, bv]) / 2 - (Dm[ar, bv] + Dm[av, br]) / 2)
    dd = np.array(dd); fl = (dd[None, :] * rng.choice([1, -1], size=(100000, len(dd)))).mean(1)
    r["T_face"] = dict(n_sheets=int(len(dd)), mean_D=dd.mean(), null_sd=fl.std(), z=dd.mean() / fl.std(),
                       p_one=(np.sum(fl >= dd.mean()) + 1) / (len(fl) + 1), closer=f"{int((dd > 0).sum())}/{len(dd)}")
    res[tag] = r; save(); print(tag, json.dumps(r, default=lambda x: round(float(x), 4)))

allp = np.ones(n, bool); k60 = F.n_words.values >= 60; k100 = F.n_words.values >= 100
Draw = L.js_matrix(L.profiles(M, wmask)); Dsraw = L.js_matrix(L.profiles(Mshuf, wmask))
jobs = [("REAL_RAW", lambda: (Draw, np.arange(n)), dict(do_perm=False)),
        ("SHUF_RAW", lambda: (Dsraw, np.arange(n)), dict(do_perm=True, do_mrm=True)),
        ("REAL_R60", lambda: stage("R60", lambda: rarefied(M, k60, 334, 261)), dict(do_perm=True, do_mrm=True)),
        ("SHUF_R60", lambda: stage("SR60", lambda: rarefied(Mshuf, k60, 334, 261)), dict(do_perm=False)),
        ("REAL_R100", lambda: stage("R100", lambda: rarefied(M, k100, 555, 435)), dict(do_perm=True))]
for tag, fn, kw in jobs:
    if tag in res: continue
    Dm, idx = fn(); tests(Dm, idx, tag, **kw)
