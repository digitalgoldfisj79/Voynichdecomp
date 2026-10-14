#!/usr/bin/env python3
"""PREREG 9: can the TEXT order sheets? Split-leaf replication of between-sheet distances within strata,
on VMS and on known-order controls (Circa instans, Galen) cut into singulions at VMS page sizes.
Stages (pickled): A build units -> data/s22_units.pkl ; B stats -> results_chain_calibration.json
usage: python3 -I s22_chain_calibration.py ROOT CONTROL_DIR
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib_methods as L

ROOT, CDIR = sys.argv[1], sys.argv[2]
D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261014)
NPERM = 10000; BLOCKS = [21, 6, 5, 5, 3, 2, 2, 1, 1, 1, 1, 1]

# ---------------------------------------------------------------- stage A: units
UNITS = os.path.join(D, "s22_units.pkl")
def feats_from_lines(lines):
    c = Counter()
    for ws in lines:
        for w in ws:
            s = "^" + w + "$"; c.update(("w", a, b) for a, b in zip(s, s[1:]))
        s = "".join(ws); c.update(("l", a, b) for a, b in zip(s, s[1:]))
    return c

if not os.path.exists(UNITS):
    S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb")); F = S["F"].copy()
    meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"].copy()
    meta["folio"] = meta.folio.astype(str); meta["word"] = meta.word.astype(str)
    meta["line_num"] = pd.to_numeric(meta.line_num, errors="coerce"); meta["word_index"] = meta.word_index.astype(int)
    F["half"] = F.groupby("sheet").leaf.transform(lambda x: (x > x.min()).astype(int))
    F["nl"] = F.groupby("sheet").leaf.transform("nunique")
    W = F.groupby(["sheet", "half"]).n_words.sum().unstack()
    two = F.groupby("sheet").nl.first() == 2
    elig = [s for s in W.index if two[s] and W.loc[s].min() >= 60]
    F["book_pos"] = F.book_pos.astype(int)
    elig = sorted(elig, key=lambda s: F[F.sheet == s].book_pos.min())
    linelens = meta.groupby(["folio", "line_num"]).size().values
    def modal(s, c): return str(F[F.sheet == s][c].mode().iat[0])
    vms = dict(sheets=elig, half={}, words={},
               strat=np.array(["|".join(modal(s, c) for c in ("scribe", "currier", "csec")) for s in elig]),
               strat2=np.array(["|".join(modal(s, c) for c in ("scribe", "currier")) for s in elig]))
    pagesizes = []
    for s in elig:
        sub = F[F.sheet == s].sort_values("book_pos")
        pagesizes.append([int(x) for x in sub.n_words.values])
        for h in (0, 1):
            lines, words = [], []
            for f in sub[sub.half == h].index:
                m = meta[meta.folio == f].sort_values(["line_num", "word_index"])
                for _, ln in m.groupby("line_num", sort=True):
                    ws = list(ln.word); lines.append(ws); words += ws
            vms["half"][(s, h)] = feats_from_lines(lines); vms["words"][(s, h)] = words
    # controls
    def clean(ws):
        out = []
        for w in ws:
            w = "".join(ch for ch in str(w).lower() if ch.isalpha())
            if w: out.append(w)
        return out
    ci = clean(pickle.load(open(os.path.join(CDIR, "ci_corpus_parsed.pkl"), "rb"))["all_words"])
    gk = clean(pickle.load(open(os.path.join(CDIR, "greek_corpus_parsed.pkl"), "rb"))["all_words"])
    need = sum(sum(p) for p in pagesizes)
    windows = {"CI_start": ci[:need], "CI_end": ci[-need:]}
    for k in range(3):
        if (k + 1) * need <= len(gk): windows[f"Galen_w{k+1}"] = gk[k * need:(k + 1) * need]
    controls = {}
    for name, ws in windows.items():
        pos = 0; half, words = {}, {}
        for i, ps in enumerate(pagesizes):
            # singulion: pages in reading order ar, av (leaf A) , br, bv (leaf B); VMS sheets with >4 pages keep their count
            npg = len(ps); hA = ps[: npg // 2]; hB = ps[npg // 2:]
            for h, sizes in ((0, hA), (1, hB)):
                lines, wl = [], []
                for sz in sizes:
                    page = ws[pos:pos + sz]; pos += sz; j = 0
                    while j < len(page):
                        ll = int(rng.choice(linelens)); lines.append(page[j:j + ll]); j += ll
                    wl += page
                half[(i, h)] = feats_from_lines(lines); words[(i, h)] = wl
        perm_blocks = rng.permutation(BLOCKS); strat = np.concatenate([[f"b{k}"] * b for k, b in enumerate(perm_blocks)])
        controls[name] = dict(sheets=list(range(len(pagesizes))), half=half, words=words, strat=strat, strat2=strat,
                              n_words=len(ws))
    pickle.dump(dict(vms=vms, controls=controls, need=need), open(UNITS, "wb"))
U = pickle.load(open(UNITS, "rb"))
print("words needed per window", U["need"], "| controls", list(U["controls"]))

# ---------------------------------------------------------------- stage B: statistics
def matrices(unit, kind):
    keys = unit["sheets"]
    if kind == "char":
        vocab = sorted(set().union(*[unit["half"][(s, h)] for s in keys for h in (0, 1)]))
        vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
        def M(hs):
            X = np.zeros((len(keys), len(vocab)))
            for r, s in enumerate(keys):
                c = Counter()
                for h in hs: c.update(unit["half"][(s, h)])
                for k, v in c.items(): X[r, vi[k]] = v
            return L.js_matrix(L.profiles(X, wmask))
        return M([0]), M([1]), M([0, 1])
    # word tf-idf cosine; idf from all halves
    docs = {(s, h): Counter(unit["words"][(s, h)]) for s in keys for h in (0, 1)}
    vocab = sorted(set().union(*docs.values())); vi = {v: i for i, v in enumerate(vocab)}
    df = Counter(); [df.update(set(d)) for d in docs.values()]
    idf = np.array([np.log(len(docs) / df[v]) for v in vocab])
    def M(hs):
        X = np.zeros((len(keys), len(vocab)))
        for r, s in enumerate(keys):
            for h in hs:
                for k, v in docs[(s, h)].items(): X[r, vi[k]] += v
        X = X * idf; X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return 1 - X @ X.T
    return M([0]), M([1]), M([0, 1])

def within_perm(strat):
    p = np.arange(len(strat))
    for g in np.unique(strat):
        idx = np.where(strat == g)[0]; p[idx] = rng.permutation(idx)
    return p

def stats(DA, DB, strat):
    n = len(strat); iu = np.triu_indices(n, 1)
    within = (strat[iu[0]] == strat[iu[1]])
    def tw(DBp): return np.corrcoef(DA[iu][within], DBp[iu][within])[0, 1]
    big = np.array([np.sum(strat == strat[i]) >= 3 for i in range(n)])
    def nn(Dm, i):
        cand = np.where((strat == strat[i]) & (np.arange(n) != i))[0]; return cand[np.argmin(Dm[i, cand])]
    def tnn(DBp): return np.mean([nn(DA, i) == nn(DBp, i) for i in np.where(big)[0]])
    def tall(DBp): return np.corrcoef(DA[iu], DBp[iu])[0, 1]
    obs = dict(T_w=tw(DB), T_nn=tnn(DB), T_all=tall(DB))
    nw, nnn, na = [], [], []
    for _ in range(NPERM):
        p = within_perm(strat); DBp = DB[np.ix_(p, p)]; nw.append(tw(DBp)); nnn.append(tnn(DBp))
    for _ in range(2000):
        p = rng.permutation(n); na.append(tall(DB[np.ix_(p, p)]))
    out = {}
    for k, nl in (("T_w", nw), ("T_nn", nnn), ("T_all", na)):
        nl = np.array(nl); sd = nl.std()
        out[k] = dict(obs=round(float(obs[k]), 4), null_mean=round(float(nl.mean()), 4), null_sd=round(float(sd), 4),
                      z=round(float((obs[k] - nl.mean()) / sd), 2) if sd > 0 else None,
                      p_one=round(float((np.sum(nl >= obs[k]) + 1) / (len(nl) + 1)), 5))
    out["n_within_pairs"] = int(within.sum()); out["n_nn_sheets"] = int(big.sum())
    return out

def recovery(Dfull, strat):
    n = Dfull.shape[0]; truth = np.arange(n)
    o = L.spectral_order(Dfull); rho_all = abs(spearmanr(o, truth)[0])
    g = max(np.unique(strat), key=lambda x: np.sum(strat == x)); idx = np.where(strat == g)[0]
    ob = L.spectral_order(Dfull[np.ix_(idx, idx)]); rho_b = abs(spearmanr(ob, truth[idx])[0])
    return dict(rho_all=round(float(rho_all), 3), rho_largest_block=round(float(rho_b), 3), block_n=int(len(idx)))

RES = os.path.join(ROOT, "results_chain_calibration.json")
res = json.load(open(RES)) if os.path.exists(RES) else {}
jobs = [("VMS", U["vms"])] + list(U["controls"].items())
for name, unit in jobs:
    for kind in ("char", "word"):
        key = f"{name}|{kind}"
        if key in res: continue
        DA, DB, Dfull = matrices(unit, kind)
        r = dict(primary=stats(DA, DB, unit["strat"]))
        if name == "VMS": r["secondary_strata"] = stats(DA, DB, unit["strat2"])
        else: r["recovery"] = recovery(Dfull, unit["strat"])
        res[key] = r; json.dump(res, open(RES, "w"), indent=1)  # atomic per job
        print(key, json.dumps(r))
