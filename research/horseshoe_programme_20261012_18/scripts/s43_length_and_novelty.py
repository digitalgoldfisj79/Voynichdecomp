#!/usr/bin/env python3
"""PREREG 28 (RUNNING_RESULTS_horseshoe.md, 2026-10-16).
(A) neighbour length pairing: OLS len(w[i+1]) ~ len(w[i]) + nested controls, line-cluster bootstrap.
(B) new words in held-out pages: one-edit (E1), recombination (RC) or neither, vs within-word atom-shuffle null.
usage: python3 -I s43_length_and_novelty.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VULJ = sys.argv[2]; sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3]]
exec(open(os.path.join(HERE, "s41_prefix.py")).read())
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

def stage(name, fn):
    p = os.path.join(OUT, "stages", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r_ = fn(); pickle.dump(r_, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r_

# ---------------- (A)
def pairs_table(lines_with_page):
    rows = []
    for li, (pg, l) in enumerate(lines_with_page):
        n_ = len(l)
        for i in range(n_ - 1):
            a, b = l[i], l[i + 1]
            rows.append((li, len(a), len(b), f"{min(n_, 14)}_{i + 1}", a[-1], b[0], pg))
    return rows
def design(rows, model):
    y = np.array([r[2] for r in rows], float); x = np.array([r[1] for r in rows], float)
    cols = [np.ones(len(rows)), x]
    def fe(k):
        levels = sorted({r[k] for r in rows}); idx = {v: i for i, v in enumerate(levels)}
        M = np.zeros((len(rows), len(levels) - 1))
        for i, r in enumerate(rows):
            j = idx[r[k]]
            if j: M[i, j - 1] = 1
        return M
    if model >= 1: cols.append(fe(3))
    if model >= 2: cols += [fe(4), fe(5)]
    if model >= 3: cols.append(fe(6))
    return np.column_stack(cols), y
def beta(X, y): return float(np.linalg.lstsq(X, y, rcond=None)[0][1])
def length_models(lines_with_page, rng):
    rows = pairs_table(lines_with_page); lid = np.array([r[0] for r in rows]); lines_ids = np.unique(lid)
    by_line = {k: np.flatnonzero(lid == k) for k in lines_ids}; out = {}
    for m in range(4):
        X, y = design(rows, m); b0 = beta(X, y); bs = []
        for _ in range(NBOOT):
            pick = rng.choice(lines_ids, size=len(lines_ids)); idx = np.concatenate([by_line[k] for k in pick])
            bs.append(beta(X[idx], y[idx]))
        sd = float(np.std(bs, ddof=1)); out[f"M{m}"] = dict(beta=b0, sd=sd, z=b0 / sd, n_params=X.shape[1])
        print("  M%d beta %.4f sd %.4f z %.1f" % (m, b0, sd, b0 / sd), flush=True)
    return out
def partA():
    r1 = np.random.default_rng(SEED + 101)
    vms = [(f, l) for f in pages for l in pages[f]]
    V = json.load(open(VULJ))["tokens"][:60000]; LL = np.array([len(l) for _, l in vms])
    vl = to_lines(V, LL, np.random.default_rng(SEED + 102)); vul = [(i // 25, l) for i, l in enumerate(vl)]  # pseudo-pages of 25 lines
    print("VMS"); a = length_models(vms, r1); print("VULGATE"); b = length_models(vul, r1)
    return {"VMS": a, "VULGATE": b}
A = stage("partA_length", partA)

# ---------------- (B)
def classify(novel, known_list, known_set, prefixes, suffixes):
    if not novel: return np.zeros(0, int)
    D_ = cdist(novel, known_list, scorer=Levenshtein.distance, score_cutoff=1, workers=-1)
    e1 = (D_ <= 1).any(1); cls = np.where(e1, 0, 2)
    for i, w in enumerate(novel):
        if e1[i]: continue
        if any(w[:k] in prefixes and w[k:] in suffixes for k in range(1, len(w))): cls[i] = 1
    return cls  # 0 E1, 1 RC, 2 NONE
def novelty(known_lines, new_lines, rng, label):
    kt = Counter(w for l in known_lines for w in l); known_list = list(kt); known_set = set(kt)
    prefixes = {w[:k] for w in known_list for k in range(1, len(w))}; suffixes = {w[k:] for w in known_list for k in range(1, len(w))}
    nt = Counter(w for l in new_lines for w in l); novel = [w for w in nt if w not in known_set]; wts = np.array([nt[w] for w in novel], float)
    cls = classify(novel, known_list, known_set, prefixes, suffixes)
    def shares(c): return [float(np.mean(c == k)) for k in range(3)], [float(np.sum(wts[c == k]) / wts.sum()) for k in range(3)]
    obs_type, obs_tok = shares(cls); nt_, ntk = [], []
    for _ in range(NBOOT):
        sh = []
        for w in novel:
            a = list(w); rng.shuffle(a); sh.append("".join(a))
        c = classify(sh, known_list, known_set, prefixes, suffixes); s1, s2 = shares(c); nt_.append(s1); ntk.append(s2)
    nt_, ntk = np.array(nt_), np.array(ntk); names = ["E1", "RC", "NONE"]
    res = dict(n_novel_types=len(novel), novel_token_share=float(wts.sum() / sum(nt.values())))
    for k, nm in enumerate(names):
        res[nm] = dict(type_share=obs_type[k], null_type=float(nt_[:, k].mean()), null_type_sd=float(nt_[:, k].std(ddof=1)),
                       z_type=float((obs_type[k] - nt_[:, k].mean()) / max(nt_[:, k].std(ddof=1), 1e-9)),
                       token_share=obs_tok[k], null_token=float(ntk[:, k].mean()), z_token=float((obs_tok[k] - ntk[:, k].mean()) / max(ntk[:, k].std(ddof=1), 1e-9)))
    print(label, json.dumps({k: (round(v["type_share"], 3), round(v["null_type"], 3), round(v["z_type"], 1)) if isinstance(v, dict) else v for k, v in res.items()}), flush=True)
    return res
def partB():
    r2 = np.random.default_rng(SEED + 111)
    vms = novelty(L(FIT) + L(DEV), L(TEST), r2, "VMS")
    V = json.load(open(VULJ))["tokens"]; vul = novelty([V[:30000]], [V[30000:40000]], r2, "VULGATE")
    return {"VMS": vms, "VULGATE": vul}
B = stage("partB_novelty", partB)
json.dump({"A_length": A, "B_novelty": B}, open(os.path.join(OUT, "results_length_novelty.json"), "w"), indent=1)
print(json.dumps(A, indent=0)); print(json.dumps(B, indent=0))
