#!/usr/bin/env python3
"""PREREG 36: one-glyph respelling drift within herbal scribe 1 along npcompl33t's order. Controls first.
usage: python3 s52_respelling_within_scribe.py REPLICATION_ROOT VOYNICHDECOMP OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s51_order_vs_vocabulary.py")).read().split("R = {}")[0])   # ids, pos, strata, tokens_zl, tokens_tt
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
def atoms(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w
H = np.flatnonzero(strata == "H|1"); posH = pos[H]
THR, NP = 0.4, 2000
def prepare(toks):
    toks = [toks[i] for i in H]; tot = Counter(); [tot.update(c) for c in toks]
    V = sorted(w for w, n in tot.items() if n >= 8)
    F = np.array([[c[w] / sum(c.values()) for w in V] for c in toks])          # bifolia x types
    RF = np.apply_along_axis(rankdata, 0, F); RF = (RF - RF.mean(0)) / RF.std(0)
    D = cdist(V, V, scorer=Levenshtein.distance, workers=-1)
    iu = np.triu_indices(len(V), 1); d = D[iu]
    return dict(V=V, RF=RF, e1=(iu[0][d == 1], iu[1][d == 1]), far=(iu[0][d >= 3], iu[1][d >= 3]), F=F, toks=toks)
def trends(RF, p):
    rp = rankdata(p); rp = (rp - rp.mean()) / rp.std(); return (RF * rp[:, None]).mean(0)
def Dstat(t, P, thr):
    def rate(pr):
        a, b = t[pr[0]], t[pr[1]]; return np.mean(((a <= -thr) & (b >= thr)) | ((a >= thr) & (b <= -thr)))
    return rate(P["e1"]) - rate(P["far"])
def test(P, p, rng, thr=THR, nperm=NP):
    obs = Dstat(trends(P["RF"], p), P, thr); null = np.array([Dstat(trends(P["RF"], rng.permutation(p)), P, thr) for _ in range(nperm)])
    return dict(D=float(obs), null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), z=float((obs - null.mean()) / null.std(ddof=1)),
                n_e1=int(len(P["e1"][0])), n_far=int(len(P["far"][0])))
def inject(toks, V, order, rng, k=20):
    """replace a frequent type w by an existing one-edit type v with prob rising along 'order' (0..1)."""
    tot = Counter(); [tot.update(c) for c in toks]; freq = [w for w, n in tot.most_common() if n >= 20]
    pairs, used = [], set()
    for w in rng.permutation(freq):
        if w in used: continue
        nb = [v for v in V if v not in used and v != w and Levenshtein.distance(w, v) == 1]
        if nb:
            v = nb[int(rng.integers(len(nb)))]; pairs.append((w, v)); used |= {w, v}
        if len(pairs) == k: break
    out = []
    for i, c in enumerate(toks):
        q = order[i]; c2 = Counter(c)
        for w, v in pairs:
            n = c2.pop(w, 0); m = int(rng.binomial(n, q)); c2[v] += m
            if n - m: c2[w] += n - m
        out.append(c2)
    return out, pairs
def controls(toks, label):
    rng = np.random.default_rng(SEED + 7); pos_res, neg_res = [], []
    for r in range(20):
        order = rng.permutation(len(H)) / (len(H) - 1)
        inj, _ = inject(toks, sorted({w for c in toks for w in c}), order, rng)
        P = prepare_from(inj); pos_res.append(test(P, order, rng, nperm=500)["z"])
        P0 = prepare_from(toks); neg_res.append(test(P0, order, rng, nperm=500)["z"])
    print(label, "POSITIVE control z:", np.round(pos_res, 1), "| NEGATIVE control z:", np.round(neg_res, 1), flush=True)
    return pos_res, neg_res
def prepare_from(toksH):   # toksH already restricted to H|1, atomised
    tot = Counter(); [tot.update(c) for c in toksH]; V = sorted(w for w, n in tot.items() if n >= 8)
    F = np.array([[c[w] / sum(c.values()) for w in V] for c in toksH])
    RF = np.apply_along_axis(rankdata, 0, F); RF = (RF - RF.mean(0)) / np.maximum(RF.std(0), 1e-12)
    D = cdist(V, V, scorer=Levenshtein.distance, workers=-1); iu = np.triu_indices(len(V), 1); d = D[iu]
    return dict(V=V, RF=RF, e1=(iu[0][d == 1], iu[1][d == 1]), far=(iu[0][d >= 3], iu[1][d >= 3]))
R = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    raw = fn(); toksH = [Counter({atoms(w): n for w, n in raw[i].items()}) if label == "ZL" else raw[i] for i in H]
    pc, nc = controls(toksH, label)
    valid = sum(z >= 2 for z in pc) >= 16 and sum(abs(z) < 2 for z in nc) >= 18
    P = prepare_from(toksH); rng = np.random.default_rng(SEED + 11)
    main = test(P, posH, rng); sens = {t: test(P, posH, rng, thr=t)["z"] for t in (0.3, 0.5)}
    R[label] = dict(controls_valid=valid, positive=pc, negative=nc, main=main, sensitivity=sens, n_types=len(P["V"]))
    print(label, "types", len(P["V"]), "valid", valid, "| D %.4f null %.4f sd %.4f z %.2f | e1 pairs %d far %d | sens %s" %
          (main["D"], main["null_mean"], main["null_sd"], main["z"], main["n_e1"], main["n_far"], {k: round(v, 2) for k, v in sens.items()}), flush=True)
if not all(R[k]["controls_valid"] for k in R): v = "VOID"
elif all(R[k]["main"]["z"] >= 2 for k in R): v = "CONFIRMS"
elif abs(R["ZL"]["main"]["z"]) < 1: v = "FALSIFIES"
else: v = "UNRESOLVED"
R["verdict"] = v; json.dump(R, open(os.path.join(OUT, "results_respelling_within_scribe.json"), "w"), indent=1); print("VERDICT", v)
