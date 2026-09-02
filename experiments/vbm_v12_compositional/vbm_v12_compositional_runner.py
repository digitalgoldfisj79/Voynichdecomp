#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.11,<2", "scikit-learn>=1.4,<2"]
# ///
from __future__ import annotations
import argparse, hashlib, json, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

NS = "VBMV12COMPOSITIONAL20260902"

def rng_for(tag: str) -> np.random.Generator:
    h = hashlib.sha256(f"{NS}::{tag}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))

@dataclass(frozen=True)
class Stage:
    name: str
    KN: int
    KB: int
    S: int
    E: int
    R: int
    L: int
    lines: int
    restarts: int

STAGE_A = Stage("A", 16, 7, 48, 3, 10, 9, 2000, 16)
STAGE_B = Stage("B", 24, 11, 64, 4, 14, 13, 4000, 24)
SOURCE_FAMILIES = {"PEAKED": 0.25, "MODERATE": 0.75}
MODES = ("POS", "NUC_BROKEN", "BRIDGE_BROKEN", "BOTH_BROKEN")

def source(stage: Stage, family: str, rep: int):
    alpha = SOURCE_FAMILIES[family]
    r = rng_for(f"{stage.name}:{family}:{rep}:source")
    p0 = r.dirichlet(np.full(stage.KN, alpha))
    pb = r.dirichlet(np.full(stage.KB, alpha), size=stage.KN)
    pn = r.dirichlet(np.full(stage.KN, alpha), size=stage.KB)
    return p0, pb, pn

def permutation_powers(pi: np.ndarray, E: int) -> np.ndarray:
    KN = len(pi)
    p = np.empty((E, KN), dtype=np.int64)
    p[0] = np.arange(KN)
    for m in range(1, E):
        p[m] = pi[p[m - 1]]
    return p

def positive_key(stage: Stage, family: str, rep: int):
    r = rng_for(f"{stage.name}:{family}:{rep}:key")
    base = np.arange(stage.S, dtype=np.int64) % stage.KN
    r.shuffle(base)
    pi = r.permutation(stage.KN).astype(np.int64)
    u = np.arange(stage.R, dtype=np.int64) % stage.KB
    v = np.arange(stage.L, dtype=np.int64) % stage.KB
    r.shuffle(u); r.shuffle(v)
    pp = permutation_powers(pi, stage.E)
    nmap = np.array([pp[m, base[s]] for s in range(stage.S) for m in range(stage.E)], dtype=np.int64)
    bmap = np.array([(u[x] + v[y]) % stage.KB for x in range(stage.R) for y in range(stage.L)], dtype=np.int64)
    return base, pi, u, v, nmap, bmap

def arbitrary_balanced(n: int, k: int, tag: str) -> np.ndarray:
    r = rng_for(tag)
    a = np.arange(n, dtype=np.int64) % k
    r.shuffle(a)
    return a

def generate(stage: Stage, family: str, rep: int, mode: str):
    p0, pb, pn = source(stage, family, rep)
    base, pi, u, v, npos, bpos = positive_key(stage, family, rep)
    nmap = npos.copy(); bmap = bpos.copy()
    if mode in ("NUC_BROKEN", "BOTH_BROKEN"):
        nmap = arbitrary_balanced(stage.S * stage.E, stage.KN, f"{stage.name}:{family}:{rep}:{mode}:nmap")
    if mode in ("BRIDGE_BROKEN", "BOTH_BROKEN"):
        bmap = arbitrary_balanced(stage.R * stage.L, stage.KB, f"{stage.name}:{family}:{rep}:{mode}:bmap")

    rw = rng_for(f"{stage.name}:{family}:{rep}:weights")
    wn = np.exp(rw.normal(0.0, 0.7, stage.S * stage.E))
    wb = np.exp(rw.normal(0.0, 0.7, stage.R * stage.L))
    ncompat = [np.where(nmap == z)[0] for z in range(stage.KN)]
    bcompat = [np.where(bmap == z)[0] for z in range(stage.KB)]
    if not all(len(x) for x in ncompat) or not all(len(x) for x in bcompat):
        raise RuntimeError("generator produced empty latent emission class")
    nprob = [wn[x] / wn[x].sum() for x in ncompat]
    bprob = [wb[x] / wb[x].sum() for x in bcompat]

    r_lat = rng_for(f"{stage.name}:{family}:{rep}:latent")
    r_emit = rng_for(f"{stage.name}:{family}:{rep}:{mode}:emit")
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    latent: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(stage.lines):
        ntok = int(r_lat.integers(5, 16))
        ns, bs, nlat, blat = [], [], [], []
        n = int(r_lat.choice(stage.KN, p=p0))
        nlat.append(n)
        ns.append(int(r_emit.choice(ncompat[n], p=nprob[n])))
        for _j in range(ntok - 1):
            b = int(r_lat.choice(stage.KB, p=pb[n]))
            blat.append(b)
            bs.append(int(r_emit.choice(bcompat[b], p=bprob[b])))
            n = int(r_lat.choice(stage.KN, p=pn[b]))
            nlat.append(n)
            ns.append(int(r_emit.choice(ncompat[n], p=nprob[n])))
        lines.append((np.asarray(ns, dtype=np.int64), np.asarray(bs, dtype=np.int64)))
        latent.append((np.asarray(nlat, dtype=np.int64), np.asarray(blat, dtype=np.int64)))

    truth = {"base": base, "pi": pi, "u": u, "v": v, "nmap": nmap, "bmap": bmap,
             "nmap_positive": npos, "bmap_positive": bpos, "p0": p0, "pb": pb, "pn": pn}
    return lines, latent, truth

def aggregate(lines, Ns: int, Bs: int):
    cnb = np.zeros((Ns, Bs), dtype=np.int64)
    cbn = np.zeros((Bs, Ns), dtype=np.int64)
    init = np.zeros(Ns, dtype=np.int64)
    ncnt = np.zeros(Ns, dtype=np.int64)
    bcnt = np.zeros(Bs, dtype=np.int64)
    for ns, bs in lines:
        init[ns[0]] += 1
        np.add.at(ncnt, ns, 1); np.add.at(bcnt, bs, 1)
        np.add.at(cnb, (ns[:-1], bs), 1); np.add.at(cbn, (bs, ns[1:]), 1)
    return cnb, cbn, init, ncnt, bcnt

def context_features(counts, smooth=0.5):
    cnb, cbn, _init, _nc, _bc = counts
    Ns, Bs = cnb.shape
    n_out = (cnb + smooth) / (cnb.sum(1, keepdims=True) + smooth * Bs)
    n_in0 = cbn.T
    n_in = (n_in0 + smooth) / (n_in0.sum(1, keepdims=True) + smooth * Bs)
    b_in0 = cnb.T
    b_in = (b_in0 + smooth) / (b_in0.sum(1, keepdims=True) + smooth * Ns)
    b_out = (cbn + smooth) / (cbn.sum(1, keepdims=True) + smooth * Ns)
    return np.hstack([np.sqrt(n_out), np.sqrt(n_in)]), np.hstack([np.sqrt(b_in), np.sqrt(b_out)])

def cluster_transition_mats(counts, nl, bl, KN, KB, smooth=0.5):
    cnb, cbn, *_ = counts
    A = np.zeros((KN, KB), dtype=np.float64)
    B = np.zeros((KB, KN), dtype=np.float64)
    for i in range(cnb.shape[0]):
        np.add.at(A[nl[i]], bl, cnb[i])
    for j in range(cbn.shape[0]):
        np.add.at(B[bl[j]], nl, cbn[j])
    A = (A + smooth) / (A.sum(1, keepdims=True) + smooth * KB)
    B = (B + smooth) / (B.sum(1, keepdims=True) + smooth * KN)
    return A, B

def align_to_source(counts, nl, bl, src):
    p0, pb, pn = src
    KN, KB = pb.shape
    A, B = cluster_transition_mats(counts, nl, bl, KN, KB)
    emp_n = np.asarray([np.r_[np.sort(A[n]), np.sort(B[:, n])] for n in range(KN)])
    src_n = np.asarray([np.r_[np.sort(pb[n]), np.sort(pn[:, n])] for n in range(KN)])
    emp_b = np.asarray([np.r_[np.sort(B[b]), np.sort(A[:, b])] for b in range(KB)])
    src_b = np.asarray([np.r_[np.sort(pn[b]), np.sort(pb[:, b])] for b in range(KB)])
    dn = ((emp_n[:, None, :] - src_n[None, :, :]) ** 2).sum(2)
    db = ((emp_b[:, None, :] - src_b[None, :, :]) ** 2).sum(2)
    rn, cn = linear_sum_assignment(dn); rb, cb = linear_sum_assignment(db)
    pnmap = np.empty(KN, dtype=np.int64); pnmap[rn] = cn
    pbmap = np.empty(KB, dtype=np.int64); pbmap[rb] = cb
    return pnmap[nl], pbmap[bl], pnmap, pbmap

def project_nucleus(labels: np.ndarray, stage: Stage):
    Y = labels.reshape(stage.S, stage.E)
    C = np.zeros((stage.KN, stage.KN), dtype=np.int64)
    for s in range(stage.S):
        for m in range(stage.E - 1):
            C[Y[s, m], Y[s, m + 1]] += 1
    r, c = linear_sum_assignment(-C)
    pi = np.empty(stage.KN, dtype=np.int64); pi[r] = c
    pp = permutation_powers(pi, stage.E)
    base = np.empty(stage.S, dtype=np.int64)
    for s in range(stage.S):
        scores = np.asarray([sum(int(pp[m, x] == Y[s, m]) for m in range(stage.E)) for x in range(stage.KN)])
        base[s] = int(np.flatnonzero(scores == scores.max())[0])
    nmap = np.asarray([pp[m, base[s]] for s in range(stage.S) for m in range(stage.E)], dtype=np.int64)
    return base, pi, nmap

def project_bridge(labels: np.ndarray, stage: Stage):
    Y = labels.reshape(stage.R, stage.L)
    u = np.zeros(stage.R, dtype=np.int64); v = Y[0].copy()
    for _ in range(50):
        changed = False
        for r in range(stage.R):
            scores = [int(np.sum(((x + v) % stage.KB) == Y[r])) for x in range(stage.KB)]
            z = int(np.flatnonzero(np.asarray(scores) == max(scores))[0])
            if z != u[r]: u[r] = z; changed = True
        for l in range(stage.L):
            scores = [int(np.sum(((u + x) % stage.KB) == Y[:, l])) for x in range(stage.KB)]
            z = int(np.flatnonzero(np.asarray(scores) == max(scores))[0])
            if z != v[l]: v[l] = z; changed = True
        if not changed: break
    bmap = np.asarray([(u[r] + v[l]) % stage.KB for r in range(stage.R) for l in range(stage.L)], dtype=np.int64)
    return u, v, bmap

def transition_ll(counts, nmap, bmap, src, per_transition=False):
    cnb, cbn, _init, *_ = counts
    _p0, pb, pn = src
    ll = float((cnb * np.log(pb[nmap[:, None], bmap[None, :]] + 1e-300)).sum())
    ll += float((cbn * np.log(pn[bmap[:, None], nmap[None, :]] + 1e-300)).sum())
    if per_transition:
        return ll / max(1, int(cnb.sum() + cbn.sum()))
    return ll

def fit_one(lines_fit, stage: Stage, src, n_restarts: int):
    counts = aggregate(lines_fit, stage.S * stage.E, stage.R * stage.L)
    nf, bf = context_features(counts)
    _cnb, _cbn, _init, ncnt, bcnt = counts
    candidates = []
    for seed in range(n_restarts):
        nl = KMeans(n_clusters=stage.KN, init="k-means++", n_init=32, random_state=seed).fit_predict(nf)
        bl = KMeans(n_clusters=stage.KB, init="k-means++", n_init=32, random_state=1000 + seed).fit_predict(bf)
        if len(np.unique(nl)) != stage.KN or len(np.unique(bl)) != stage.KB: continue
        na, ba, _np, _bp = align_to_source(counts, nl, bl, src)
        base, pi, nmap = project_nucleus(na, stage); u, v, bmap = project_bridge(ba, stage)
        nrec = float((ncnt * (na == nmap)).sum() / max(1, ncnt.sum()))
        brec = float((bcnt * (ba == bmap)).sum() / max(1, bcnt.sum()))
        joint = float(((ncnt * (na == nmap)).sum() + (bcnt * (ba == bmap)).sum()) / max(1, ncnt.sum() + bcnt.sum()))
        ll = transition_ll(counts, nmap, bmap, src, False)
        candidates.append((joint, ll, -seed, seed, base, pi, u, v, nmap, bmap, nrec, brec, na, ba))
    if not candidates: raise RuntimeError("no valid clustering restart")
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    return {"seed": int(best[3]), "base": best[4], "pi": best[5], "u": best[6], "v": best[7],
            "nmap": best[8], "bmap": best[9], "fit_recon_n": float(best[10]), "fit_recon_b": float(best[11]),
            "fit_recon_joint": float(best[0]), "fit_ll": float(best[1]), "cluster_n": best[12], "cluster_b": best[13]}, counts

def weighted_recovery(fit_counts, hold_counts, fit, truth, stage: Stage, mode: str):
    _a, _b, _c, nfit, bfit = fit_counts; _d, _e, _f, nh, bh = hold_counts
    ntrue = truth["nmap"]; btrue = truth["bmap"]
    nmatch = fit["nmap"] == ntrue; bmatch = fit["bmap"] == btrue
    rn = float((nh * nmatch).sum() / max(1, nh.sum())); rb = float((bh * bmatch).sum() / max(1, bh.sum()))
    rall = float(((nh * nmatch).sum() + (bh * bmatch).sum()) / max(1, nh.sum() + bh.sum()))
    nmask = nfit >= 5; bmask = bfit >= 5
    rn5 = float((nh[nmask] * nmatch[nmask]).sum() / max(1, nh[nmask].sum()))
    rb5 = float((bh[bmask] * bmatch[bmask]).sum() / max(1, bh[bmask].sum()))
    rall5 = float(((nh[nmask] * nmatch[nmask]).sum() + (bh[bmask] * bmatch[bmask]).sum()) / max(1, nh[nmask].sum() + bh[bmask].sum()))
    out = {"REC_N": rn, "REC_B": rb, "REC_ALL": rall, "REC_N5": rn5, "REC_B5": rb5, "REC_ALL5": rall5,
           "COV_N": float(nh[nfit > 0].sum() / max(1, nh.sum())), "COV_B": float(bh[bfit > 0].sum() / max(1, bh.sum()))}
    if mode == "POS":
        out["REC_BASE"] = float(np.mean(fit["base"] == truth["base"])); out["REC_PI"] = float(np.mean(fit["pi"] == truth["pi"]))
        vals = []
        for c in range(stage.KB):
            aa = np.mean(fit["u"] == ((truth["u"] + c) % stage.KB)); bb = np.mean(fit["v"] == ((truth["v"] - c) % stage.KB))
            vals.append((stage.R * aa + stage.L * bb) / (stage.R + stage.L))
        out["REC_HALF_GAUGE"] = float(max(vals))
    return out

def truth_source(truth):
    return truth["p0"], truth["pb"], truth["pn"]

def run_replicate(stage: Stage, family: str, rep: int, mode: str):
    lines, latent, truth = generate(stage, family, rep, mode)
    cut = int(stage.lines * 0.8); fit_lines = lines[:cut]; hold_lines = lines[cut:]
    fit, fc = fit_one(fit_lines, stage, truth_source(truth), stage.restarts)
    hc = aggregate(hold_lines, stage.S * stage.E, stage.R * stage.L)
    rec = weighted_recovery(fc, hc, fit, truth, stage, mode)
    true_ll = transition_ll(hc, truth["nmap"], truth["bmap"], truth_source(truth), True)
    fit_ll = transition_ll(hc, fit["nmap"], fit["bmap"], truth_source(truth), True)
    rec.update({"HOLD_LM_TRUE": float(true_ll), "HOLD_LM_FIT": float(fit_ll), "HOLD_REGRET": float(true_ll - fit_ll),
                "fit_recon_n": fit["fit_recon_n"], "fit_recon_b": fit["fit_recon_b"], "fit_recon_joint": fit["fit_recon_joint"], "restart": int(fit["seed"])})
    return {"stage": stage.name, "family": family, "rep": rep, "mode": mode, **rec}

def gate(rows, stage: Stage):
    pos = [r for r in rows if r["mode"] == "POS"]
    adv = {m: [r for r in rows if r["mode"] == m] for m in MODES if m != "POS"}
    c1 = sum(r["REC_ALL"] >= 0.90 for r in pos)
    c2 = sum((r["REC_N"] >= 0.85 and r["REC_B"] >= 0.85) for r in pos)
    c4 = sum((r["REC_ALL5"] >= 0.95 and r["REC_N5"] >= 0.90 and r["REC_B5"] >= 0.90) for r in pos)
    byfam = {}; famok = True
    for f in SOURCE_FAMILIES:
        q = [r for r in pos if r["family"] == f]
        a = sum(r["REC_ALL"] >= 0.90 for r in q); b = sum((r["REC_N"] >= 0.85 and r["REC_B"] >= 0.85) for r in q)
        byfam[f] = {"all_pass": a, "key_pass": b}; famok &= (a >= 2 and b >= 2)
    med_pos = float(np.median([r["HOLD_REGRET"] for r in pos]))
    med_adv = {m: float(np.median([r["HOLD_REGRET"] for r in q])) for m, q in adv.items()}
    sep = all(med_pos < v for v in med_adv.values())
    ok = c1 >= 5 and c2 >= 5 and c4 >= 5 and famok and sep
    return {"stage": stage.name, "n_pos": len(pos), "REC_ALL_pass": c1, "REC_NB_pass": c2, "FREQ_pass": c4, "by_family": byfam,
            "median_positive_regret": med_pos, "median_adversary_regret": med_adv, "adversary_separation": bool(sep), "PASS": bool(ok)}

def smoke():
    st = Stage("SMOKE", 8, 4, 16, 3, 6, 5, 300, 4)
    lines, latent, truth = generate(st, "PEAKED", 99, "POS"); cut = 240
    fit, counts = fit_one(lines[:cut], st, truth_source(truth), st.restarts)
    assert counts[0].shape == (st.S * st.E, st.R * st.L)
    assert len(np.unique(fit["cluster_n"])) == st.KN and len(np.unique(fit["cluster_b"])) == st.KB
    assert np.array_equal(np.sort(fit["pi"]), np.arange(st.KN))
    _u2, _v2, b2 = project_bridge(fit["cluster_b"], st); _u3, _v3, b3 = project_bridge(fit["cluster_b"], st); assert np.array_equal(b2, b3)
    hc = aggregate(lines[cut:], st.S * st.E, st.R * st.L); rec = weighted_recovery(counts, hc, fit, truth, st, "POS")
    print("V12_SMOKE_PASS=" + json.dumps({"shapes": [list(counts[0].shape), list(counts[1].shape)], "restart": fit["seed"], "software_rec_all": rec["REC_ALL"]}, sort_keys=True), flush=True)

def run_stage(stage: Stage):
    rows = []
    for family in SOURCE_FAMILIES:
        for rep in range(3):
            for mode in MODES:
                print("V12_START=" + json.dumps({"stage": stage.name, "family": family, "rep": rep, "mode": mode}, sort_keys=True), flush=True)
                row = run_replicate(stage, family, rep, mode); rows.append(row)
                print("V12_ROW=" + json.dumps(row, sort_keys=True), flush=True)
    g = gate(rows, stage); print("V12_GATE=" + json.dumps(g, sort_keys=True), flush=True); return rows, g

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-only", action="store_true"); args = ap.parse_args()
    smoke()
    if args.smoke_only: return
    print("V12_META=" + json.dumps({"protocol": "VBM_V12_COMPOSITIONAL_TRANSDUCER_PROTOCOL.md", "solver_addendum": "VBM_V12_PREBINDING_SOLVER_ADDENDUM.md", "implementation": "VBM_V12_IMPLEMENTATION_SPEC.md", "stageA": STAGE_A.__dict__, "stageB": STAGE_B.__dict__, "source_families": SOURCE_FAMILIES}, sort_keys=True), flush=True)
    t0 = time.time(); rowsA, gateA = run_stage(STAGE_A)
    if not gateA["PASS"]:
        print("VBM_V12_FINAL_RESULT=" + json.dumps({"verdict": "V12_COMPOSITIONAL_CONSTRAINTS_NOT_IDENTIFYING_AT_V11_SCALE", "stageA": gateA, "stageB_opened": False, "voynich_plaintext_opened": False, "elapsed_s": time.time() - t0}, sort_keys=True), flush=True); return
    rowsB, gateB = run_stage(STAGE_B)
    verdict = "V12_COMPOSITIONAL_TRANSDUCER_IDENTIFIABLE_SYNTHETICALLY" if gateB["PASS"] else "V12_COMPOSITIONAL_TRANSDUCER_FAILS_PRESSURE_TEST"
    print("VBM_V12_FINAL_RESULT=" + json.dumps({"verdict": verdict, "stageA": gateA, "stageB": gateB, "stageB_opened": True, "voynich_plaintext_opened": False, "elapsed_s": time.time() - t0}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
