#!/usr/bin/env python3
"""PREREG 23 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): slot-wise periodicity signature of a rotating volvelle.
Per naive slot, within-line recurrence excess (z vs within-line permutation null) at lags 1..6;
PI = mean z(even lags) - mean z(odd lags); SD by line bootstrap. Positive/negative synthetic controls run first.
Each arm pickled atomically to OUT/arms/. usage: python3 -I s38_volvelle_periodicity.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import json, os, pickle, re, sys
import numpy as np

REPO, VUL, OUT = sys.argv[1:4]; os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
SEED, NPERM, NBOOT, LAGS = 20261016, 200, 200, range(1, 7)
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
def atoms(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w
SLOTS = {"WHOLE": lambda a: a, "FIRST": lambda a: a[:1], "LAST": lambda a: a[-1:], "LAST2": lambda a: a[-2:]}

D = json.load(open(os.path.join(REPO, "voynich_transcriptions_slim.json")))
vms_lines = []
for fid in sorted(D["pages"]):
    pg = D["pages"][fid]
    for ln in sorted(pg, key=lambda x: int(x) if x.isdigit() else 99999):
        u = pg[ln].get("u", "")
        if len(u) >= 2 and u[1] == "P":
            t = [atoms(x) for x in pg[ln]["t"].get("ZLZI", "").split() if re.fullmatch(r"[a-z]+", x)]
            if len(t) >= 2: vms_lines.append(t)
LL = np.array([len(l) for l in vms_lines])
rng0 = np.random.default_rng(SEED)

def pseudo_lines(tokens, rng):
    out, i = [], 0
    while i < len(tokens):
        n = int(rng.choice(LL)); out.append(tokens[i:i + n]); i += n
    return [l for l in out if len(l) >= 2]

V = json.load(open(VUL))["tokens"][:60000]
vul_lines = pseudo_lines(V, np.random.default_rng(SEED + 1))

# ---- synthetic volvelles (Cham toy architecture)
VMS_ATOMS = sorted({c for l in vms_lines for w in l for c in w})
def make_volvelle(rng, n=24):
    rings = []
    for r in range(4):
        pool = list(rng.choice(VMS_ATOMS, size=3, replace=False))
        rings.append([("".join(rng.choice(pool, size=rng.integers(1, 3))) if i % 2 == 0 else "") for i in range(n)])
    return rings
def synth(rotating, rng, ntok=34000):
    vols = [make_volvelle(rng) for _ in range(2)]; pos = [[int(rng.integers(24)) for _ in range(4)] for _ in range(2)]
    toks = []
    while len(toks) < ntok:
        v = int(rng.integers(2))
        if not rotating: pos[v] = [int(rng.integers(24)) for _ in range(4)]
        w = "".join(vols[v][r][pos[v][r]] for r in range(4))
        if w: toks.append(w)
        if rotating:
            for r in range(4):
                if rng.random() < 0.7: pos[v][r] = (pos[v][r] + 1) % 24
    return pseudo_lines(toks, rng)

def lag_profile(lines, slot, rng):
    vals = [[SLOTS[slot](w) for w in l] for l in lines]
    voc = {}; tok = np.array([voc.setdefault(x, len(voc)) for l in vals for x in l]); lid = np.repeat(np.arange(len(vals)), [len(l) for l in vals])
    def counts(t): return np.array([((t[d:] == t[:-d]) & (lid[d:] == lid[:-d])).sum() for d in LAGS], float)
    obs = counts(tok); null = np.array([counts(tok[np.argsort(lid + rng.random(len(tok)))]) for _ in range(NPERM)])
    z = (obs - null.mean(0)) / null.std(0, ddof=1); ratio = obs / null.mean(0)
    return z, ratio
def pi(z): return float(np.mean(z[[1, 3, 5]]) - np.mean(z[[0, 2, 4]]))

def run_arm(name, lines):
    p = os.path.join(OUT, "arms", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    rng = np.random.default_rng(SEED + 7); res = {}
    for slot in SLOTS:
        z, ratio = lag_profile(lines, slot, rng)
        boots = []
        for _ in range(NBOOT):
            idx = rng.integers(len(lines), size=len(lines)); bl = [lines[i] for i in idx]
            zb, _ = lag_profile_fast(bl, slot, rng); boots.append(pi(zb))
        sd = float(np.std(boots, ddof=1))
        res[slot] = dict(z=[float(x) for x in z], ratio=[float(x) for x in ratio], PI=pi(z), PI_sd=sd, PI_over_sd=pi(z) / sd)
        print(name, slot, "PI/SD", round(pi(z) / sd, 2), "z by lag", [round(float(x), 1) for x in z], flush=True)
    pickle.dump(res, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return res

def lag_profile_fast(lines, slot, rng, nperm=40):  # bootstrap uses fewer permutations
    global NPERM
    keep, NPERM = NPERM, nperm
    try: return lag_profile(lines, slot, rng)
    finally: NPERM = keep

R = {"SYN_ROTATING": run_arm("SYN_ROTATING", synth(True, np.random.default_rng(SEED + 2))),
     "SYN_RANDOM": run_arm("SYN_RANDOM", synth(False, np.random.default_rng(SEED + 3)))}
pos_ok = max(v["PI_over_sd"] for v in R["SYN_ROTATING"].values()) >= 3
neg_ok = all(abs(v["PI_over_sd"]) < 2 for v in R["SYN_RANDOM"].values())
R["VMS"] = run_arm("VMS", vms_lines); R["VULGATE"] = run_arm("VULGATE", vul_lines)
if not (pos_ok and neg_ok): verdict = "VOID_CONTROLS_FAILED"
elif any(R["VMS"][s]["PI_over_sd"] >= 3 and R["VULGATE"][s]["PI_over_sd"] < 2 for s in SLOTS): verdict = "CONFIRMS_ROTATION_SIGNATURE"
elif all(R["VMS"][s]["PI_over_sd"] < 2 for s in SLOTS): verdict = "FALSIFIES_STEPWISE_ROTATION"
else: verdict = "PARTIAL"
R["controls"] = dict(positive_ok=pos_ok, negative_ok=neg_ok); R["verdict"] = verdict
json.dump(R, open(os.path.join(OUT, "results_volvelle_periodicity.json"), "w"), indent=1)
print("CONTROLS", pos_ok, neg_ok, "VERDICT", verdict)
