#!/usr/bin/env python3
"""PREREG 17 (RUNNING_RESULTS_horseshoe.md, 2026-10-15).
Can memoryless unit-level substitution of real plaintext (word / letter / Naibbe-style 1-2 letter chunk /
syllable, with or without k=3 homophones) reproduce the SGT12 within-line exact-repetition geometry
(lag-1 not suppressed, lag-2 enriched)?

Statistic = SGT12 method: within-line exact-token pairs at lag 1 and 2; null = within-line exact-multiset
permutations (200). Each arm is pickled atomically to OUT/arms/<name>.pkl; summary -> OUT/results_unit_substitution_sgt12.json

usage: python3 -I s31_unit_substitution_sgt12.py REPO_ROOT OUT_DIR
"""
import gzip, json, os, pickle, re, sys, zlib
from collections import Counter
import numpy as np

REPO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
NPERM, CAP, SEED = 200, 40000, 20261015
VOW = set("aeiouy")
DIPH = {"ae", "oe", "au", "ei", "eu", "ai", "oi", "ou"}

# ---------------- VMS paragraph lines (ZLZI)
D = json.load(open(os.path.join(REPO, "voynich_transcriptions_slim.json")))
vms_lines = []
for fid in sorted(D["pages"]):
    pg = D["pages"][fid]
    for ln in sorted(pg, key=lambda x: int(x) if x.isdigit() else 99999):
        u = pg[ln].get("u", "")
        if len(u) >= 2 and u[1] == "P":
            toks = pg[ln]["t"].get("ZLZI", "").split()
            toks = [t for t in toks if re.fullmatch(r"[a-z]+", t)]
            if len(toks) >= 2:
                vms_lines.append(toks)
LL = np.array([len(l) for l in vms_lines])

def words(name):
    return gzip.open(os.path.join(REPO, "research/bifolium_order_tests/data/controls", name), "rt").read().split()

def syllabify(w):
    # vowel-nucleus heuristic; diphthongs one nucleus; VCV -> V.CV, VCCV -> VC.CV
    i, nuclei = 0, []
    while i < len(w):
        if w[i] in VOW:
            j = i + 2 if w[i:i + 2] in DIPH else i + 1
            nuclei.append((i, j)); i = j
        else:
            i += 1
    if not nuclei:
        return [w]
    cuts = []
    for (a0, a1), (b0, b1) in zip(nuclei, nuclei[1:]):
        nc = b0 - a1
        cuts.append(a1 if nc <= 1 else a1 + 1)
    out, prev = [], 0
    for c in cuts:
        out.append(w[prev:c]); prev = c
    out.append(w[prev:])
    return out

def units(ws, level, rng):
    s = "".join(ws)
    if level == "WORD":
        return list(ws)
    if level == "LETTER":
        return list(s)
    if level == "CHUNK":
        out, i = [], 0
        while i < len(s):
            k = 2 if (rng.random() < 0.5 and i + 1 < len(s)) else 1
            out.append(s[i:i + k]); i += k
        return out
    if level == "SYLLABLE":
        return [sy for w in ws for sy in syllabify(w)]
    raise ValueError(level)

def to_lines(stream, rng):
    stream = stream[:CAP]; lines, i = [], 0
    while i < len(stream):
        n = int(rng.choice(LL)); lines.append(stream[i:i + n]); i += n
    return [l for l in lines if len(l) >= 2]

def sgt12(lines, rng):
    vocab = {}
    tok = np.array([vocab.setdefault(t, len(vocab)) for l in lines for t in l])
    lid = np.repeat(np.arange(len(lines)), [len(l) for l in lines])
    def count(t):
        return [int(((t[d:] == t[:-d]) & (lid[d:] == lid[:-d])).sum()) for d in (1, 2)]
    obs = count(tok)
    null = np.array([count(tok[np.argsort(lid + rng.random(len(tok)))]) for _ in range(NPERM)])
    res = {"n_units": int(len(tok)), "n_lines": len(lines), "n_types": len(vocab)}
    for k, d in enumerate((1, 2)):
        m, sd = null[:, k].mean(), null[:, k].std(ddof=1)
        res[f"lag{d}"] = dict(obs=obs[k], null_mean=float(m), null_sd=float(sd),
                              ratio=float(obs[k] / m) if m else None, z=float((obs[k] - m) / sd) if sd else None)
    return res

def run_arm(name, make_lines):
    p = os.path.join(OUT, "arms", name + ".pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    rng = np.random.default_rng(zlib.crc32(name.encode()) ^ SEED)
    r = sgt12(make_lines(rng), rng); r["arm"] = name
    pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    print(name, {k: (round(r[k]["ratio"], 3), round(r[k]["z"], 2)) for k in ("lag1", "lag2")}, flush=True)
    return r

def reproduces(r):
    return r["lag1"]["ratio"] >= 0.90 and r["lag2"]["ratio"] >= 1.10 and r["lag2"]["z"] >= 2

results = {"VMS_ZLZI": run_arm("VMS_ZLZI", lambda rng: vms_lines)}
for lang, f in (("LATIN_circa_instans", "circa_instans_words.txt.gz"), ("GREEK_galen", "galen_words.txt.gz")):
    ws = [w for w in words(f) if re.fullmatch(r"[a-z]+", w)]
    for level in ("WORD", "LETTER", "CHUNK", "SYLLABLE"):
        results[f"{lang}_{level}"] = run_arm(f"{lang}_{level}", lambda rng, l=level: to_lines(units(ws, l, rng), rng))
        if level != "WORD":
            def homo(rng, l=level):
                u = units(ws, l, rng)
                return to_lines([f"{x}#{rng.integers(3)}" for x in u], rng)
            results[f"{lang}_{level}_HOMO3"] = run_arm(f"{lang}_{level}_HOMO3", homo)

v = results["VMS_ZLZI"]
gate_vms = 0.90 <= v["lag1"]["ratio"] <= 1.15 and v["lag2"]["ratio"] >= 1.10 and v["lag2"]["z"] >= 3
gate_word = all(results[k]["lag1"]["ratio"] < 0.90 for k in results if k.endswith("_WORD"))
sub = {k: r for k, r in results.items() if k != "VMS_ZLZI" and not k.endswith("_WORD")}
any_rep = [k for k, r in sub.items() if reproduces(r)]
all_l1_supp = all(r["lag1"]["ratio"] < 0.90 for r in sub.values())
if not (gate_vms and gate_word):
    verdict = "VOID_INSTRUMENT_GATE_FAILED"
elif any_rep:
    verdict = "CONFIRMS_UNIT_SUBSTITUTION_CAN_PRODUCE_SGT12"
elif all_l1_supp:
    verdict = "FALSIFIES_UNIT_SUBSTITUTION_PRODUCES_SGT12"
else:
    verdict = "PARTIAL_NO_CLAIM"
summary = dict(prereg="PREREG 17", nperm=NPERM, cap=CAP, seed=SEED, gate_vms=gate_vms, gate_word=gate_word,
               reproducing_arms=any_rep, all_subword_lag1_suppressed=all_l1_supp, verdict=verdict, arms=results)
json.dump(summary, open(os.path.join(OUT, "results_unit_substitution_sgt12.json"), "w"), indent=1)
print("VERDICT", verdict, "gate_vms", gate_vms, "gate_word", gate_word, "reproducing", any_rep)
