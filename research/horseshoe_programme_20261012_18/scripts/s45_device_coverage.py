#!/usr/bin/env python3
"""PREREG 30 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): does a realistic ring device's reachable vocabulary contain the
held-out Voynich vocabulary? FREE vs COUPLED (notched) rings; decoy (atom-shuffled) specificity; Latin control.
usage: python3 -I s45_device_coverage.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VULJ = sys.argv[2]; sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3]]
exec(open(os.path.join(HERE, "s41_prefix.py")).read())
MAXCHUNK = 4

def tokenize_atoms(w):  # atoms are single characters after atoms(); Latin: letters
    return list(w)

def learn(train_counts, R, C):
    rings = [set() for _ in range(R)]; parses = {}
    for w, _ in sorted(train_counts.items(), key=lambda x: -x[1]):
        a = tokenize_atoms(w); n = len(a); best = [None, None]
        def rec(k, pos, parts, new):
            if best[0] is not None and new >= best[0]: return
            if k == R - 1:
                ch = "".join(a[pos:])
                if len(a) - pos > MAXCHUNK: return
                nn = new + (1 if ch and ch not in rings[k] else 0)
                if best[0] is None or nn < best[0]: best[0], best[1] = nn, parts + [ch]
                return
            for e in range(pos, min(n, pos + MAXCHUNK) + 1):
                ch = "".join(a[pos:e]); rec(k + 1, e, parts + [ch], new + (1 if ch and ch not in rings[k] else 0))
        rec(0, 0, [], 0)
        if best[1] is None: continue
        need = [(k, ch) for k, ch in enumerate(best[1]) if ch and ch not in rings[k]]
        if all(len(rings[k]) < C for k, _ in need):
            for k, ch in need: rings[k].add(ch)
            parses[w] = best[1]
    pairs = set()
    for p in parses.values():
        for k in range(R - 1): pairs.add((k, p[k], p[k + 1]))
    return rings, parses, pairs

def coverable(w, rings, pairs=None):
    a = tokenize_atoms(w); n = len(a); R = len(rings)
    from functools import lru_cache
    @lru_cache(None)
    def ok(k, pos, prev):
        if k == R: return pos == n
        for e in range(pos, min(n, pos + MAXCHUNK) + 1):
            ch = "".join(a[pos:e])
            if ch and ch not in rings[k]: continue
            if pairs is not None and k > 0 and (k - 1, prev, ch) not in pairs: continue
            if ok(k + 1, e, ch): return True
        return False
    return ok(0, 0, None)

def reach_size(rings, pairs=None):
    cells = [sorted(r) + [""] for r in rings]
    if pairs is None:
        return float(np.prod([len(c) for c in cells]))
    cnt = {c: 1.0 for c in cells[0]}
    for k in range(1, len(cells)):
        cnt = {c: sum(v for p, v in cnt.items() if (k - 1, p, c) in pairs) for c in cells[k]}
    return float(sum(cnt.values()))

def evaluate(train_tokens, test_tokens, label, rng):
    tc = Counter(train_tokens); te = Counter(test_tokens); novel = [w for w in te if w not in tc]
    decoys = []
    for _ in range(50):
        for w in novel:
            a = list(w)
            for _ in range(10):
                rng.shuffle(a)
                if "".join(a) != w: break
            decoys.append("".join(a))
    res = []
    for R in (3, 4, 5, 6):
        for C in (12, 24, 36, 48):
            rings, parses, pairs = learn(tc, R, C)
            for variant, pr in (("FREE", None), ("COUPLED", pairs)):
                cov = {w: coverable(w, rings, pr) for w in te}
                tok = sum(te[w] for w in te if cov[w]) / sum(te.values())
                nov = float(np.mean([cov[w] for w in novel])) if novel else float("nan")
                dec = float(np.mean([coverable(d, rings, pr) for d in decoys]))
                train_tok = sum(tc[w] for w in tc if coverable(w, rings, pr)) / sum(tc.values())
                row = dict(R=R, C=C, variant=variant, cells=sum(len(r) for r in rings), train_token_cov=train_tok,
                           test_token_cov=tok, novel_type_cov=nov, decoy_cov=dec, reachable=reach_size(rings, pr), n_novel=len(novel))
                row["solves"] = bool(R <= 6 and C <= 36 and tok >= .95 and nov >= .80 and dec <= .5 * nov)
                res.append(row)
                print(label, variant, R, C, "train %.3f test %.3f novel %.3f decoy %.3f reach %.3g" % (train_tok, tok, nov, dec, row["reachable"]), flush=True)
    return res

p = os.path.join(OUT, "results_device_coverage.json")
rng = np.random.default_rng(SEED + 141)
vms = evaluate([w for l in L(FIT) + L(DEV) for w in l], [w for l in L(TEST) for w in l], "VMS", rng)
V = json.load(open(VULJ))["tokens"]
lat = evaluate(V[:30000], V[30000:40000], "LATIN", rng)
out = dict(VMS=vms, LATIN=lat, vms_solves=[r for r in vms if r["solves"]], latin_solves=[r for r in lat if r["solves"]])
out["verdict"] = "SOLVES_WORD_PROBLEM" if out["vms_solves"] else "FALSIFIES_REALISTIC_DEVICE_COVERAGE"
json.dump(out, open(p, "w"), indent=1)
print("VERDICT", out["verdict"], "| VMS solving configs", len(out["vms_solves"]), "| Latin solving configs", len(out["latin_solves"]))
