#!/usr/bin/env python3
"""Bounding checks for PREREG 32 (post hoc, flagged): (i) does 'known training word + one slip' do as well as 'device + one slip'
on the same novel types and decoys, and token coverage of each closure; (ii) mantle pair support size (is 0.944 trivial?)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s47_salvage.py")).read().split('R1 = stage("S1_layout", S1)')[0])
ALPH = sorted({a for w in tc for a in w})
def edits1(w):
    s = set()
    for i in range(len(w) + 1):
        for a in ALPH: s.add(w[:i] + a + w[i:])
        if i < len(w):
            s.add(w[:i] + w[i + 1:])
            for a in ALPH: s.add(w[:i] + a + w[i + 1:])
    s.discard(w); s.discard(""); return s
V = set(tc); d = learn(tc, 36, False, "FIVE_N")
models = {"VOCAB+SLIP": lambda w: w in V or any(v in V for v in edits1(w)),
          "DEVICE+SLIP": lambda w: cov_stolfi(w, d) or any(cov_stolfi(v, d) for v in edits1(w)),
          "DEVICE_OR_VOCAB+SLIP": lambda w: w in V or cov_stolfi(w, d) or any(v in V or cov_stolfi(v, d) for v in edits1(w))}
for nm, f in models.items():
    cv = {w: f(w) for w in te}
    tok = sum(te[w] for w in te if cv[w]) / sum(te.values())
    print(nm, "test tokens %.3f novel %.3f decoy %.3f" % (tok, np.mean([cv[w] for w in novel]), np.mean([f(x) for x in DEC])), flush=True)
fitW = [w for l in L(FIT) for w in l]
for kind in ("stolfi_crust", "stolfi_mantle"):
    c = Counter(pairs_from(fitW, kind)); tot = sum(c.values())
    print(kind, "distinct pairs", len(c), "top-36 mass %.3f" % (sum(v for _, v in c.most_common(36)) / tot))
