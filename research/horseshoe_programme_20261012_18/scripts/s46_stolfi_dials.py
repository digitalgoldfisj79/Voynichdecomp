#!/usr/bin/env python3
"""PREREG 31 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): Ed's 3-dial Stolfi device. Crust dial holds (left,right) crust pairs,
mantle dial (left,right) mantle pairs, core disk single cores; reading across a diameter gives crust-mantle-core-mantle-crust.
Rivals: five independent slots (FIVE_N, FIVE_MATCHED). Coverage of held-out tokens, novel types, atom-shuffle decoys; Latin control.
usage: python3 -I s46_stolfi_dials.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VULJ = sys.argv[2]; sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3]]
exec(open(os.path.join(HERE, "s41_prefix.py")).read())   # atoms: K T P F benched gallows, C=ch, S=sh
CORE, MANTLE, CIRC = set("ktpfKTPF"), set("CSe"), set("oay")
def layer(a):
    return 2 if a in CORE else 1 if a in MANTLE else 0   # 0 crust, 1 mantle, 2 core
SEG_LAYER = (0, 1, 2, 1, 0)

def ok_seg(seg, L_, circles):
    return all(layer(a) == L_ or (circles and a in CIRC) for a in seg)
def parses(w, circles):
    n = len(w); out = []
    for i0 in range(n + 1):
        if not ok_seg(w[:i0], 0, circles): break
        for i1 in range(i0, n + 1):
            if not ok_seg(w[i0:i1], 1, circles): break
            for i2 in range(i1, n + 1):
                if not ok_seg(w[i1:i2], 2, circles): break
                for i3 in range(i2, n + 1):
                    if not ok_seg(w[i2:i3], 1, circles): break
                    if ok_seg(w[i3:], 0, circles): out.append((w[:i0], w[i0:i1], w[i1:i2], w[i2:i3], w[i3:]))
    return out
def canonical(w, circles):
    lab, prev = [], 0
    for a in w:
        if circles and a in CIRC: lab.append(prev)
        else: prev = layer(a); lab.append(prev)
    # strict onion check on labels: 0* 1* 2* 1* 0*
    s = "".join(map(str, lab)); m = re.fullmatch(r"(0*)(1*)(2*)(1*)(0*)", s)
    if not m: return None
    b = [len(g) for g in m.groups()]; cut = np.cumsum([0] + b)
    return tuple(w[cut[k]:cut[k + 1]] for k in range(5))

def learn(tc, N, circles, mode):
    comp = [Counter() for _ in range(5)]; pc, pm, core = Counter(), Counter(), Counter()
    for w, c in tc.items():
        p = canonical(w, circles)
        if p is None: continue
        for k in range(5): comp[k][p[k]] += c
        pc[(p[0], p[4])] += c; pm[(p[1], p[3])] += c; core[p[2]] += c
    top = lambda C_, k: {x for x, _ in C_.most_common(k)}
    if mode == "DIAL": return dict(crust=top(pc, N), mantle=top(pm, N), core=top(core, N))
    k_ = N if mode == "FIVE_N" else max(1, round(3 * N / 5))
    return dict(slots=[top(comp[k], k_) for k in range(5)])

def covered(w, dev, circles, mode, cache):
    key = (w, circles)
    ps = cache.get(key)
    if ps is None: ps = cache[key] = parses(w, circles)
    if mode == "DIAL":
        return any((p[0], p[4]) in dev["crust"] and (p[1], p[3]) in dev["mantle"] and p[2] in dev["core"] for p in ps)
    return any(all(p[k] in dev["slots"][k] for k in range(5)) for p in ps)

def evaluate(train, test, label, rng):
    tc, te = Counter(train), Counter(test); novel = [w for w in te if w not in tc]
    decoys = []
    for _ in range(50):
        for w in novel:
            a = list(w)
            for _ in range(10):
                rng.shuffle(a)
                if "".join(a) != w: break
            decoys.append("".join(a))
    cache, rows = {}, []
    for circles in (False, True):
        V = "CIRCLES" if circles else "STRICT"
        parse_rate = sum(te[w] for w in te if parses(w, circles)) / sum(te.values())
        nov_parse = float(np.mean([bool(parses(w, circles)) for w in novel]))
        dec_parse = float(np.mean([bool(parses(d, circles)) for d in decoys]))
        print(label, V, "onion parse: test tokens %.3f novel types %.3f decoys %.3f" % (parse_rate, nov_parse, dec_parse), flush=True)
        for N in (12, 24, 36, 48, 72, 96):
            for mode in ("DIAL", "FIVE_N", "FIVE_MATCHED"):
                dev = learn(tc, N, circles, mode)
                cov = {w: covered(w, dev, circles, mode, cache) for w in te}
                tok = sum(te[w] for w in te if cov[w]) / sum(te.values())
                nov = float(np.mean([cov[w] for w in novel])); dec = float(np.mean([covered(d, dev, circles, mode, cache) for d in decoys]))
                trn = sum(tc[w] for w in tc if covered(w, dev, circles, mode, cache)) / sum(tc.values())
                cells_ = 3 * N if mode != "FIVE_N" else 5 * N
                reach = (N ** 3) if mode == "DIAL" else float(np.prod([len(s) for s in dev["slots"]]))
                row = dict(variant=V, N=N, mode=mode, cells=cells_, parse_rate=parse_rate, novel_parse=nov_parse, decoy_parse=dec_parse,
                           train_token_cov=trn, test_token_cov=tok, novel_type_cov=nov, decoy_cov=dec, reach=float(reach), n_novel=len(novel))
                row["solves"] = bool(N <= 36 and tok >= .95 and nov >= .80 and dec <= .5 * nov)
                rows.append(row)
                print(label, V, mode, N, "train %.3f test %.3f novel %.3f decoy %.3f reach %.3g" % (trn, tok, nov, dec, reach), flush=True)
    return rows

p = os.path.join(OUT, "results_stolfi_dials.json")
rng = np.random.default_rng(SEED + 151)
vms = evaluate([w for l in L(FIT) + L(DEV) for w in l], [w for l in L(TEST) for w in l], "VMS", rng)
Vt = json.load(open(VULJ))["tokens"]
lat = evaluate([atoms(w) for w in Vt[:30000]], [atoms(w) for w in Vt[30000:40000]], "LATIN", rng)
out = dict(VMS=vms, LATIN=lat, vms_solves=[r for r in vms if r["solves"]])
def beats(rows, rival):
    res = {}
    for r in rows:
        if r["mode"] != "DIAL": continue
        o = next(x for x in rows if x["mode"] == rival and x["N"] == r["N"] and x["variant"] == r["variant"])
        res[f'{r["variant"]}_N{r["N"]}'] = (r["novel_type_cov"] - r["decoy_cov"]) - (o["novel_type_cov"] - o["decoy_cov"])
    return res
out["dial_minus_five_matched"] = beats(vms, "FIVE_MATCHED"); out["dial_minus_five_n"] = beats(vms, "FIVE_N")
out["verdict"] = "SOLVES_WORD_PROBLEM" if out["vms_solves"] else "FALSIFIES_STOLFI_DIAL_COVERAGE"
json.dump(out, open(p, "w"), indent=1)
print("dial - five_matched (novel-decoy):", {k: round(v, 3) for k, v in out["dial_minus_five_matched"].items()})
print("dial - five_N (novel-decoy):", {k: round(v, 3) for k, v in out["dial_minus_five_n"].items()})
print("VERDICT", out["verdict"], "| VMS solving configs", len(out["vms_solves"]))
