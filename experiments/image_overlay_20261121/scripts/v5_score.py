"""v4 Phase 2 scoring: unblind A/B with the sealed key; CORRECT/PARTIAL/WRONG per method; paired sign test.
Usage: python3 -I v5_score.py <out_dir> -> out/v4/test/v4_scores.csv"""
import csv, json, math, os, sys, collections
OUT = sys.argv[1]; V = os.path.join(OUT, "v4", "test")
key = {d["id"]: d for d in json.load(open(os.path.join(V, "SEALED_key.json")))["key"]}
rows = list(csv.DictReader(open(os.path.join(V, "verdicts_AB.csv"))))
pairs = []
for r in rows:
    k = key[int(r["id"])]
    v4, v3 = (r["A"], r["B"]) if k["left_is_v4"] else (r["B"], r["A"])
    pairs.append(dict(id=int(r["id"]), char=k["char"], word=k["word"], v3=v3, v4=v4))
jp = [p for p in pairs if p["v3"] != "UNJUDGEABLE"]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 3), round(c - h, 3), round(c + h, 3)
def binom_two_sided(k, n):
    from math import comb
    p = sum(comb(n, i) for i in range(0, n + 1) if comb(n, i) <= comb(n, k)) / 2 ** n
    return min(1.0, p)
out = []
for m in ("v3", "v4"):
    c = collections.Counter(p[m] for p in jp)
    out.append(dict(method=m, judged=len(jp), correct=c["CORRECT"], partial=c["PARTIAL"], wrong=c["WRONG"],
                    correct_share=wilson(c["CORRECT"], len(jp)), wrong_share=wilson(c["WRONG"], len(jp))))
only4 = sum(1 for p in jp if p["v4"] == "CORRECT" and p["v3"] != "CORRECT"); only3 = sum(1 for p in jp if p["v3"] == "CORRECT" and p["v4"] != "CORRECT")
rank = {"WRONG": 0, "PARTIAL": 1, "CORRECT": 2}
better = sum(1 for p in jp if rank[p["v4"]] > rank[p["v3"]]); worse = sum(1 for p in jp if rank[p["v4"]] < rank[p["v3"]])
print("unjudgeable", len(pairs) - len(jp))
for o in out: print(o)
print("CORRECT only under v4:", only4, "| only under v3:", only3, "| sign test p = %.5f" % binom_two_sided(min(only4, only3), only4 + only3))
print("any improvement (W<P<C): v4 better", better, "worse", worse, "| p = %.5f" % binom_two_sided(min(better, worse), better + worse))
by = collections.defaultdict(lambda: [0, 0, 0])
for p in jp:
    g = "gallows" if p["char"] in "ktpf" else ("o next to gallows or other" if p["char"] in "o" else p["char"])
    by[g][0] += 1; by[g][1] += p["v3"] == "CORRECT"; by[g][2] += p["v4"] == "CORRECT"
print("by target (n, v3 correct, v4 correct):", sorted(by.items(), key=lambda kv: -kv[1][0]))
with open(os.path.join(V, "v4_pairs_unblinded.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
