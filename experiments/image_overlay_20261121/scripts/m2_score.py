"""Step 2 scoring: join verdicts to the sealed sample; Wilson intervals by stratum, est flag and character.
Usage: python3 -I m2_score.py <out_dir> -> out/step2/step2_scores.csv"""
import csv, json, math, os, sys, collections
OUT = sys.argv[1]; S = os.path.join(OUT, "step2")
items = {it["id"]: it for it in json.load(open(os.path.join(S, "step2_sample.json")))["items"]}
V = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(S, "verdicts.csv")))}
assert set(V) == set(items), (len(V), len(items))
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
rows = []
def report(name, sel):
    js = [i for i in items if sel(items[i]) and V[i]["verdict"] != "UNJUDGEABLE"]; c = collections.Counter(V[i]["verdict"] for i in js)
    un = sum(1 for i in items if sel(items[i]) and V[i]["verdict"] == "UNJUDGEABLE")
    p, lo, hi = wilson(c["CORRECT"], len(js)); pw, lw, hw = wilson(c["WRONG"], len(js))
    rows.append(dict(group=name, judged=len(js), unjudgeable=un, correct=c["CORRECT"], partial=c["PARTIAL"], wrong=c["WRONG"],
                     correct_share=round(p, 3), ci_low=round(lo, 3), ci_high=round(hi, 3), wrong_share=round(pw, 3), wrong_ci_low=round(lw, 3), wrong_ci_high=round(hw, 3)))
report("A: all glyphs", lambda it: it["stratum"] == "A")
report("B: narrow o a e i y", lambda it: it["stratum"] == "B")
report("A, flagged approximate cut", lambda it: it["stratum"] == "A" and it["est"] == 1)
report("A, not flagged", lambda it: it["stratum"] == "A" and it["est"] == 0)
report("A+B, flagged", lambda it: it["est"] == 1)
report("A+B, not flagged", lambda it: it["est"] == 0)
for ch in "oaeiydklhtqcnrs":
    report("A+B char " + ch, lambda it, ch=ch: it["char"] == ch)
with open(os.path.join(S, "step2_scores.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
