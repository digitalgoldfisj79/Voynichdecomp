"""Fragment-count feasibility check on the HF crop manifest (runs where the manifest is mounted).
Usage: python3 h1_fragment_counts.py /data/results/corpus_crop_manifest.jsonl > h1_out.json
"""
import collections, json, sys

G = ["cth", "ckh", "cph", "cfh", "ch", "sh"]
def units(s):
    out = []; i = 0
    while i < len(s):
        g = next((x for x in G if s.startswith(x, i)), None); out.append(g or s[i]); i += len(g) if g else 1
    return out

words = {}                               # (folio, word_index) -> word
n = {"cc": {}, "ccmerge": {}}            # kind -> (folio, word_index) -> n_slots
rows = 0; kinds = collections.Counter()
for line in open(sys.argv[1]):
    r = json.loads(line); rows += 1; kinds[(r["kind"], r["view"])] += 1
    key = (r["folio"], r["word_index"])
    if r["kind"] == "word": words[key] = r["word"]
    elif r["kind"] in n and r["view"] == "orig": n[r["kind"]][key] = r["n_slots"]

def summary(kind, unit):
    diffs = []; by_len = collections.defaultdict(list)
    for key, w in words.items():
        if not w: continue
        u = len(w) if unit == "raw" else len(units(w))
        p = n[kind].get(key, 0)
        diffs.append(p - u); by_len[min(u, 9)].append(p - u)
    N = len(diffs); c = collections.Counter(diffs)
    return dict(kind=kind, unit=unit, words=N, exact=sum(d == 0 for d in diffs) / N, within1=sum(abs(d) <= 1 for d in diffs) / N,
                fewer=sum(d < 0 for d in diffs) / N, more=sum(d > 0 for d in diffs) / N,
                zero_pieces=sum(1 for key in words if words[key] and n[kind].get(key, 0) == 0) / N,
                mean_diff=sum(diffs) / N, median_pieces=sorted(n[kind].get(k, 0) for k in words)[len(words) // 2],
                hist={str(k): c[k] for k in range(-6, 7)},
                exact_by_units={str(k): [len(v), round(sum(d == 0 for d in v) / len(v), 4)] for k, v in sorted(by_len.items())})

out = dict(rows=rows, kinds={"%s/%s" % k: v for k, v in kinds.items()}, n_words=len(words),
           folios=len({k[0] for k in words}),
           results=[summary(k, u) for k in ("cc", "ccmerge") for u in ("raw", "grouped")])
print(json.dumps(out))
