#!/usr/bin/env python3
"""VSN-B4-v1.1 source-only audit and metrics runner.

Changes from v1, made while the source corpus is still gated and before any
new Voynich target access:
  * H(next|prev) now exactly matches the B3 implementation (no boundary tokens).
  * positional entropy is character-count weighted exactly as in B3.
  * edit-1 location uses B3's exact prefix/last-character/internal rule rather
    than quartile buckets.
  * preregistered C2 and C3 controls are implemented.

This program MUST NOT read Voynich data. Until every required f.8v row has an
A/B transcription, it returns BLOCKED_SOURCE_TRANSCRIPTION and does not run
binding controls.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

NS = "VSN_B4_V1"
REQUIRED = ["group_id", "compound_norm", "text_conf", "syll1", "syll2", "syll3", "syll4"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy(vals) -> float:
    c = Counter(vals); n = sum(c.values())
    return -sum(v/n * math.log2(v/n) for v in c.values()) if n else 0.0


def surface_metrics(strings: list[str]):
    """Exact metric definitions copied from VSN-B3 state_gated_k2_v1.py."""
    lens = [len(t) for t in strings]
    chars = [c for t in strings for c in t]
    byp = defaultdict(list); byr = defaultdict(list); bg = []; bp = defaultdict(list)
    for t in strings:
        for i, c in enumerate(t):
            byp[i+1].append(c)
            byr[len(t)-i].append(c)
        for a, b in zip(t, t[1:]):
            bg.append((a, b)); bp[a].append(b)
    nc = sum(lens)
    h_abs = sum(len(v)/nc * entropy(v) for v in byp.values()) if nc else 0.0
    h_right = sum(len(v)/nc * entropy(v) for v in byr.values()) if nc else 0.0
    hnext = sum(len(v)/len(bg) * entropy(v) for v in bp.values()) if bg else 0.0
    return {
        "mean_len": sum(lens)/len(lens) if lens else 0.0,
        "hnext": hnext,
        "rml": h_right - h_abs,
        "h_abs": h_abs,
        "h_right": h_right,
        "h_char": entropy(chars),
    }


def edit_pairs(strings: list[str]):
    """Exact edit-1 construction/location semantics copied from VSN-B3."""
    toks = sorted(set(strings)); S = set(toks); pairs = set()
    for w in toks:
        for i in range(len(w)):
            d = w[:i] + w[i+1:]
            if d in S:
                pairs.add(tuple(sorted((w, d))))
    B = defaultdict(list)
    for w in toks:
        for i in range(len(w)):
            B[(len(w), i, w[:i], w[i+1:])].append(w)
    for ws in B.values():
        if len(ws) > 1:
            ws = sorted(set(ws))
            for i in range(len(ws)):
                for j in range(i+1, len(ws)):
                    pairs.add((ws[i], ws[j]))
    loc = Counter()
    for a, b in pairs:
        if len(a) == len(b):
            k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            p = "prefix" if k == 0 else ("suffix" if k == len(a)-1 else "internal")
        else:
            long, short = (a, b) if len(a) > len(b) else (b, a)
            poss = [i for i in range(len(long)) if long[:i] + long[i+1:] == short]
            pcs = [("prefix" if i == 0 else ("suffix" if i == len(long)-1 else "internal")) for i in poss]
            p = pcs[0] if pcs and all(x == pcs[0] for x in pcs) else "internal"
        loc[p] += 1
    n = len(pairs)
    degree = Counter({w: 0 for w in toks})
    for a, b in pairs:
        degree[a] += 1; degree[b] += 1
    return pairs, {
        "pairs": n,
        "mean_degree": mean(degree.values()) if degree else 0.0,
        "isolated_fraction": sum(v == 0 for v in degree.values()) / len(degree) if degree else 0.0,
        "prefix": loc["prefix"]/n if n else 0.0,
        "internal": loc["internal"]/n if n else 0.0,
        "suffix": loc["suffix"]/n if n else 0.0,
    }


def complete_row(r):
    if r.get("text_conf") not in {"A", "B"}:
        return False
    return all(r.get(k) and r[k].strip().upper() != "U" for k in REQUIRED)


def compute_metrics(rows):
    strings = [r["compound_norm"].replace(" ", "").lower() for r in rows]
    slots = [[r[f"syll{i}"].lower() for r in rows] for i in range(1, 5)]
    sm = surface_metrics(strings); _, em = edit_pairs(strings)
    all_components = [x for slot in slots for x in slot]
    return {
        "n": len(strings),
        "surface": sm,
        "edit1": em,
        "slot_entropy_bits": [entropy(s) for s in slots],
        "initial_component_entropy_bits": entropy(slots[0]),
        "final_component_entropy_bits": entropy(slots[3]),
        "component_reuse": dict(Counter(all_components)),
    }


def rng_for(label: str):
    seed = int.from_bytes(hashlib.sha256((NS + "|" + label).encode()).digest()[:8], "big")
    return random.Random(seed), seed


def c0_slot_marginal(rows, rng):
    slots = [[r[f"syll{i}"].lower() for r in rows] for i in range(1, 5)]
    for s in slots: rng.shuffle(s)
    return ["".join(slots[i][j] for i in range(4)) for j in range(len(rows))]


def c1_order(rows, rng):
    out = []
    for r in rows:
        s = [r[f"syll{i}"].lower() for i in range(1, 5)]
        rng.shuffle(s); out.append("".join(s))
    return out


def c2_within_component_chars(rows, rng):
    out = []
    for r in rows:
        parts = []
        for i in range(1, 5):
            chars = list(r[f"syll{i}"].lower()); rng.shuffle(chars); parts.append("".join(chars))
        out.append("".join(parts))
    return out


def c3_iid_chars(rows, rng):
    observed = [r["compound_norm"].replace(" ", "").lower() for r in rows]
    pool = [c for s in observed for c in s]
    return ["".join(rng.choice(pool) for _ in range(len(s))) for s in observed]


def control_summary(strings):
    sm = surface_metrics(strings); _, em = edit_pairs(strings)
    return {"hnext": sm["hnext"], "rml": sm["rml"], "mean_len": sm["mean_len"],
            "edit1_pairs": em["pairs"], "edit_prefix": em["prefix"],
            "edit_internal": em["internal"], "edit_suffix": em["suffix"]}


def quantile(xs, q):
    ys = sorted(xs)
    if not ys: return None
    pos = (len(ys)-1)*q; lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return ys[lo]
    return ys[lo]*(hi-pos) + ys[hi]*(pos-lo)


def run_controls(rows, nrep):
    observed = control_summary([r["compound_norm"].replace(" ", "").lower() for r in rows])
    makers = [
        ("C0_slot_marginal_shuffle", c0_slot_marginal),
        ("C1_within_codeword_order_shuffle", c1_order),
        ("C2_within_component_character_shuffle", c2_within_component_chars),
        ("C3_matched_length_iid_character_marginal", c3_iid_chars),
    ]
    out = {}
    for name, maker in makers:
        vals = []; first_seed = None
        for i in range(nrep):
            rng, seed = rng_for(f"{name}|{i}")
            if first_seed is None: first_seed = seed
            vals.append(control_summary(maker(rows, rng)))
        fields = vals[0].keys()
        out[name] = {
            "replicates": nrep,
            "first_seed": first_seed,
            "observed": observed,
            "null": {k: {"mean": mean(v[k] for v in vals),
                         "q025": quantile([v[k] for v in vals], .025),
                         "median": quantile([v[k] for v in vals], .5),
                         "q975": quantile([v[k] for v in vals], .975)} for k in fields},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--output", type=Path, default=Path("b4_source_audit_v1_1.json"))
    ap.add_argument("--controls", type=int, default=10000)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    complete = [r for r in rows if complete_row(r)]
    unresolved = [r["group_id"] for r in rows if not complete_row(r)]
    out = {
        "namespace": "VSN-B4-v1.1", "seed_namespace": NS,
        "input": str(args.csv_path), "input_sha256": sha256_file(args.csv_path),
        "total_rows": len(rows), "complete_AB_rows": len(complete),
        "unresolved_rows": unresolved, "voynich_data_accessed": False,
        "metric_compatibility": "Exact B3 surface_metrics/edit_pairs definitions",
    }
    if complete: out["draft_metrics_complete_rows_only"] = compute_metrics(complete)
    if unresolved:
        out.update(status="BLOCKED_SOURCE_TRANSCRIPTION", binding_controls_run=False)
        args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(out, indent=2))
        if args.require_complete: raise SystemExit(3)
        return
    out.update(status="SOURCE_COMPLETE", binding_metrics=compute_metrics(rows),
               binding_controls=run_controls(rows, args.controls), binding_controls_run=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
