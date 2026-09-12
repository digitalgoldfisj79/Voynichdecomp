#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

PROGRAMME = "yiddish_m0_invariants_v01_20260912"
MS262_URL = "https://github.com/cu-mkp/ms-262-data.git"
MS262_COMMIT = "1e7b9f78ae1b4d2bc3d6c2c593d0d14855bc664f"
PPCHY_URL = "https://github.com/beatrice57/penn-parsed-corpus-of-historical-yiddish.git"
PPCHY_COMMIT = "b5864bd02a315c1d436a82553667bbf81eab6537"
OUT = Path(__file__).resolve().parent / "run_output_control"
SRC = OUT / "sources"
N_REPS = 32
WIN = 512
LONG_WIN = 2048
CELL_PASS = 29

LEAF_RE = re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def public_seed(*parts) -> int:
    payload = "|".join([PROGRAMME, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def clone_at(url: str, commit: str, dest: Path) -> str:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    got = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if got != commit:
        raise RuntimeError(f"commit mismatch {dest}: {got} != {commit}")
    return got


def tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def render_node(node: ET.Element, include_exp: bool) -> str:
    pieces = []
    if node.text:
        pieces.append(node.text)
    for child in list(node):
        t = tag_name(child.tag)
        if t == "lb":
            pieces.append(" ")
        elif t == "exp" and not include_exp:
            # D0: editor-supplied expansion content is not a source atom.
            pass
        else:
            pieces.append(render_node(child, include_exp))
        if child.tail:
            pieces.append(child.tail)
    return "".join(pieces)


def declared_word(raw: str) -> str:
    # Frozen atomisation: raw code points in Unicode Letter/Mark categories only.
    return "".join(ch for ch in raw if unicodedata.category(ch)[:1] in {"L", "M"})


def tokenize(text: str):
    words = []
    skipped = 0
    for raw in text.split():
        w = declared_word(raw)
        if w:
            words.append(w)
        else:
            skipped += 1
    return words, skipped


def extract_ms262(repo: Path, include_exp: bool):
    all_words = []
    per_page = []
    skipped_total = 0
    nested = collections.Counter()
    exp_count = 0
    files = []
    for pno in range(1, 22):
        for side in ("r", "v"):
            page = f"{pno:03d}{side}"
            p = repo / "xml" / "transcription" / f"tc_p{page}.xml"
            if not p.exists():
                raise RuntimeError(f"missing frozen page {p}")
            files.append(p)
            root = ET.parse(p).getroot()
            page_words = []
            page_skipped = 0
            page_abs = 0
            for ab in root.iter():
                if tag_name(ab.tag) != "ab" or ab.attrib.get("language") != "owy":
                    continue
                page_abs += 1
                exp_count += sum(1 for x in ab.iter() if tag_name(x.tag) == "exp")
                text = render_node(ab, include_exp=include_exp)
                ws, sk = tokenize(text)
                page_words.extend(ws)
                page_skipped += sk
                for x in ab.iter():
                    tn = tag_name(x.tag)
                    if tn in {"heb", "jit", "lad", "lat", "arm", "laz"}:
                        nws, _ = tokenize(render_node(x, include_exp=include_exp))
                        nested[tn] += len(nws)
            all_words.extend(page_words)
            skipped_total += page_skipped
            per_page.append({"page": page, "owy_ab_count": page_abs, "words": len(page_words), "skipped_nonatom_tokens": page_skipped, "sha256": file_sha(p)})
    return all_words, {
        "files": len(files),
        "word_count": len(all_words),
        "skipped_nonatom_tokens": skipped_total,
        "nested_marked_word_counts": dict(sorted(nested.items())),
        "exp_element_count": exp_count,
        "per_page": per_page,
    }


def local_pattern(word: str):
    ids = {}
    nxt = 0
    out = []
    for ch in word:
        if ch not in ids:
            ids[ch] = nxt
            nxt += 1
        out.append(ids[ch])
    return tuple(out)


def token_partition(words):
    ids = {}
    nxt = 0
    out = []
    for w in words:
        if w not in ids:
            ids[w] = nxt
            nxt += 1
        out.append(ids[w])
    return tuple(out)


def global_canonical_stream(words):
    ids = {}
    nxt = 0
    stream = []
    for wi, w in enumerate(words):
        if wi:
            stream.append(-1)
        for ch in w:
            if ch not in ids:
                ids[ch] = nxt
                nxt += 1
            stream.append(ids[ch])
    return tuple(stream)


def hash_tuple(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def entropy_from_counts(counts):
    n = sum(counts)
    if n <= 0:
        return None
    ans = 0.0
    for c in sorted(counts):
        if c:
            p = c / n
            ans -= p * math.log2(p)
    return ans


def recurrence_gaps(words):
    last = {}
    gaps = []
    for i, w in enumerate(words):
        if w in last:
            gaps.append(i - last[w])
        last[w] = i
    return gaps


def gap_hist(gaps):
    bins = [0] * 9
    for g in gaps:
        if g == 1: bins[0] += 1
        elif g == 2: bins[1] += 1
        elif g <= 4: bins[2] += 1
        elif g <= 8: bins[3] += 1
        elif g <= 16: bins[4] += 1
        elif g <= 32: bins[5] += 1
        elif g <= 64: bins[6] += 1
        elif g <= 128: bins[7] += 1
        else: bins[8] += 1
    return bins


def exact_structures(words):
    return {
        "canonical_symbol_stream_hash": hash_tuple(global_canonical_stream(words)),
        "word_lengths_hash": hash_tuple(tuple(map(len, words))),
        "token_equality_partition_hash": hash_tuple(token_partition(words)),
        "within_word_patterns_hash": hash_tuple(tuple(local_pattern(w) for w in words)),
    }


def summary_vector(words):
    if not words:
        return {"status": "ABSTAIN_EMPTY"}
    lengths = [len(w) for w in words]
    atoms = [c for w in words for c in w]
    ac = collections.Counter(atoms)
    tc = collections.Counter(words)
    pat = collections.Counter(local_pattern(w) for w in words)
    gaps = recurrence_gaps(words)
    n_atom = len(atoms)
    n_word = len(words)
    sorted_atom_counts = sorted(ac.values(), reverse=True)
    sorted_token_counts = sorted(tc.values(), reverse=True)
    length_hist = [0] * 13
    for x in lengths:
        length_hist[x - 1 if 1 <= x <= 12 else 12] += 1
    adjacent_total = sum(max(0, len(w) - 1) for w in words)
    adjacent_same = sum(sum(a == b for a, b in zip(w, w[1:])) for w in words)
    repeat_word_count = sum(any(v > 1 for v in collections.Counter(w).values()) for w in words)
    coll = None
    if n_atom >= 2:
        coll = sum(c * (c - 1) for c in ac.values()) / (n_atom * (n_atom - 1))
    patt = [[list(k), v] for k, v in sorted(pat.items(), key=lambda kv: (len(kv[0]), kv[0]))]
    return {
        "status": "OK",
        "n_words": n_word,
        "n_atoms": n_atom,
        "alphabet_cardinality": len(ac),
        "sorted_atom_frequency_counts": sorted_atom_counts,
        "atom_collision_probability": coll,
        "atom_entropy_bits": entropy_from_counts(ac.values()),
        "word_length_hist_1_12_13plus": length_hist,
        "word_length_mean": sum(lengths) / n_word,
        "word_length_pstdev": statistics.pstdev(lengths) if lengths else None,
        "token_type_count": len(tc),
        "type_token_ratio": len(tc) / n_word,
        "hapax_token_fraction": sum(c for c in tc.values() if c == 1) / n_word,
        "singleton_type_fraction": sum(1 for c in tc.values() if c == 1) / len(tc) if tc else None,
        "top1_token_mass": sum(sorted_token_counts[:1]) / n_word,
        "top5_token_mass": sum(sorted_token_counts[:5]) / n_word,
        "top10_token_mass": sum(sorted_token_counts[:10]) / n_word,
        "token_entropy_bits": entropy_from_counts(tc.values()),
        "fraction_words_with_repeated_atoms": repeat_word_count / n_word,
        "adjacent_identical_atom_fraction": adjacent_same / adjacent_total if adjacent_total else None,
        "within_word_pattern_hist_complete": patt,
        "recurrence_gap_hist": gap_hist(gaps),
        "median_recurrence_gap": statistics.median(gaps) if gaps else None,
        "n_recurrence_gaps": len(gaps),
    }


def fingerprint(words):
    ex = exact_structures(words)
    su = summary_vector(words)
    return {"exact": ex, "summary": su, "summary_hash": sha256_bytes(canonical_json(su).encode("utf-8"))}


def observed_atoms(words):
    return sorted(set(c for w in words for c in w), key=ord)


def permute_symbols(words, rng):
    atoms = observed_atoms(words)
    vals = atoms.copy()
    rng.shuffle(vals)
    mp = dict(zip(atoms, vals))
    inv = {v: k for k, v in mp.items()}
    enc = ["".join(mp[c] for c in w) for w in words]
    dec = ["".join(inv[c] for c in w) for w in enc]
    return enc, dec


def word_order_shuffle(words, rng):
    x = list(words)
    rng.shuffle(x)
    return x


def within_word_shuffle(words, rng):
    out = []
    for w in words:
        x = list(w)
        rng.shuffle(x)
        out.append("".join(x))
    return out


def boundary_perturb(words, rng):
    n = len(words)
    if n < 20:
        return None
    candidates = list(range(n - 1))
    rng.shuffle(candidates)
    merge_starts = set()
    target_m = max(1, int(0.10 * (n - 1)))
    for i in candidates:
        if i in merge_starts or i - 1 in merge_starts or i + 1 in merge_starts:
            continue
        merge_starts.add(i)
        if len(merge_starts) >= target_m:
            break
    eligible_split = [i for i, w in enumerate(words) if len(w) >= 4 and i not in merge_starts and i - 1 not in merge_starts]
    rng.shuffle(eligible_split)
    split_idxs = set(eligible_split[:max(1, int(0.10 * len(eligible_split)))]) if eligible_split else set()
    out = []
    i = 0
    while i < n:
        if i in merge_starts and i + 1 < n:
            out.append(words[i] + words[i + 1])
            i += 2
            continue
        w = words[i]
        if i in split_idxs and len(w) >= 4:
            cut = 1 + rng.randrange(len(w) - 1)
            if cut >= len(w): cut = len(w) - 1
            out.extend([w[:cut], w[cut:]])
        else:
            out.append(w)
        i += 1
    return out


def symbol_merge(words):
    c = collections.Counter(ch for w in words for ch in w)
    if len(c) < 2:
        return None
    ordered = sorted(c.items(), key=lambda kv: (-kv[1], ord(kv[0])))
    a, b = ordered[0][0], ordered[1][0]
    return [w.replace(b, a) for w in words]


def normalize_leaf(raw):
    if raw.startswith("*") or raw in {"0", "-NONE-"}:
        return []
    raw = raw.replace("@", "").split("^", 1)[0]
    out = []
    for part in raw.split("_"):
        w = "".join(c for c in part.lower() if "a" <= c <= "z")
        if w:
            out.append(w)
    return out


def extract_penn(path: Path):
    out = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    for tag, raw in LEAF_RE.findall(txt):
        if tag.startswith(("ID", "CODE", "PUNC")):
            continue
        out.extend(normalize_leaf(raw))
    return out


def penn_sensitivity(repo: Path):
    files = [
        "1507w-bovo.psd", "1579e-shir-preface.psd", "1579e-shir.psd",
        "1588e-letters-cracow.psd", "1589e-ester-preface.psd", "1589e-ester.psd",
        "1590e-sam-hayyim.psd", "1620e-lev-tov-1.psd", "1648w-kine.psd",
    ]
    rows = []
    for fn in files:
        p = repo / "data" / fn
        if not p.exists():
            rows.append({"file": fn, "status": "MISSING"})
            continue
        ws = extract_penn(p)
        row = {"file": fn, "words": len(ws), "sha256": file_sha(p), "complete_512_windows": len(ws) // WIN}
        if len(ws) >= WIN:
            row["first_512_fingerprint"] = fingerprint(ws[:WIN])
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    script_hash = file_sha(Path(__file__))
    ms_repo = SRC / "ms262"
    penn_repo = SRC / "ppchy"
    ms_commit = clone_at(MS262_URL, MS262_COMMIT, ms_repo)
    penn_commit = clone_at(PPCHY_URL, PPCHY_COMMIT, penn_repo)

    d0, meta0 = extract_ms262(ms_repo, include_exp=False)
    d1, meta1 = extract_ms262(ms_repo, include_exp=True)

    fixtures = {}
    fixtures["empty"] = fingerprint([])
    one = ["אאא"] * 16
    fixtures["one_symbol"] = fingerprint(one)
    unique = [chr(0x05D0 + (i % 20)) + chr(0x05D0 + ((i + 1) % 20)) + str(i) for i in range(20)]
    # digits are not part of a real extracted control; make unique Letter-only words deterministically.
    unique = [chr(0x05D0 + (i % 20)) + chr(0x0400 + i) for i in range(20)]
    fixtures["all_unique"] = fingerprint(unique)
    fixture_ok = fixtures["empty"]["summary"].get("status") == "ABSTAIN_EMPTY" and fixtures["all_unique"]["summary"].get("median_recurrence_gap") is None

    windows = [d0[i:i + WIN] for i in range(0, len(d0) - WIN + 1, WIN)]
    long_windows = [d0[i:i + LONG_WIN] for i in range(0, len(d0) - LONG_WIN + 1, LONG_WIN)]
    if not windows:
        raise RuntimeError("no complete primary 512-word window")

    positive_rows = []
    destructive_rows = []
    all_positive_exact = True
    for wi, w in enumerate(windows):
        base = fingerprint(w)
        for rep in range(N_REPS):
            rng = random.Random(public_seed(MS262_COMMIT, wi, rep, "m0_plant"))
            enc, dec = permute_symbols(w, rng)
            roundtrip = dec == w
            fp = fingerprint(enc)
            exact_equal = fp["exact"] == base["exact"]
            summary_equal = fp["summary"] == base["summary"] and fp["summary_hash"] == base["summary_hash"]
            passed = roundtrip and exact_equal and summary_equal
            all_positive_exact &= passed
            positive_rows.append({"window": wi, "rep": rep, "roundtrip": roundtrip, "exact_equal": exact_equal, "summary_equal": summary_equal, "pass": passed})

            transforms = {
                "WORD_ORDER_SHUFFLE": word_order_shuffle(w, random.Random(public_seed(MS262_COMMIT, wi, rep, "word_order"))),
                "WITHIN_WORD_SHUFFLE": within_word_shuffle(w, random.Random(public_seed(MS262_COMMIT, wi, rep, "within_word"))),
                "BOUNDARY_PERTURB": boundary_perturb(w, random.Random(public_seed(MS262_COMMIT, wi, rep, "boundary"))),
                "SYMBOL_MERGE": symbol_merge(w),
            }
            for family, tw in transforms.items():
                if tw is None:
                    destructive_rows.append({"window": wi, "rep": rep, "family": family, "eligible": False})
                    continue
                tfp = fingerprint(tw)
                exact_changed = tfp["exact"] != base["exact"]
                summary_changed = tfp["summary_hash"] != base["summary_hash"]
                destructive_rows.append({"window": wi, "rep": rep, "family": family, "eligible": True, "exact_changed": exact_changed, "summary_changed": summary_changed, "detected": exact_changed or summary_changed})

    cells = {}
    for r in destructive_rows:
        k = f"{r['window']}|{r['family']}"
        x = cells.setdefault(k, {"window": r["window"], "family": r["family"], "n": 0, "detected": 0, "ineligible": 0})
        if not r.get("eligible"):
            x["ineligible"] += 1
        else:
            x["n"] += 1
            x["detected"] += int(r.get("detected", False))
    nondegenerate = True
    for x in cells.values():
        x["pass"] = (x["n"] == 0 and x["ineligible"] == N_REPS) or (x["n"] == N_REPS and x["detected"] >= CELL_PASS)
        nondegenerate &= x["pass"]

    d01 = []
    n_shared = min(len(d0), len(d1)) // WIN
    for wi in range(n_shared):
        a = d0[wi * WIN:(wi + 1) * WIN]
        b = d1[wi * WIN:(wi + 1) * WIN]
        fa, fb = fingerprint(a), fingerprint(b)
        d01.append({
            "window": wi,
            "D0_D1_exact_equal": fa["exact"] == fb["exact"],
            "D0_D1_summary_equal": fa["summary"] == fb["summary"],
            "D0_summary_hash": fa["summary_hash"],
            "D1_summary_hash": fb["summary_hash"],
        })

    penn_rows = penn_sensitivity(penn_repo)

    if not fixture_ok:
        state = "IMPLEMENTATION_INVALID"
    elif not all_positive_exact:
        state = "IMPLEMENTATION_INVALID"
    elif not nondegenerate:
        state = "M0_INVARIANT_INSTRUMENT_DEGENERATE"
    else:
        state = "M0_INVARIANT_INSTRUMENT_IMPLEMENTATION_PASS__SOURCE_TRANSFER_UNASSESSED"

    summary = {
        "programme": PROGRAMME,
        "state": state,
        "target_loaded": False,
        "script_sha256": script_hash,
        "source_commits": {"ms262": ms_commit, "ppchy": penn_commit},
        "D0": meta0,
        "D1": meta1,
        "earlier_census_expected_contiguous_words": 4262,
        "D0_count_delta_vs_earlier_census": meta0["word_count"] - 4262,
        "complete_512_windows": len(windows),
        "complete_2048_windows": len(long_windows),
        "fixture_ok": fixture_ok,
        "positive_trials": len(positive_rows),
        "positive_passes": sum(int(r["pass"]) for r in positive_rows),
        "positive_all_exact": all_positive_exact,
        "destructive_cells": sorted(cells.values(), key=lambda x: (x["window"], x["family"])),
        "nondegeneracy_pass": nondegenerate,
        "D0_D1_sensitivity": d01,
        "penn_secondary_sensitivity": penn_rows,
        "bound": "Exact deterministic implementation counts on one named MS262 transcription and registered seeded transforms; windows are dependent pieces of one manuscript. No historical-population CI. Null SD not applicable to exact M0 identity checks.",
    }

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "positive_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in positive_rows), encoding="utf-8")
    (OUT / "destructive_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in destructive_rows), encoding="utf-8")

    report = []
    report.append("# Yiddish M0 invariant control-stage result\n")
    report.append(f"\n**State: `{state}`**\n")
    report.append("\nVoynich target loaded: **NO**.\n")
    report.append(f"\nD0 MS262 words: **{meta0['word_count']}**; earlier census expectation 4,262; delta **{meta0['word_count'] - 4262:+d}**. Complete 512 windows: **{len(windows)}**; complete 2,048 windows: **{len(long_windows)}**.\n")
    report.append(f"\nPlanted M0 exact identity: **{summary['positive_passes']}/{summary['positive_trials']}**. Null SD: not applicable; this is a deterministic identity requirement.\n")
    report.append(f"\nRegistered destructive-control nondegeneracy gate: **{'PASS' if nondegenerate else 'FAIL'}**.\n")
    for x in summary["destructive_cells"]:
        report.append(f"- window {x['window']} {x['family']}: {x['detected']}/{x['n']} detected; ineligible={x['ineligible']}; pass={x['pass']}\n")
    neq = sum(not x["D0_D1_summary_equal"] for x in d01)
    report.append(f"\nD0 vs D1 editorial-expansion sensitivity: summary differs in **{neq}/{len(d01)}** matched 512-word windows. This is representation sensitivity, not a language result.\n")
    report.append("\nBound: exact deterministic control results on one named manuscript; the 512-word windows are not independent historical works. No population CI is licensed. Source transfer remains unassessed.\n")
    (OUT / "REPORT.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps({"state": state, "D0_words": meta0["word_count"], "positive": [summary["positive_passes"], summary["positive_trials"]], "windows_512": len(windows), "windows_2048": len(long_windows), "nondegeneracy": nondegenerate}, sort_keys=True))


if __name__ == "__main__":
    main()
