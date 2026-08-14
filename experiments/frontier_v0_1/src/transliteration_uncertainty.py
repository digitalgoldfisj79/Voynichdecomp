from __future__ import annotations
import argparse, json, math, random
from collections import Counter
from pathlib import Path
from .common import atomic_json, load_config, load_json

def ordered_lines(slim: dict):
    for folio in sorted(slim["pages"]):
        lines = slim["pages"][folio]
        def key(x):
            try:
                return (0, int(x))
            except Exception:
                return (1, str(x))
        for line_no in sorted(lines, key=key):
            yield folio, str(line_no), lines[line_no]

def family_readings(line: dict, reps: dict) -> dict:
    t = line.get("t", {})
    out = {}
    for fam, rid in reps.items():
        s = str(t.get(rid, "") or "").strip()
        if s and "*" not in s:
            out[fam] = " ".join(s.split())
    return out

def entropy_of_support(values):
    c = Counter(values)
    n = sum(c.values())
    if n == 0:
        return None
    return -sum((v/n) * math.log2(v/n) for v in c.values())

def build_lattice(slim_path: Path, config_path: Path, out_dir: Path) -> dict:
    cfg = load_config(config_path)
    slim = load_json(slim_path)
    reps = cfg["u1"]["independent_family_representatives"]
    ref = cfg["u1"]["reference"]
    rows = []
    n_ref = n_admit = 0
    family_cov = Counter()
    for folio, line_no, line in ordered_lines(slim):
        ref_txt = str(line.get("t", {}).get(ref, "") or "").strip()
        if not ref_txt:
            continue
        n_ref += 1
        reads = family_readings(line, reps)
        for fam in reads:
            family_cov[fam] += 1
        admitted = len(reads) >= cfg["u1"]["min_family_readings_per_line"]
        if admitted:
            n_admit += 1
        rows.append({
            "folio": folio,
            "line_no": line_no,
            "unit": line.get("u", ""),
            "reference": " ".join(ref_txt.split()),
            "readings": reads,
            "n_families": len(reads),
            "support_entropy_bits": entropy_of_support(list(reads.values())),
            "admitted": admitted
        })
    coverage = n_admit / n_ref if n_ref else 0
    verdict = "PASS" if coverage >= cfg["u1"]["min_reference_line_coverage"] else "FAIL"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "LINE_LATTICE.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "formal_verdict": verdict,
        "target_opened": False,
        "reference_lines": n_ref,
        "admitted_lines": n_admit,
        "coverage": coverage,
        "family_coverage": dict(family_cov),
        "rule": "one complete line reading per independent transliterator family; correlated boundary/glyph decisions preserved"
    }
    atomic_json(out_dir / "U1_LATTICE_SUMMARY.json", summary)
    return summary

def sample_corpus(lattice_path: Path, seed: int, out_path: Path) -> None:
    rng = random.Random(seed)
    with lattice_path.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
        for line in f:
            row = json.loads(line)
            reads = row["readings"]
            if row["admitted"] and reads:
                fam = rng.choice(sorted(reads))
                txt = reads[fam]
            else:
                fam = "REFERENCE"
                txt = row["reference"]
            g.write(json.dumps({"folio": row["folio"], "line_no": row["line_no"], "family": fam, "text": txt}, ensure_ascii=False, sort_keys=True) + "\n")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--slim", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "PROGRAMME_CONFIG.json")
    s = sub.add_parser("sample")
    s.add_argument("--lattice", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build_lattice(a.slim, a.config, a.out), indent=2))
    else:
        sample_corpus(a.lattice, a.seed, a.out)

if __name__ == "__main__":
    main()
