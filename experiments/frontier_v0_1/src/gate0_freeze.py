from __future__ import annotations
import argparse, json
from pathlib import Path
from .common import GateFailure, atomic_json, find_first, load_config, load_json, load_records, sha256_file

def check_fold_manifest(path: Path) -> dict:
    obj = load_json(path)
    rows = obj.get("rows", obj if isinstance(obj, list) else None)
    if not isinstance(rows, list) or not rows:
        raise GateFailure("fold manifest must be a non-empty list or {'rows': [...]} object")
    seen = {}
    folds = set()
    for row in rows:
        b = row.get("bifolium") or row.get("bifolium_id") or row.get("physical_bifolium")
        f = row.get("fold")
        if b is None or f is None:
            raise GateFailure("fold row missing bifolium/fold")
        folds.add(int(f))
        if b in seen and seen[b] != int(f):
            raise GateFailure(f"physical bifolium split across folds: {b}")
        seen[b] = int(f)
    if len(folds) != 5:
        raise GateFailure(f"expected exactly five folds, observed {sorted(folds)}")
    return {"n_rows": len(rows), "n_bifolia": len(seen), "folds": sorted(folds)}

def run(repo_root: Path, fold_manifest: Path, config_path: Path, out: Path) -> dict:
    cfg = load_config(config_path)
    recs = load_records(repo_root, cfg)
    can = cfg["canonical"]
    if len(recs) != can["expected_record_count"]:
        raise GateFailure(f"record count {len(recs)} != {can['expected_record_count']}")
    missing = sorted(set(can["required_record_fields"]) - set(recs[0]))
    if missing:
        raise GateFailure(f"missing required record fields: {missing}")
    sections = sorted({r["section"] for r in recs})
    unknown = sorted(set(sections) - set(can["allowed_sections"]))
    if unknown:
        raise GateFailure(f"unknown section values: {unknown}")

    slim = find_first(repo_root, can["slim_paths"])
    sobj = load_json(slim)
    if "pages" not in sobj or "transcribers" not in sobj:
        raise GateFailure("multi-transliterator slim container missing pages/transcribers")
    available = {x["id"] for x in sobj["transcribers"]}
    needed = set(cfg["u1"]["independent_family_representatives"].values())
    missing_trans = sorted(needed - available)
    if missing_trans:
        raise GateFailure(f"missing U1 family representatives: {missing_trans}")

    p70 = find_first(repo_root, can["p70c_paths"])
    pobj = load_json(p70)
    entries = pobj.get("entries", [])
    if len(entries) != can["expected_p70c_entries"]:
        raise GateFailure(f"P70-C entries {len(entries)} != {can['expected_p70c_entries']}")

    fold_info = check_fold_manifest(fold_manifest)
    inputs = [find_first(repo_root, can["record_paths"]), slim, p70, fold_manifest, config_path]
    manifest = {str(p): sha256_file(p) for p in inputs}
    result = {
        "formal_verdict": "PASS",
        "target_opened": False,
        "record_count": len(recs),
        "sections": sections,
        "slim_transliterator_count": len(available),
        "fold_info": fold_info,
        "input_sha256": manifest
    }
    atomic_json(out / "GATE0_RESULT.json", result)
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for p, h in sorted(manifest.items()):
            f.write(f"{h}  {p}\n")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--fold-manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "PROGRAMME_CONFIG.json")
    ap.add_argument("--out", type=Path, default=Path("results/gate0"))
    a = ap.parse_args()
    try:
        print(json.dumps(run(a.repo_root, a.fold_manifest, a.config, a.out), indent=2))
    except GateFailure as e:
        print(json.dumps({"formal_verdict":"FAIL","target_opened":False,"error":str(e)}, indent=2))
        raise SystemExit(2)

if __name__ == "__main__":
    main()
