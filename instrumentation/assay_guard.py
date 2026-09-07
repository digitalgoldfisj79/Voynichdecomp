#!/usr/bin/env python3
import argparse, hashlib, json, math, pathlib, sys
from typing import Any, Dict


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def load_json(path: str) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def dump_json(path: str, obj: Any) -> None:
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return {"rate": p, "lo": max(0.0, ctr-half), "hi": min(1.0, ctr+half)}


def effect_over_null_sd(effect: float, null_sd: float) -> float:
    if null_sd < 0:
        raise ValueError("null_sd cannot be negative")
    if null_sd == 0:
        return math.inf if effect != 0 else 0.0
    return abs(effect) / null_sd


def freeze(manifest: Dict[str, Any], out: str) -> Dict[str, Any]:
    required = ["assay_id", "version", "scientific_question", "null_source", "alternatives", "target", "decision", "licensed_inference", "prohibited_inference"]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    if manifest["target"].get("sealed_before_qualification") is not True:
        raise ValueError("target.sealed_before_qualification must be true")
    decision = manifest["decision"]
    for k in ["max_null_rejection_rate", "min_alternative_rejection_rate", "min_effect_over_null_sd"]:
        if k not in decision:
            raise ValueError(f"decision missing {k}")
    rec = {
        "kind": "ASSAY_FREEZE",
        "assay_id": manifest["assay_id"],
        "version": manifest["version"],
        "manifest_sha256": sha256_obj(manifest),
        "target_sealed": True,
        "lifecycle_stage": "FROZEN",
    }
    dump_json(out, rec)
    return rec


def qualify(manifest: Dict[str, Any], freeze_rec: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    msha = sha256_obj(manifest)
    if freeze_rec.get("manifest_sha256") != msha:
        raise ValueError("manifest hash does not match freeze record")
    if summary.get("manifest_sha256") != msha:
        raise ValueError("summary manifest_sha256 does not match frozen manifest")
    decision = manifest["decision"]
    nr = summary["null_validation"]
    null_rej = int(nr["rejected"]); null_n = int(nr["n"])
    null_ci = wilson(null_rej, null_n)
    null_gate = null_ci["rate"] <= float(decision["max_null_rejection_rate"])

    alt_results = []
    all_alt = True
    alt_by_name = {x["name"]: x for x in summary.get("alternatives", [])}
    expected = [x["name"] for x in manifest["alternatives"]]
    if set(alt_by_name) != set(expected):
        raise ValueError(f"alternative set mismatch: expected {expected}, got {sorted(alt_by_name)}")

    for name in expected:
        x = alt_by_name[name]
        rej = int(x["rejected"]); n = int(x["n"])
        ci = wilson(rej, n)
        e = float(x["effect"]); nsd = float(x["null_sd"])
        z = effect_over_null_sd(e, nsd)
        rate_ok = ci["rate"] >= float(decision["min_alternative_rejection_rate"])
        effect_ok = z >= float(decision["min_effect_over_null_sd"])
        passed = bool(rate_ok and effect_ok)
        all_alt = all_alt and passed
        alt_results.append({
            "name": name,
            "rejected": rej,
            "n": n,
            "rejection_rate": ci["rate"],
            "wilson95": [ci["lo"], ci["hi"]],
            "effect": e,
            "null_sd": nsd,
            "effect_over_null_sd": z,
            "rate_gate_pass": rate_ok,
            "effect_gate_pass": effect_ok,
            "pass": passed,
            "headline": ("THE METRIC DOES NOT RESOLVE THIS DEPARTURE" if z < 2 else "resolved at >=2 null SD"),
        })

    repr_checks = summary.get("representation_checks", [])
    repr_required = bool(decision.get("require_representation_robustness", False))
    repr_pass = all(bool(x.get("pass")) for x in repr_checks) if repr_checks else (not repr_required)

    leakage = summary.get("leakage_checks", {})
    leakage_required = ["target_inaccessible_during_fit", "disjoint_seed_namespaces", "training_only_model_selection"]
    leakage_pass = all(leakage.get(k) is True for k in leakage_required)

    pass_all = bool(null_gate and all_alt and repr_pass and leakage_pass)
    out = {
        "kind": "ASSAY_QUALIFICATION",
        "assay_id": manifest["assay_id"],
        "version": manifest["version"],
        "manifest_sha256": msha,
        "summary_sha256": sha256_obj(summary),
        "lifecycle_stage": "ADVERSARIAL_POWERED" if pass_all else "BLOCKED",
        "known_source": {
            "rejected": null_rej,
            "n": null_n,
            "false_rejection_rate": null_ci["rate"],
            "wilson95": [null_ci["lo"], null_ci["hi"]],
            "gate_pass": null_gate,
        },
        "alternatives": alt_results,
        "representation_gate_pass": repr_pass,
        "leakage_gate_pass": leakage_pass,
        "overall_pass": pass_all,
        "target_open_permitted": pass_all,
        "licensed_inference": manifest["licensed_inference"] if pass_all else "NO TARGET INFERENCE LICENSED",
        "prohibited_inference": manifest["prohibited_inference"],
    }
    out["qualification_sha256"] = sha256_obj(out)
    return out


def check_target(manifest: Dict[str, Any], qualification: Dict[str, Any]) -> None:
    if qualification.get("manifest_sha256") != sha256_obj(manifest):
        raise ValueError("qualification does not match current manifest")
    if qualification.get("overall_pass") is not True or qualification.get("target_open_permitted") is not True:
        raise PermissionError("TARGET SEALED: assay has not passed qualification")


def main() -> None:
    ap = argparse.ArgumentParser(description="Voynich assay qualification guard")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("freeze"); p.add_argument("manifest"); p.add_argument("out")
    p = sp.add_parser("qualify"); p.add_argument("manifest"); p.add_argument("freeze_record"); p.add_argument("summary"); p.add_argument("out")
    p = sp.add_parser("check-target"); p.add_argument("manifest"); p.add_argument("qualification")
    args = ap.parse_args()
    try:
        if args.cmd == "freeze":
            rec = freeze(load_json(args.manifest), args.out); print(json.dumps(rec, sort_keys=True))
        elif args.cmd == "qualify":
            obj = qualify(load_json(args.manifest), load_json(args.freeze_record), load_json(args.summary)); dump_json(args.out, obj); print(json.dumps(obj, sort_keys=True))
        elif args.cmd == "check-target":
            check_target(load_json(args.manifest), load_json(args.qualification)); print("TARGET_OPEN_PERMITTED")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
