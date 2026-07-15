#!/usr/bin/env python3
"""Cross-solver analysis for corrected v0.3.1 control audits."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"{path}: missing results")
    return payload


def trial_key(row: dict) -> tuple:
    return (
        int(row["trial_index"]),
        int(row["seed"]),
        str(row["control_family"]),
        str(row["length_profile"]),
        str(row.get("true_selector")),
    )


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    payloads = {}
    for value in args.input:
        if "=" not in value:
            raise SystemExit("--input requires NAME=PATH")
        name, path = value.split("=", 1)
        payloads[name] = load(Path(path))

    indexed = {
        name: {trial_key(row): row for row in payload["results"]}
        for name, payload in payloads.items()
    }
    inventories = {name: set(rows) for name, rows in indexed.items()}
    reference_name = sorted(inventories)[0]
    reference = inventories[reference_name]
    inventory_equal = all(keys == reference for keys in inventories.values())
    inventory_differences = {
        name: {
            "missing_vs_reference": sorted(reference - keys),
            "extra_vs_reference": sorted(keys - reference),
        }
        for name, keys in inventories.items()
        if keys != reference
    }

    legacy_sets = {
        name: {key for key, row in rows.items() if bool(row.get("false_positive"))}
        for name, rows in indexed.items()
    }
    strict_sets = {
        name: {
            key
            for key, row in rows.items()
            if bool(row.get("remediation_audit", {}).get("strict_cipher_selected"))
        }
        for name, rows in indexed.items()
    }

    names = sorted(indexed)
    pairwise = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairwise[f"{left}&{right}"] = {
                "legacy_intersection": len(legacy_sets[left] & legacy_sets[right]),
                "strict_intersection": len(strict_sets[left] & strict_sets[right]),
            }

    all_legacy = set.intersection(*(legacy_sets[name] for name in names)) if names else set()
    all_strict = set.intersection(*(strict_sets[name] for name in names)) if names else set()

    per_solver = {}
    for name, rows in indexed.items():
        by_family = defaultdict(lambda: {"trials": 0, "legacy_fp": 0, "strict_fp": 0})
        by_length = defaultdict(lambda: {"trials": 0, "legacy_fp": 0, "strict_fp": 0})
        production_models = Counter()
        fp_production_models = Counter()
        heldout = []
        legacy_heldout = []
        strict_heldout = []
        for key, row in rows.items():
            audit = row.get("remediation_audit", {})
            legacy = bool(row.get("false_positive"))
            strict = bool(audit.get("strict_cipher_selected"))
            family = str(row["control_family"])
            length = str(row["length_profile"])
            by_family[family]["trials"] += 1
            by_family[family]["legacy_fp"] += int(legacy)
            by_family[family]["strict_fp"] += int(strict)
            by_length[length]["trials"] += 1
            by_length[length]["legacy_fp"] += int(legacy)
            by_length[length]["strict_fp"] += int(strict)
            model = str(audit.get("production_model"))
            production_models[model] += 1
            if legacy:
                fp_production_models[model] += 1
            difference = audit.get("heldout_cipher_minus_production_bits_per_token")
            if difference is not None and math.isfinite(float(difference)):
                heldout.append(float(difference))
                if legacy:
                    legacy_heldout.append(float(difference))
                if strict:
                    strict_heldout.append(float(difference))
        per_solver[name] = {
            "trials": len(rows),
            "legacy_false_positives": len(legacy_sets[name]),
            "strict_false_positives": len(strict_sets[name]),
            "by_family": dict(sorted(by_family.items())),
            "by_length": dict(sorted(by_length.items())),
            "production_models": dict(production_models),
            "legacy_fp_production_models": dict(fp_production_models),
            "heldout_difference_bits_per_token": quantiles(heldout),
            "legacy_fp_heldout_difference_bits_per_token": quantiles(legacy_heldout),
            "strict_fp_heldout_difference_bits_per_token": quantiles(strict_heldout),
        }

    report = {
        "programme": "morpholocal-calibration-v0.3.1-remediation-control-overlap",
        "formal": False,
        "solvers": names,
        "inventory_equal": inventory_equal,
        "inventory_differences": inventory_differences,
        "per_solver": per_solver,
        "pairwise": pairwise,
        "all_solver_legacy_intersection": len(all_legacy),
        "all_solver_strict_intersection": len(all_strict),
        "all_solver_legacy_trials": [list(key) for key in sorted(all_legacy)],
        "all_solver_strict_trials": [list(key) for key in sorted(all_strict)],
        "legacy_union": len(set.union(*(legacy_sets[name] for name in names))) if names else 0,
        "strict_union": len(set.union(*(strict_sets[name] for name in names))) if names else 0,
    }
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# v0.3.1 corrected control audit",
        "",
        f"- Identical trial inventory: **{inventory_equal}**",
        f"- Solvers: {', '.join(names)}",
        f"- All-solver legacy FP intersection: **{len(all_legacy)}**",
        f"- All-solver strict FP intersection: **{len(all_strict)}**",
        f"- Legacy FP union: **{report['legacy_union']}**",
        f"- Strict FP union: **{report['strict_union']}**",
        "",
        "## Solver totals",
        "",
        "| Solver | Trials | Legacy FP | Strict FP |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        row = per_solver[name]
        lines.append(
            f"| {name} | {row['trials']} | {row['legacy_false_positives']} | "
            f"{row['strict_false_positives']} |"
        )
    lines.extend(["", "## Pairwise overlap", ""])
    for pair, values in pairwise.items():
        lines.append(
            f"- {pair}: legacy {values['legacy_intersection']}; "
            f"strict {values['strict_intersection']}"
        )
    lines.extend(["", "## All-solver legacy false-positive trials", ""])
    if all_legacy:
        for key in sorted(all_legacy):
            lines.append(f"- `{key}`")
    else:
        lines.append("None.")
    args.markdown_output.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
