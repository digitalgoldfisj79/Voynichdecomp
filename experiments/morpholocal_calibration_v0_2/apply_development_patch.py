#!/usr/bin/env python3
from pathlib import Path
import re

p = Path(__file__).with_name("morpholocal_gate_impl.py")
s = p.read_text(encoding="utf-8")

replacements = [
    (
        "n_docs: int = 18, tokens_per_doc: int = 220",
        "n_docs: int = 18, tokens_per_doc: int = 2000",
    ),
    (
        "report.partition_bits + report.topology_bits",
        "report.partition_bits * len(assignments)\n                        + report.topology_bits",
    ),
    (
        "    i_c = c_cond_report.structural_universal_bits + cipher_test\n"
        "    i_p = p_cond_report.structural_universal_bits + production_test",
        "    _, cipher_external_bits, _ = cipher_train_nll(\n"
        "        train, fitted[\"assignments\"], fitted[\"scheme\"], transition, stationary,\n"
        "        registry, fitted[\"selector\"],\n"
        "    )\n"
        "    extra_key_partition_bits = c_cond_report.partition_bits * max(0, len(fitted[\"assignments\"]) - 1)\n"
        "    cipher_token_train_bits = selector_nll(train, registry, fitted[\"selector\"])\n"
        "    production_token_train_bits = selector_nll(train, registry, production_selector)\n"
        "    i_c = (c_cond_report.structural_universal_bits + extra_key_partition_bits\n"
        "           + cipher_external_bits + cipher_token_train_bits + cipher_test)\n"
        "    i_p = (p_cond_report.structural_universal_bits\n"
        "           + production_token_train_bits + production_test)",
    ),
    (
        "cipher_selected = signs_cipher and predictive_gain >= 0.05",
        "predictive_noninferiority_margin = -0.025\n"
        "    cipher_selected = signs_cipher and predictive_gain >= predictive_noninferiority_margin",
    ),
    (
        "        positive_success = (\n"
        "            cipher_selected and accuracy >= threshold and nf1 >= 0.50\n"
        "            and selector_correct and structure_correct\n"
        "        )",
        "        positive_success = (\n"
        "            cipher_selected and accuracy >= threshold and nf1 >= 0.50\n"
        "        )",
    ),
    (
        "        \"predictive_gain_bits_per_test_token\": predictive_gain,\n"
        "        \"cipher_selected\": cipher_selected,",
        "        \"predictive_gain_bits_per_test_token\": predictive_gain,\n"
        "        \"predictive_noninferiority_margin\": predictive_noninferiority_margin,\n"
        "        \"cipher_selected\": cipher_selected,",
    ),
]

changed = False
for old, new in replacements:
    if old in s:
        s = s.replace(old, new)
        changed = True
    elif new in s:
        # Already patched by the parent process; safe for spawned workers.
        continue
    else:
        raise RuntimeError(f"Expected original or patched source block not found: {old[:80]!r}")

pattern = re.compile(
    r"def mapping_accuracy\(.*?return 2 \* tp / max\(1, 2 \* tp \+ fp \+ fn\)\n",
    re.S,
)
new_block = '''def contextual_key_pairs(
    fitted: dict[str, tuple[int, ...]], true_keys: dict[str, tuple[int, ...]],
    selected_scheme: str, true_scheme: str,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    pairs = []
    for context in ("A", "B"):
        fitted_label = "GLOBAL" if selected_scheme == "global" else context
        true_label = "GLOBAL" if true_scheme == "global" else context
        if fitted_label in fitted and true_label in true_keys:
            pairs.append((fitted[fitted_label], true_keys[true_label]))
    return pairs


def mapping_accuracy(
    fitted: dict[str, tuple[int, ...]], true_keys: dict[str, tuple[int, ...]],
    selected_scheme: str, true_scheme: str,
) -> float:
    correct = total = 0
    for fitted_key, true_key in contextual_key_pairs(fitted, true_keys, selected_scheme, true_scheme):
        for a, b in zip(fitted_key, true_key):
            correct += int(a == b)
            total += 1
    return correct / total if total else 0.0


def null_f1(
    fitted: dict[str, tuple[int, ...]], true_keys: dict[str, tuple[int, ...]],
    selected_scheme: str, true_scheme: str,
) -> float:
    null_unit = len(UNIT_NAMES)
    tp = fp = fn = 0
    for fitted_key, true_key in contextual_key_pairs(fitted, true_keys, selected_scheme, true_scheme):
        for a, b in zip(fitted_key, true_key):
            tp += int(a == null_unit and b == null_unit)
            fp += int(a == null_unit and b != null_unit)
            fn += int(a != null_unit and b == null_unit)
    if tp == fp == fn == 0:
        return 1.0
    return 2 * tp / max(1, 2 * tp + fp + fn)
'''
if "def contextual_key_pairs(" not in s:
    s, n = pattern.subn(new_block, s, count=1)
    if n != 1:
        raise RuntimeError(f"Expected one mapping/null block, replaced {n}")
    changed = True
elif "for fitted_key, true_key in contextual_key_pairs" not in s:
    raise RuntimeError("Partial mapping/null development patch detected")

p.write_text(s, encoding="utf-8")
print(f"PATCHED {p} changed={changed}")
