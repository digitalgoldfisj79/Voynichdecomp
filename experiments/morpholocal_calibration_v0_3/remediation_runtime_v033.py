#!/usr/bin/env python3
"""v0.3.3 diagnostic runtime: selector parity plus latent-order randomization.

The v0.3.2 score audit showed that recoverable partitions and held-out
cipher-vs-production likelihood do not reject an unseen permuted-cipher
control.  This runtime adds a conditional test of the missing property:
message-like order in the held-out latent sequence.

After fitting the mapping and all model choices on training data, the observed
held-out transition likelihood is compared with deterministic within-line
randomizations.  Each randomization keeps the first latent unit fixed and
permutes the remaining units within that line.  Thus it preserves:

* the fitted mapping and model;
* document and line membership;
* line lengths;
* each line's exact latent-unit multiset;
* the line-start stationary contribution.

Only latent transition order changes.  Lower observed transition codelength
than the randomizations supports ordered latent structure.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np

import remediation_runtime as v031
import remediation_runtime_v032 as v032

base = v032.base


def _latent_lines(module: Any, events: Sequence[Any], assignments, scheme: str) -> list[list[int]]:
    lines: list[list[int]] = []
    current: list[int] = []
    previous = None
    for event in events:
        marker = (event.doc, event.line)
        if previous is not None and marker != previous:
            if current:
                lines.append(current)
            current = []
        current.append(int(base.mapping_unit(module, event, assignments, scheme)))
        previous = marker
    if current:
        lines.append(current)
    return lines


def _transition_bits(lines: Sequence[Sequence[int]], log_transition: np.ndarray) -> tuple[float, int]:
    bits = 0.0
    transitions = 0
    for line in lines:
        if len(line) < 2:
            continue
        left = np.asarray(line[:-1], dtype=np.int64)
        right = np.asarray(line[1:], dtype=np.int64)
        bits -= float(log_transition[left, right].sum())
        transitions += len(line) - 1
    return float(bits), int(transitions)


def sequence_randomization_audit(
    module: Any,
    test: Sequence[Any],
    fitted: dict[str, Any],
    transition: np.ndarray,
    seed: int,
    randomizations: int,
) -> dict[str, Any]:
    """One-sided conditional randomization test for latent transition order."""
    lines = _latent_lines(module, test, fitted["assignments"], fitted["scheme"])
    log_transition = np.log2(np.clip(transition, 1e-300, None))
    observed, transitions = _transition_bits(lines, log_transition)
    randomizations = max(1, int(randomizations))
    rng = random.Random(int(seed) ^ 0x5E017A11)
    null_bits: list[float] = []
    effective_lines = sum(len(line) >= 3 and len(set(line[1:])) > 1 for line in lines)

    for _ in range(randomizations):
        permuted: list[list[int]] = []
        for line in lines:
            if len(line) <= 2:
                permuted.append(list(line))
                continue
            tail = list(line[1:])
            rng.shuffle(tail)
            permuted.append([int(line[0]), *tail])
        value, _ = _transition_bits(permuted, log_transition)
        null_bits.append(float(value))

    null_array = np.asarray(null_bits, dtype=float)
    null_mean = float(null_array.mean())
    null_sd = float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0
    # A randomization is at least as favourable to the external sequence model
    # when it has transition codelength no greater than the observation.
    favourable = int(np.count_nonzero(null_array <= observed + 1e-12))
    p_value = float((favourable + 1) / (len(null_array) + 1))
    advantage = float(null_mean - observed)
    z_value = float(advantage / null_sd) if null_sd > 0 else (math.inf if advantage > 0 else 0.0)

    return {
        "mode": "within_line_keep_first",
        "randomizations": int(randomizations),
        "test_lines": int(len(lines)),
        "effective_lines": int(effective_lines),
        "transitions": int(transitions),
        "observed_transition_bits": float(observed),
        "null_mean_transition_bits": null_mean,
        "null_sd_transition_bits": null_sd,
        "advantage_bits": advantage,
        "advantage_bits_per_transition": float(advantage / max(1, transitions)),
        "z": z_value,
        "p_value": p_value,
        "pass_0_05": bool(p_value <= 0.05 and advantage > 0.0),
    }


def score_trial_v033(module, gr, events, registry, external, seed, solver_name, cfg, truth):
    """Exact v0.3.2 accounting plus a held-out latent-order diagnostic."""
    train = [event for event in events if not event.test]
    test = [event for event in events if event.test]
    fitted = base.fit_candidate(module, gr, train, registry, external, seed, solver_name, cfg)
    transition, stationary = module.extend_external_model(
        external, fitted["external_profile"], fitted["null_count"]
    )
    external_train, external_test = base.external_nll_by_split(
        module, events, fitted["assignments"], fitted["scheme"], transition, stationary
    )
    policy_train, policy_test = base.policy_nll(
        module, events, fitted["assignments"], fitted["scheme"], fitted["policy"], registry,
        return_by_split=True,
    )
    selector_train = module.selector_nll(train, registry, fitted["selector"])
    selector_test = module.selector_nll(test, registry, fitted["selector"])
    cipher_train = external_train + policy_train + selector_train
    cipher_test = external_test + policy_test + selector_test

    production_selector, _ = module.select_selector(train, registry)
    production_test = module.production_predictive_nll(test, train, registry, production_selector)
    production_train = module.production_predictive_nll(train, train, registry, production_selector)
    c_full = module.cipher_model_record(
        registry, fitted["assignments"], fitted["scheme"], fitted["sizes"],
        fitted["null_count"], fitted["size_profile"], fitted["external_profile"],
        fitted["selector"], train, full=True,
    )
    c_cond = module.cipher_model_record(
        registry, fitted["assignments"], fitted["scheme"], fitted["sizes"],
        fitted["null_count"], fitted["size_profile"], fitted["external_profile"],
        fitted["selector"], train, full=False,
    )
    p_full = module.production_model_record(registry, production_selector, train, full=True)
    p_cond = module.production_model_record(registry, production_selector, train, full=False)
    cfr, ccr = module.cost_model(c_full), module.cost_model(c_cond)
    pfr, pcr = module.cost_model(p_full), module.cost_model(p_cond)
    policy_index_bits = math.log2(len(base.POLICIES))
    h_full_c = cfr.canonical_serialization_bits + policy_index_bits + cipher_train + cipher_test
    h_full_p = pfr.canonical_serialization_bits + production_train + production_test
    h_cond_c = ccr.canonical_serialization_bits + policy_index_bits + cipher_train + cipher_test
    h_cond_p = pcr.canonical_serialization_bits + production_train + production_test
    extra_partition = ccr.partition_bits * max(0, len(fitted["assignments"]) - 1)
    i_c = ccr.structural_universal_bits + extra_partition + policy_index_bits + cipher_train + cipher_test
    i_p = pcr.structural_universal_bits + production_train + production_test
    differences = {
        "H_full_cipher_minus_production": h_full_c - h_full_p,
        "H_conditional_cipher_minus_production": h_cond_c - h_cond_p,
        "I_full_cipher_minus_production": i_c - i_p,
        "I_conditional_cipher_minus_production": i_c - i_p,
        "heldout_predictive_cipher_minus_production": cipher_test - production_test,
    }
    signs = all(
        differences[key] < 0
        for key in (
            "H_full_cipher_minus_production",
            "H_conditional_cipher_minus_production",
            "I_full_cipher_minus_production",
            "I_conditional_cipher_minus_production",
        )
    )
    gain = (production_test - cipher_test) / max(1, len(test))
    cipher_selected = signs and gain >= -0.025
    sequence = sequence_randomization_audit(
        module,
        test,
        fitted,
        transition,
        int(seed),
        int(cfg.get("sequence_randomizations", 199)),
    )
    v031._ACTIVE_AUDIT.update(
        {
            "sequence_randomization_mode": sequence["mode"],
            "sequence_randomizations": sequence["randomizations"],
            "sequence_randomization_transitions": sequence["transitions"],
            "sequence_randomization_effective_lines": sequence["effective_lines"],
            "sequence_randomization_p": sequence["p_value"],
            "sequence_randomization_z": sequence["z"],
            "sequence_randomization_advantage_bits_per_transition": sequence[
                "advantage_bits_per_transition"
            ],
            "sequence_randomization_pass_0_05": sequence["pass_0_05"],
        }
    )

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "solver": solver_name,
        "fitted": {
            "scheme": fitted["scheme"],
            "null_count": fitted["null_count"],
            "size_profile": fitted["size_profile"],
            "external_profile": fitted["external_profile"],
            "selector": fitted["selector"],
            "selection_policy": fitted["policy"],
            "class_sizes": list(fitted["sizes"]),
        },
        "production_selector": production_selector,
        "differences_bits": differences,
        "predictive_gain_bits_per_test_token": gain,
        "cipher_selected": bool(cipher_selected),
        "sequence_randomization": sequence,
        "solver_meta": fitted["solver_meta"],
        "policy_scores": fitted["policy_scores"],
    }
    if truth is not None:
        accuracy = module.mapping_accuracy(
            fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"]
        )
        nf1 = module.null_f1(
            fitted["assignments"], truth["keys"], fitted["scheme"], truth["key_scheme"]
        )
        selector_correct = fitted["selector"] == truth["selector"]
        policy_correct = fitted["policy"] == truth["selection_policy"]
        structure_correct = (
            fitted["scheme"] == truth["key_scheme"]
            and fitted["null_count"] == truth["null_count"]
            and fitted["size_profile"] == truth["size_profile"]
            and fitted["external_profile"] == truth["external_profile"]
        )
        test_errors = sum(
            base.mapping_unit(module, event, fitted["assignments"], fitted["scheme"])
            != int(event.true_unit)
            for event in test
        )
        latent_error = test_errors / max(1, len(test))
        threshold = 0.55 if truth["key_scheme"] == "currier" else 0.65
        positive_success = (
            cipher_selected
            and accuracy >= threshold
            and nf1 >= 0.50
            and policy_correct
            and latent_error <= 0.35
        )
        result.update(
            {
                "truth": {key: value for key, value in truth.items() if key != "keys"},
                "mapping_accuracy": accuracy,
                "null_f1": nf1,
                "selector_correct": selector_correct,
                "policy_correct": policy_correct,
                "structure_correct": structure_correct,
                "latent_unit_error": latent_error,
                "positive_success": bool(positive_success),
            }
        )
    return result


def install() -> None:
    """Install v0.3.2 fixes while substituting the v0.3.3 trial scorer."""
    v031._ORIGINAL_SCORE_TRIAL = score_trial_v033
    v032.install()


if __name__ == "__main__":
    install()
    base.main()
