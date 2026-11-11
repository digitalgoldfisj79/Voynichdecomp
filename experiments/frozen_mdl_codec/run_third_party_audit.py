#!/usr/bin/env python3
"""Run a frozen independent-model audit. Requires vLLM and one GPU."""
from __future__ import annotations

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    args = parser.parse_args()

    files = [
        "THIRD_PARTY_AUDIT_PROMPT.md",
        "SPEC.md",
        "codec.py",
        "test_codec.py",
        "fuzz_codec.py",
        "synthetic_gate.py",
        "CONFORMANCE_VECTORS.json",
        "registry_fixture.json",
        "fixtures/model_minimal.json",
        "fixtures/model_stateful.json",
    ]
    sections = []
    for name in files:
        sections.append(f"\n===== FILE: {name} =====\n" + (args.source / name).read_text(errors="replace"))
    for label, path in [
        ("UNIT TEST TRANSCRIPT", args.source / "AUDIT_UNIT.log"),
        ("CONFORMANCE TRANSCRIPT", args.source / "AUDIT_CONFORMANCE.log"),
        ("FUZZ TRANSCRIPT", args.source / "AUDIT_FUZZ.log"),
        ("FULL SYNTHETIC RESULT", args.result),
    ]:
        sections.append(f"\n===== {label} =====\n" + path.read_text(errors="replace"))

    system = (
        "You are an independent critical auditor of software, information-theoretic accounting, "
        "and experimental design. You did not author the supplied implementation. Find concrete "
        "defects rather than agreeing by default. Separate executable bugs, mathematical errors, "
        "asymmetric accounting, defensible convention choices, and limitations of the synthetic "
        "experiment. Do not discuss whether the Voynich manuscript is a cipher. Every material "
        "objection must include a minimal counterexample or explicit calculation. The final line "
        "must be exactly one of: PASS_REPRODUCED, FAIL_IMPLEMENTATION, "
        "FAIL_ASYMMETRIC_ACCOUNTING, UNRESOLVED_CONVENTION, UNRESOLVED_SYNTHETIC_GATE."
    )
    model = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=98304,
        gpu_memory_utilization=0.94,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.15, top_p=0.9, max_tokens=7000, seed=20260714)
    response = model.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(sections)}],
        params,
        use_tqdm=True,
    )[0].outputs[0].text
    args.output.write_text(response)
    print(response)


if __name__ == "__main__":
    main()
