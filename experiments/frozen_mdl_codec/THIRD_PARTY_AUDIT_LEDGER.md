# Third-party model audit ledger

The audit target was source archive SHA-256:

`7dd12696bb6b6e550be325a373cf5c44d60b0ea2429b8e8e9fd5090a6f73999e`

The full synthetic result had SHA-256:

`98e28540e86deae8c517eaa6250a8d9af7431ca939329198ab90a2e9f1598917`

Neither audit had access to the current ChatGPT conversation. Both were supplied the frozen source, specification, tests, regenerated test transcripts and full synthetic result.

## Audit 1: Qwen3 Coder 30B A3B

- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
- Job: `6a56897bb1669a49bf072b87`
- Seed: `20260714`
- Final verdict: `PASS_REPRODUCED`

Quality register: **low**. The audit initially labelled `log2_choose` defective, immediately demonstrated that its formula and tested values were correct, repeatedly failed to produce a counterexample, and finally withdrew the objection. Its final verdict is retained, but its narrative supplies no valid demolition beyond reproduction.

The raw output is preserved verbatim in `THIRD_PARTY_AUDIT_QWEN3_CODER.md`.

## Audit 2: Devstral Small 2507

- Model: `mistralai/Devstral-Small-2507`
- Job: `6a568ad4b1669a49bf072c04`
- Seed: `20260714`
- Temperature: `0.05`
- Final verdict: `PASS_REPRODUCED`

The second prompt required an explicit expected-versus-actual calculation or executable counterexample before any defect could be asserted. The model returned only the verdict and no findings. This is a clean reproduction verdict but not a substantive mathematical review.

The raw output is preserved in `THIRD_PARTY_AUDIT_DEVSTRAL.md`.

## Evidential register

The codec and equal-class selection-policy gate have now been:

1. built and frozen before any Voynich use;
2. reproduced in a clean CPU environment;
3. checked by exact conformance vectors and 300 deterministic fuzz cases;
4. tested across four held-out selection-policy families;
5. submitted to two separately trained coding models, both ending `PASS_REPRODUCED`.

This supports the register **INTERNALLY VALIDATED / THIRD-PARTY-LLM REPRODUCED**.

It does not support human-reviewed status, and it does not validate the future mixed-unit or morpho-local nomenclator. The present synthetic gate is limited to eight equal plaintext classes with three homophones each and no nulls, mixed units, multiple keys, house/gallows constraints or adjacent-length production selector.
