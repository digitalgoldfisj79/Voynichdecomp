# Morpholocal calibration v0.2 — reproduction, audit and closure

Date: 2026-07-15

## Final status

**Formal verdict: `FAIL_MORPHOLOCAL_CLASS_CALIBRATION`.**

The bounded morpho-local cipher class is not authorised for Voynich-manuscript application under the frozen v0.2 protocol. The decoder is highly conservative against the tested production controls but insufficiently sensitive to the planted positive class.

## Formal run

- Hugging Face job: `6a5725bf85d9643ce16d378d`
- Hardware: `cpu-xl`, 24 trial workers
- Scientific commit at launch: `06b124608e8c1a1253aea99a861ff3a7ebc66abc`
- Seed: `8675309`
- Positives: 96
- Controls: 320
- Annealing steps: 400
- Restarts: 2
- Runtime: 500 seconds
- Result SHA-256: `c12c48d5585dd4efc5935d29ca2eae46df3c1dabd6475ed89ae6eb7a3c0b1705`

### Formal result

- Positive successes: 34/96
- Positive Wilson 90% interval: `[0.2788822371513577, 0.4374457519344646]`
- False positives: 0/320
- Control Wilson 90% interval: `[0.0, 0.00838393857488932]`
- Median mapping accuracy: `0.5833333333333334`
- Median null F1: `1.0`
- Selector recovery: `1.0`
- Structure recovery: `0.5729166666666666`

Passed gates:

- overall false-positive upper bound <= 0.05;
- every control-family upper bound <= 0.10;
- median null F1 >= 0.75;
- selector recovery >= 0.80;
- numerical accelerated/scalar validation <= 1e-7 bits.

Failed gates:

- overall positive lower bound >= 0.70;
- every positive-stratum lower bound >= 0.50;
- median mapping accuracy >= 0.60;
- structure recovery >= 0.65.

The decisive policy failures were frequency-weighted selection at 1/24 recoveries and sticky-line-reset selection at 2/24. IID-uniform recovered 18/24 and cyclic recovered 13/24.

## Exact clean-clone reproduction

- Hugging Face job: `6a5728c385d9643ce16d37b9`
- Clean-clone commit: `bf0a900c0d81028a9b6a0c7a2801cce9c4efe408`
- Hardware: `cpu-xl`, 24 trial workers
- Runtime: 540 seconds
- Reproduction result SHA-256: `c12c48d5585dd4efc5935d29ca2eae46df3c1dabd6475ed89ae6eb7a3c0b1705`
- First audit JSON SHA-256: `c1dc1e44cf06c326c97de4f9367d3d320000b2e022c349d24566b0e7566dc7b1`

The reproduction is byte-for-byte identical to the formal result. It reproduces all 416 per-trial records, all strata, every criterion and the final failure verdict.

## Independent result recomputation

`formal_audit.py` independently recomputed overall counts, Wilson intervals, mapping accuracy, null F1, selector recovery and structure recovery. It returned `PASS` for consistency with the result artifact.

A subsequent hostile audit identified two limitations in that first audit:

1. it did not independently recompute the declared all-positive-strata gate;
2. it hard-coded the expected failure verdict rather than deriving the verdict generically.

These are audit-design weaknesses, not changes to the formal result.

## Corrected hostile audit

- Fresh-environment job: `6a572cdfb1669a49bf0738fb`
- Audit commit: `d038ca68998effe8af269ff7cf5547a37f1c5a1d`
- Audit output SHA-256: `d6730407b02127e3f2f64789cff707ffdfb18bff1652e5ea479663e125e6ceb3`
- Status: `PASS_WITH_FINDINGS`

The corrected audit independently verified:

- exact result hash;
- all 416 unique trial indices;
- the complete deterministic positive and control seed schedules;
- all six positive-stratification dimensions and their Wilson intervals;
- all four control-family counts and intervals;
- all eight preregistered criteria;
- the dynamically derived failure verdict;
- reconstruction and hashes of the frozen source bundle.

It found no scientific-result inconsistency.

### Hostile provenance findings

**High — incomplete effective-source freeze.** `FREEZE_RECORD.json` hashes the unpatched base source bundle but does not hash `apply_development_patch.py`, `gpu_runner.py`, or `cpu_batched_runner.py`, although those files define the effective formal implementation. Exact clean-clone reproduction substantially mitigates this, but it cannot retroactively make the preregistered freeze complete.

Post-hoc effective-source hashes are recorded in `POSTHOC_EFFECTIVE_SOURCE_HASHES.json`. They are explicitly a provenance supplement, not a retroactive repair.

**Medium — first audit omitted the positive-stratum gate.** Corrected by the hostile audit, which confirmed that the gate fails.

**Medium — first audit hard-coded the expected failure outcome.** Corrected by dynamically deriving the verdict from every gate.

**Low — misleading artifact label.** The formal JSON retains the field name `development_accelerator`, although the implementation had been frozen for the formal run.

**Low — procedural manuscript interlock.** The prohibition on manuscript analysis is documented rather than cryptographically enforced. No executable manuscript-application entry point exists in this calibration directory, and no manuscript inference was run.

## Independence qualification

The audits are independent implementations executed in separate clean environments, but they were produced within the same AI-assisted research workflow. They are **not an external human third-party audit**. No claim of external peer review is made.

## Scientific interpretation

This result establishes a narrow but useful negative:

> The frozen v0.2 decoder cannot reliably recover the full planted bounded morpho-local nomenclator class, despite excellent rejection of the matched production controls.

It does not show that historical nomenclator mechanisms are impossible, nor that the Voynich manuscript is not enciphered. It shows that this decoder and calibration architecture are not admissible for manuscript inference.

The literature review indicates that any future v0.3 should treat segmentation/composition, homophone-selection policy and key inference explicitly, and benchmark multiple decoder families. That would be a new preregistered programme rather than a rescue of v0.2.

## Closure

- v0.2 development run: failed.
- v0.2 formal run: failed.
- clean-clone reproduction: exact.
- corrected hostile audit: scientific result validated, provenance findings recorded.
- Voynich manuscript run: not performed and not authorised.
- v0.2 programme: closed.
