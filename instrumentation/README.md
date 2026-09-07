# Voynich Instrument Qualification Harness v1

This directory is a gating layer for Voynich experiments. It is deliberately separate from any scientific model. Its job is to prevent target inference unless an assay has first demonstrated that it can recognise its own null/source and reject the prespecified scientifically relevant alternatives.

## Lifecycle

`DEV -> FROZEN -> CALIBRATED -> VALIDATED -> ADVERSARIAL_POWERED -> TARGET_OPENED -> SENSITIVITY -> CLOSED`

Any failed gate becomes `BLOCKED`; target data remain sealed. A new threshold/metric/model after a failed gate requires a new assay version and fresh calibration/validation/control splits.

## Mandatory principles

1. **Adequacy != identification.** Accepting the true source is necessary but not sufficient. Relevant alternatives must also be rejected at the preregistered operating point.
2. **No target tuning.** The target must be inaccessible to fitting, thresholding, representation selection and model selection.
3. **Null SD is explicit.** `effect_over_null_sd = abs(effect)/null_sd`. Sampling/physical-unit uncertainty and model-refit uncertainty are separate quantities.
4. **<2 null SD means unresolved.** The qualification output emits the mandatory headline automatically.
5. **Power is source-specific.** Alternatives are named in the frozen manifest. Failure to distinguish one alternative does not license a claim about another.
6. **Representation robustness is a gate when required.** It is never a post-hoc rescue.
7. **Licensed inference is machine-readable.** A passing qualification returns the exact bounded statement that may be used; otherwise it returns `NO TARGET INFERENCE LICENSED`.

## Files

- `assay_manifest.schema.json` — common manifest contract.
- `assay_guard.py` — freeze, qualify and target-seal checks; standard library only.
- `examples/t0b_demo_*` — reproduces why T0b should remain target-sealed: known-source false rejection is acceptable, but alternative power is inadequate.
- `tests/test_assay_guard.py` — guardrail tests.

## Usage

```bash
python instrumentation/assay_guard.py freeze manifest.json freeze.json
python instrumentation/assay_guard.py qualify manifest.json freeze.json summary.json qualification.json
python instrumentation/assay_guard.py check-target manifest.json qualification.json
```

A scientific target runner must call `check-target` (or the same Python function) before loading target observations.

## Required summary fields

The qualification summary contains a null/known-source validation block and one block for every named alternative. For each alternative, provide the number rejected, total trials, raw effect (`observed - null mean` or the assay's frozen contrast), and the matched null SD. Leakage assertions are mandatory. If representation robustness is required in the manifest, all registered representation checks must pass.

## What this harness does not do

It does not choose the statistic, model, representation, null, controls, mechanism strengths or power grid. Those are scientific decisions that must be frozen in the assay manifest. The harness merely makes their gates explicit and hard to bypass accidentally.
