# Yiddish cipher control v1

This directory is the first implementation of the repaired Voynich cipher-instrument programme.

## Scientific boundary

It validates an **instrument**, not a Voynich hypothesis. The target manuscript is deliberately inaccessible to the benchmark entry point. A run may qualify only the declared operating envelope of the M0 solver: globally fixed one-to-one substitution, preserved boundaries/order, historical-Yiddish control material, registered lengths/damage/search budget.

## Evidence contract

A result is scientifically interpretable only through the certificate states in `summary.json`:

- `C0` provenance/freeze
- `C1` representation verification
- `C2` known-answer implementation tests
- `C3` blinded planted-positive recovery/power
- `C4a` structural matched negatives / false-positive calibration
- `C4b` German language-specificity diagnostic (not an M0-mechanism failure gate; separate language certificate remains required)
- `C5` metamorphic verification
- `C6` independent historical/source transfer — separate, not run here
- `C7` sealed Voynich target — prohibited here

Allowed terminal semantics are `PASS`, `FAIL`, `UNRESOLVED`, `UNBOUNDED`, `OUT_OF_DOMAIN`, plus explicit non-qualifying smoke states. No other score may be promoted to a scientific verdict.

## Profiles

`smoke` verifies that acquisition, imports, KATs, planted controls, negative controls, comparator parsing and evidence writing execute. It is **never qualifying**.

`full` is the expensive calibration surface. Before a full run, apply `STATISTICAL_BOUND_ADDENDUM_V01.md`: family-wise false-positive bounds require enough negative trials (minimum 94 at Bonferroni alpha 0.05/6 if there are zero false calls) and stochastic metamorphic relations require >=32 planted instances.

## Current data roles

The executable algorithm controls use already-consumed Penn historical-Yiddish material. That is intentional: no fresh historical source is spent validating code. The Penn representation is explicitly secondary/normalized and cannot issue a primary-script transfer certificate.

Official ReF 1.0.2 diplomatic German material is acquired independently for the comparator diagnostic. German recovery by the M0 solver demonstrates that M0 key recovery is not itself language identification; it does not make the solver wrong. Yiddish identity still requires a separately qualified L instrument.

Basel 1599 is consumed DEVELOPMENT for the image-level source-transfer programme after v0.1 failed its frozen page-geometry audit. It cannot be reused as fresh confirmation. The 1544 Augsburg *Shmuel-bukh* remains reserved for a separately preregistered transfer stage and is not touched by this control.

## Canonical persistence

Supabase handoff: `voynich_cipher_instrument_recovery_standard_20260912_v01`.

GitHub branch: `gpt56/yiddish-control-v1-20260912`.

Future cipher experiments should retrieve the Supabase handoff first and must not infer mechanism absence from a solver failure unless the relevant instrument family has passed this qualification architecture.