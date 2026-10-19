# Q1B registered after completed Q1 discovery: emission/dynamics separation

2026-10-19. **New experiment, not a retroactive reinterpretation of Q1.** Parent Q1 first two seeds were used as discovery and may not count for any confirmation.

### Motivation
The blind Q1 HMM could recover signal in sequence likelihood without consistently recovering the true emitting SELECT controller category (e.g. weak-language NMI .456; notation .587; table IID NMI .648; grammar-only NMI .040). A single HMM appears able to re-purpose latent states for sequence structure. Separate the tasks prospectively.

### New Q1B instrument
1. Use same frozen 57-piece FORM/12-class, strict legality/STOP/8-controller forward model and same abstract source-family generators as Q1, *unchanged*. All synthetic only, no Voynich inversion.
2. Generate independent 1200/400/400 TRAIN/VAL/TEST sequences per planted instance; 32 source items; same 8 latent controller groups. Do not use controller identity or source identity during any optimisation or model selection.
3. Fit **IID emission mixture** alone for 8 SELECT categories using the same exact frozen FORM emission family (START and token-constant shared ROUTE bias). Fix the mixture emission parameters using VAL exact marginal likelihood. Frozen `l2` {.3, 1.0}; restarts 3; no truth-selected choice. Test yields blind controller-class NMI and same-category pair AUC.
4. **Freeze** the selected emission parameters and separately fit only an 8×8 HMM transition matrix using unlabeled TRAIN observations. HMM selected on VAL exact marginal likelihood against the frozen emission-only IID model. Transition prior smoothing .5 or 3.0, fixed 8 EM iterations, model selection through VALIDATION only.
5. Evaluate untouched TEST by trained source-category posterior and incremental test marginal log likelihood (HMM minus IID, bits/token). Even if HMM loses model selection, report the comparison; no selection by truth. Pairwise source **item identity** AUC is separately exploratory, and is *not* inferred from controller-category AUC.
6. Grammar-only negative and IID table-repertoire negative must retain near-zero positive sequence gain; nuisance capacity unchanged for matched comparisons.

### Gates and interpretation
- As original Q1: blind controller-category recovery median NMI ≥.70, 10th-percentile NMI ≥.60 and at least 90% of independently seeded instances NMI ≥.60; pair AUC ≥.80. Sequence claims: ≥.02 test bits/token in 90% of true positive families, false sequence detection ≤5% of 20+ negatives, requiring matched high-structure negatives.
- Pilot 2 seeds/family only for code/discovery; then *fresh* 20-seed panel at 32 types if design stable. Do not silently tune after confirmation. One-label-per-SELECT state is a **proxy** until calibrated, not proof of semantics.
- Crucial: this test freezes the **emission interface** but does NOT yet install CONNECT and LINE_ENTRY into the full forward generator, so it cannot license real Voynich claims. Full renderer-native synthetic calibration remains obligatory.
- If still fails, do not spend on broad K9–K20 brute force; document instrument bottleneck and consider identifying lower-dimensional state equivalences or relationships instead of surface words.


## Confirmation addendum (before new 20-seed run, after 2-seed discovery)
Pilot Q1B at 32 latent types: NMI medians LANG ~.530, NOTATION ~.625, TABLE ~.502, NONE ~.036. Same-controller-pair AUC showed a materially stronger signal than per-token NMI: LANG ~.854, NOTATION ~.883, TABLE ~.806, NONE ~.500. These pilot outcomes may not be counted towards confirmation and show the existing ≥.70 NMI gate is unlikely to pass.
**Additional prespecified diagnostic:** source-item pair AUC *conditioned on the same known planted SELECT class* at TEST time. The true class label is opened solely after encoder/model selection; within-class source-item equality AUC should be ~.50 if the FORM renderer loses lexical identity within its compressed SELECT channel. This is not an instrument input. It must not be used to select architectures.
Fresh confirmation seeds: 20269000–20269019 for 32-type source cases, four source families. No changes to source generators, candidate bank, thresholds, or model-selection procedure from pilot. Confirmed NMI below threshold remains failure of this instrument; a good pair AUC alone may license an alternative **equivalence-class** recovery claim only after independent category-specific calibration and matched structural negatives. Matched repertoire and source-neutral negatives still required for dynamic information gates.
