# SVT v0.4.1 — end-to-end hidden-segmentation gate

Frozen: 2026-08-21, before v0.4.1 binding results.

Parent: SVT v0.3.4 primitive-period PASS + corrected SVT v0.4 S0 segmentation PASS.

Voynich status: **SEALED**.

## Question

Can the qualified components be composed without access to true cipher-unit boundaries?

The decoder receives only:

1. the visible unsegmented 1–3-glyph surface stream;
2. visible line boundaries;
3. the already-fixed German language model used for mechanism calibration.

It does **not** receive true unit boundaries, true head positions, true unit count, true state mode, true period, base key, local state swaps, or plaintext.

## Binding data

Eight fresh synthetic FSVT trials at latent plaintext length 1536:

- 4 periodic;
- 4 line-reset;
- replicate namespace offset **19000**, disjoint from all S0/v0.3.x calibration and binding namespaces.

No result from offset 19000 may be used to alter this protocol.

## Frozen decoder

### Stage 1 — hidden segmentation

Use the qualified v0.4 semi-Markov segmenter unchanged:

- lengths {1,2,3};
- prior (0.30, 0.45, 0.25);
- alpha 0.5;
- 12 EM iterations;
- 6 deterministic restarts;
- ciphertext only;
- line-aware;
- choose maximum fitted segmentation score.

The predicted head stream is the first visible glyph at each predicted unit start.

Predicted head-line starts are derived solely from the visible line starts and the selected segmentation. True head-line indexes are prohibited.

### Stage 2 — blind state/key recovery

On the predicted head stream:

1. screen all 22 structures: mode in {periodic, line_reset}, period 2–12;
2. retain the frozen top-6 screen shortlist;
3. refine each shortlisted structure with the qualified 12-start shared-base + sparse-state key solver;
4. select solely by penalised model score;
5. exactly as in v0.3.4, refit all proper divisors of the selected period under the selected mode and choose the autonomous primitive period by the same penalised score.

No plaintext truth or true schedule information enters screening, refinement, multistart selection, or divisor canonicalisation.

### Stage 3 — post-hoc evaluation only

After the complete model is selected, compare its decoded sequence with the known synthetic plaintext using normalized Levenshtein recovery:

`recovery = 1 - edit_distance(decoded, truth) / max(len(decoded), len(truth))`.

This handles insertion/deletion errors introduced by imperfect segmentation without pretending that surface-unit index equals plaintext index.

## Binding gates

The original SVT joint-development recoverability thresholds remain the mechanism gate, translated from 20 to 8 trials without relaxing proportions:

1. fresh segmentation replication:
   - mean boundary F1 >= 0.90;
   - median boundary F1 >= 0.90;
   - minimum boundary F1 >= 0.85;
   - 8/8 boundary F1 >= 0.85;
   - mean unit-count relative error <= 0.05;
2. end-to-end plaintext:
   - mean normalized edit recovery >= 0.75;
   - median >= 0.85;
   - at least 6/8 trials >= 0.70;
3. blind structure:
   - canonical mode + primitive period exact in at least 6/8 trials (>=0.75);
4. no post-binding parameter change.

PASS requires all three metric families.

A PASS qualifies this composed decoder for a later, separately frozen language-blind locked test. It does **not** authorize Voynich yet.

A FAIL does not reject stateful verbose ciphers. It says that the current two-stage composition does not recover this known mechanism reliably enough. The next admissible amendment would be a joint segmentation/state lattice, versioned separately.

## Leakage / fairness audit

- circularity: truth is unavailable until final metrics;
- leakage: fresh offset-19000 namespace; no boundaries or schedules supplied;
- confounds: same generator family as prior qualification, but all keys/schedules/plaintext chunks are fresh;
- matched controls: component gates already separately calibrated; this gate tests composition, not family discrimination;
- measurement degeneracy: recovery is edit-based because predicted unit count may differ;
- representation dependence: one frozen semi-Markov segmentation output only; no truth-guided alternative segmentation;
- decision fragility: shortlist K=6, 12 starts, BIC weights and divisor rule inherited unchanged;
- audit completeness: every trial records segmentation F1/count error, truth rank post hoc, selected/canonical structure, model score and edit recovery.
