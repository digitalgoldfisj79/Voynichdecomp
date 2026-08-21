# SVT v0.4.1 — End-to-end hidden segmentation + blind state/key qualification

Status: **FROZEN BEFORE BINDING RUN**.

Parent results:
- v0.3.4 PASS: blind state mode + primitive period + state-dependent key, given true unit boundaries.
- v0.4 S0 PASS: ciphertext-only hidden 1–3 glyph segmentation on eight fresh trials (mean boundary F1 0.9491; minimum 0.9219). The initial red workflow was an aggregate dependency error, not a scientific failure.

## Binding question
Can the complete synthetic mechanism be inverted without being given unit boundaries, state mode, period, or key?

## Data
- German (`de`) dev corpus only.
- 8 fresh trials: `periodic` and `line_reset`, four replicates each.
- Head/plaintext length 1536.
- New replicate namespace: offset **19000**. No overlap with v0.2/v0.3/v0.3.4/v0.4 binding sets.

## Decoder
1. Observe only the unsegmented verbose surface plus surface line starts.
2. Infer 1–3 glyph unit starts with the frozen v0.4 semi-Markov segmenter, unchanged.
3. Extract the predicted head stream from inferred starts; derive head-line indices only from those inferred starts and the observed surface line starts.
4. Screen all 22 `(mode, period)` candidates exactly as in v0.3.3; retain top 6 by ciphertext/language-model score only.
5. For each retained candidate run the frozen 12-start shared-key/state-local optimiser.
6. Select by the frozen penalised score only.
7. Canonicalise periodic superperiods by refitting proper divisors, as in v0.3.4, and select by the same penalised score.
8. Only after final selection reveal truth for evaluation.

No true boundary, mode, period, key, plaintext or recovery score may enter selection.

## Evaluation
Because inferred segmentation may insert/delete units, plaintext recovery is normalized Levenshtein sequence recovery:

`1 - edit_distance(true_plain, decoded_plain) / max(len(true_plain), len(decoded_plain))`.

Also report boundary F1 and unit-count error against hidden truth.

## Binding PASS gate
All conditions must hold:
- canonical `(mode, primitive period)` exact on **8/8** trials;
- mean sequence recovery >= **0.90**;
- median sequence recovery >= **0.90**;
- minimum sequence recovery >= **0.85**;
- mean segmentation boundary F1 >= **0.90**;
- minimum segmentation boundary F1 >= **0.85**;
- mean absolute unit-count relative error <= **0.05**.

No threshold may be changed after the binding run begins. Failure is recorded as failure; any repair requires a separately frozen v0.4.2 protocol and a fresh namespace.

## Target seal
Voynich remains sealed regardless of partial results. Only a full v0.4.1 PASS qualifies the combined synthetic mechanism for a separately preregistered target-transfer stage.
