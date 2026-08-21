# SVT v0.4.2 — Joint semi-Markov segmentation/state qualification

Status: **FROZEN before binding namespace 23000 is opened**.

## Motivation

v0.3.4 qualified blind state mode + primitive period + state-dependent key when unit boundaries are known. v0.4 S0 qualified ciphertext-only 1–3 glyph segmentation (mean boundary F1 about 0.95). v0.4.1 nevertheless failed end-to-end: small boundary insertion/deletion errors changed the inferred unit index and therefore desynchronised the state clock.

v0.4.2 changes the architecture, not the gate. Segmentation and state phase are inferred jointly.

## Decoder information

The decoder receives only:

- the unsegmented 1–3 glyph ciphertext surface;
- observed line starts;
- the candidate German language model already used in the qualified synthetic programme;
- the frozen mechanism class from v0.1–v0.3.4.

It does **not** receive true boundaries, plaintext, state mode, period, phase, key, or continuation offsets.

## Joint model

For each candidate `(mode, period)` with mode in `{periodic, line_reset}` and period 2–12:

1. initialise a ciphertext-only semi-Markov surface model from v0.4 S0;
2. infer a state-dependent substitution key from the current head path;
3. run a joint semi-Markov Viterbi/beam search in which every proposed 1–3 glyph unit:
   - chooses a boundary;
   - emits its first glyph as the cipher head;
   - decodes that head under the current state-dependent inverse key;
   - receives plaintext trigram/unigram likelihood;
   - receives the frozen v0.4 surface-unit likelihood;
   - advances the hidden state clock by exactly one unit;
   - resets state phase only at observed line starts for `line_reset` mode;
4. refit the factorised key on the newly inferred head sequence;
5. alternate for a fixed number of iterations.

No truth-dependent path repair or phase snapping is permitted.

The surface likelihood weight is frozen at **0.35**, inherited from the original v0.1 joint scoring constant. Candidate complexity uses the already frozen v0.2 BIC penalties.

## Search

- all 22 `(mode, period)` structures receive a cheap joint pass;
- top 6 by truth-free penalised joint score receive the full pass;
- full pass uses 12 independent key starts and 3 joint alternations;
- after final selection, proper divisors of the selected period are re-fit under the same joint model and the highest penalised score defines the primitive canonical period.

## Development namespace

The already-open v0.4.1 offset-19000 trials may be used only for implementation checks and nonbinding diagnostics. They cannot contribute to the v0.4.2 binding verdict.

## Binding data

Eight entirely fresh German synthetic trials:

- head/plaintext length: 1536;
- modes: 4 periodic + 4 line-reset;
- replicate namespace: `23000 + replicate`, replicate 0–3 per mode.

Truth is read only after final model selection for evaluation.

## Binding gate

PASS requires **all** of:

1. 8/8 exact canonical `(mode, primitive period)` recovery;
2. mean normalized Levenshtein plaintext sequence recovery >= 0.90;
3. median sequence recovery >= 0.90;
4. minimum sequence recovery >= 0.85;
5. mean boundary F1 >= 0.90;
6. minimum boundary F1 >= 0.85;
7. mean absolute unit-count relative error <= 0.05.

No criterion may be weakened after binding execution begins.

## Target seal

Voynich remains sealed. A v0.4.2 PASS qualifies this mechanism class for a separately frozen target-transfer protocol; it does not itself identify the Voynich mechanism or plaintext language.
