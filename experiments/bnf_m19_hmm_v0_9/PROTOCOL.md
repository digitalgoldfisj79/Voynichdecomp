# BnF 7342 free-switch M19 + forward-HMM programme v0.9 — preregistration

Date: 2026-08-09
Status at freeze: no v0.9 qualification-control or Voynich score observed.

## Rationale

v0.8 correctly identified fresh Latin, Italian, German and French controls by held-out exact forward likelihood, but failed the frozen optimizer-reproducibility gate because one French pairwise mapping fit landed in a local optimum: independent-fit agreement 0.89082 < 0.90. No Voynich score was generated.

Control-only optimizer development then showed the same French ciphertext is recovered at 100% mapping accuracy and 100% agreement with a stronger search budget of 24,000 annealing steps × 6 restarts. v0.9 therefore changes only optimizer convergence and uses a new untouched qualification partition.

## Cipher and language model

The M19 cipher model, 19 exact unmarked BnF values, 25-surface-form multiplicity constraints, BnF emission law, exact word-level hidden-letter forward likelihood, language panel, normalization, Viterbi decoder and transfer rules are unchanged from v0.8.

## Strong mapping optimizer

Every surface→number fit uses:
- 24,000 simulated-annealing steps per restart;
- 6 independent restarts;
- the unchanged deterministic legal-neighbor polish from v0.7.

No result-dependent restart extension is allowed.

Two completely independent fit seed namespaces are still required for target-control and Voynich reproducibility checks.

## Fresh sentence partition

For every UD corpus, sentence indices are split by `i mod 10`:

- **LM training:** residues `{3,4,8,9}` (40%);
- **v0.9 qualification pool:** residues `{2,7}` (20%);
- residues `{0,5}` are excluded because used in HMM development;
- residues `{1,6}` are excluded because used for v0.8 qualification.

Therefore no v0.9 qualification sentence appeared in an earlier M19-HMM development/qualification control or in its own LM training set.

Qualification languages remain the six full-repertoire languages:
Latin, Italian, German, French, Arabic, Spanish/Castilian.

Greek and Hebrew remain Voynich candidates but are not full-repertoire qualification languages under the frozen romanization; this limitation remains explicit.

## Fresh positive-control gate

One new 84,000-letter M19 control per qualification language:
- first 45,000 letters for fit;
- next 39,000 for held-out ranking.

Plaintext span-selection namespace: `20260809|v09qual|language`.
Control value/surface randomization uses the v0.8 generator law on this entirely new plaintext material; all 25 forms must occur in training.

For each control, fit separate maps under all eight languages with the strong optimizer and rank by held-out exact forward nats/letter. Refit the true target language independently.

PASS requires all:
- Q1: correct language ranks first 6/6;
- Q2: minimum correct-language forward margin >=0.05 nats/letter;
- Q3: median true weighted numerical mapping accuracy >=0.95;
- Q4: minimum mapping accuracy >=0.85;
- Q5: minimum independent-fit mapping agreement >=0.90.

Any failure stops before Voynich.

## Voynich stage

If and only if the gate passes, use a new whole-folio split namespace `20260809|M19HMMv09|folio`.

- 20% held-out folios;
- training folios selected in hash order until >=45,000 glyph positions and all 25 lowercased ZLZI surface labels present in the non-hold partition are observed;
- held-out mapped-symbol coverage >=99%.

For every language, fit two strong M19 maps independently on training. Choose the map with the higher frozen pairwise training objective, record their occurrence-weighted agreement, and rank languages solely by held-out exact forward likelihood.

Primary candidate criteria are unchanged:
- top-v-second forward margin >=0.05 nats/letter;
- top-language independent-fit agreement >=0.90;
- valid exact M19 surjection/multiplicity;
- held-out coverage >=99%.

A primary candidate triggers unchanged Viterbi lexical enrichment (z>=5), then literal-map transfer to TTLI and VDRB. Each transfer must have candidate language rank1, forward margin>=0.03 nats/letter, lexical z>=3 and shared-glyph coverage>=90%.

Only all-stage success is `CONFIRMED M19-HMM SIGNAL`.

## Scope

A positive result would establish statistical compatibility with this exact free-switch unmarked-number mechanism, not prove historical use of BnF lat.7342. A qualified negative rejects only the exact global M19 model with preserved spaces and the frozen language/normalization panel.