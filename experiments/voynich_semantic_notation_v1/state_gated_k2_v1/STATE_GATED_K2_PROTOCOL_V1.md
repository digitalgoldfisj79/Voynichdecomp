# VSN-B3-v1 — State-Gated Matteo K2 Protocol

Frozen: 2026-08-12 (Europe/London), before any state-gated target scoring.
Parent programme: VSN-v1 Workstream B. This is not a third programme.

## Question

Can a historically motivated **context-gating layer**, added without altering Matteo da Verona's frozen two-first-syllable (`K=2`) composition operation, reproduce the hierarchical Voynich properties that literal ungated K2 failed?

The hierarchy to explain was established before this protocol:

1. aggregate K2-like edit-location topology;
2. substantially greater one-edit density within every Voynich section than matched-size ungated K2;
3. section-specific prefix/internal/suffix regimes;
4. one-edit-neighbour concentration within ordinary running-text lines after within-section token shuffling;
5. stronger Voynich local transition constraint and right-edge positional asymmetry.

## Historical/source boundary

Source-derived:
- Matteo provides an artificial-word operation based on first syllables;
- Matteo separately provides typed/state-like encodings involving classes/dimensions/degrees;
- related Bartolomeo/Ragona systems provide context/field-restricted encoding precedents.

Not source-derived:
- the exact four-state hashes below;
- assigning a source syllable to a synthetic domain/class/degree state;
- mapping one synthetic state to one Voynich section or line;
- any claim that a Voynich prefix/suffix has a particular medieval meaning.

The experiment therefore tests a **mechanism class**: `K2 composition + context-dependent component availability`. It does not decode Voynich.

## Frozen lexical source and K2 operation

Same source and syllabifier as VSN-B2:
- PyWORDS `pywords/data/lingualatina_voclist.txt`;
- expected raw SHA-256 `5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c`;
- 1,846 eligible normalized source words in the prior frozen run;
- first-syllable operation unchanged;
- sanity examples: `tripode -> tri`, `pepo -> pe`, `corvus -> cor`, `vetula -> ve`.

No syllabifier or lexical item is modified using Voynich outcomes.

## Frozen target representation

Primary representation is RF Basic EVA exact-letter types, using the corrected **distinct unordered one-edit pair** definition.

The target bundle is committed as `state_gated_targets_v1.json` before scoring. It records:
- section type counts;
- section one-edit pair counts and prefix/internal/suffix proportions;
- type-weighted mean length;
- type-weighted `H(next|previous)`;
- type-weighted `H(char|right position)-H(char|left position)`;
- running-text (`layout_family=P`) line-length histograms;
- running-text one-edit pair enrichment.

### Important correction carried forward

`rf_edit1_pairs` has 28,435 edit-path rows but 27,307 distinct unordered token pairs. Hierarchical work uses distinct pairs.

A new SQL extraction of type-weighted section entropies was used for this bundle because an earlier exploratory section-entropy query did not exactly reproduce the frozen whole-corpus entropy implementation. The new extraction reproduces the frozen whole-corpus `H(char|left)` and `H(char|right)` values exactly. The failed intermediate SQL attempt (`column reference n is ambiguous`) is an execution-log item, not a scientific result.

## Discovery / holdout firewall

Frozen before model scoring:

**Discovery sections**
- Stars
- Herbal-A
- Balneological
- text-only

**Primary held-out sections**
- Pharmaceutical
- Herbal-B

**Diagnostics only**
- Zodiac: no ordinary running-text `P` line set in the current materialisation;
- Cosmological: only four `P` lines / 35 tokens, inadequate for a line-gating test;
- `(missing)`: no stable section label.

No holdout section may determine architecture choice, gate widths, seed selection, scoring weights or thresholds.

## Synthetic state assignment

Namespace: `VSN_STATE_GATED_K2_V1`.

Every unique first syllable is assigned deterministic synthetic categorical attributes by SHA-256:
- `domain in {0,1,2,3}`;
- `class in {0,1,2,3}`;
- `degree in {0,1,2,3}`;
- independent line attributes for slot 1 and slot 2.

Every section receives deterministic target categories from its literal section name and slot. These assignments are not fitted.

A width `w` means that `w` consecutive classes modulo four are available. Width 4 is ungated.

## Frozen model families

### BASE
Ungated frozen Matteo K2. Included as the reference model.

### DOMAIN
Both K2 positions are restricted by deterministic section-specific domain categories. Slot 1 and slot 2 may have different globally fixed widths.

### STATE
A two-dimensional typed-state abstraction:
- slot 1 availability is restricted by the four-class attribute;
- slot 2 availability is restricted by the four-degree attribute.

This is an abstraction of class × degree style state coding, not a claim about Matteo's literal surface syntax.

### LINE
Uses the STATE section vocabulary and adds a line-persistent latent state. Each synthetic running-text line receives one deterministic state; tokens on that line are sampled only from section-vocabulary types whose two component line attributes are compatible with that line state.

The line state changes only at a line boundary.

## Frozen parameter grid

For DOMAIN and STATE, slot-width pairs are exactly:

`(1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,3), (4,4)`.

For LINE, every pair above is crossed with line width `1` or `2`.

BASE is `(4,4)` without line gating.

Total frozen configurations: 33.

No additional width, asymmetric special case, combined Bartolomeo/Ragona rule, or Voynich-derived component mapping may be added in v1.

## Section-vocabulary generation

For every section/configuration/seed:
- generate exactly the section's observed unique-type count;
- draw eligible source lemmas uniformly from the frozen lexical source;
- concatenate their first syllables;
- discard duplicate generated surfaces;
- retain one source-syllable provenance pair for each unique synthetic type;
- fail the configuration if the requested inventory cannot be reached in 5,000,000 attempts.

Thus section vocabulary size is nuisance-matched; edit topology is not.

## Synthetic running-text lines

For each section, reproduce the exact observed histogram of running-text line lengths.

- BASE/DOMAIN/STATE: sample with replacement from the full generated section vocabulary.
- LINE: sample with replacement only from the current line-state-compatible subset.
- no Voynich token frequencies are copied;
- no line is selected or excluded because of its observed edit content.

Line enrichment is computed against the synthetic corpus's own frequency-weighted random-pair baseline, matching the corrected SQL definition.

## Metrics

Per section:
1. distinct one-edit pair count and pair-count ratio to target;
2. prefix/internal/suffix edit-location distribution and total-variation distance;
3. mean type length;
4. `H(next character | previous character)`;
5. `H(character | right position) - H(character | left position)`;
6. running-text within-line one-edit enrichment.

No plaintext, decoded text or semantic output is produced.

## Discovery scoring

Four deterministic discovery seeds:
`2026081201` through `2026081204`.

For each section/run, normalized errors use frozen tolerances:
- pair density: `|log(pair_ratio)| / log(1.25)`;
- edit-location TV: `/ 0.08`;
- line-enrichment difference: `/ 0.25`;
- `H(next|prev)` difference: `/ 0.35` bits;
- right-minus-left difference: `/ 0.10` bits;
- mean-length difference: `/ 0.75` characters.

Section loss = mean squared normalized error over these six metrics.
Configuration discovery loss = mean over all discovery sections and seeds.

Winner = minimum discovery loss.
Tie rule: configurations within 2% of the minimum resolve to the **simpler family** (`BASE < DOMAIN < STATE < LINE`), then to the less restrictive/larger-width configuration.

## Holdout unlock rule

Primary holdout is opened only if the selected gated model:
1. improves mean discovery loss by at least 20% over BASE; and
2. improves section-specific mean loss over BASE in at least 3 of 4 discovery sections; and
3. is not BASE itself.

Otherwise VSN-B3-v1 stops at discovery.

## Held-out execution

If unlocked, the winning configuration is frozen verbatim and evaluated on Pharmaceutical and Herbal-B with 20 new deterministic seeds:
`2026081301` through `2026081320`.

No configuration reselection occurs.

A held-out section median passes only if all are true:
- pair ratio in `[0.80, 1.25]`;
- edit-location TV `<= 0.08`;
- line-enrichment difference `<= 0.25`;
- `H(next|prev)` difference `<= 0.35` bits;
- right-minus-left is negative and differs by `<= 0.10` bits;
- mean type length differs by `<= 0.75` characters.

Overall STRUCTURAL-GATING PASS requires both held-out section medians to pass all six criteria.

Seed-level pass counts and full raw rows are reported regardless of verdict.

## Interpretation

Possible outcomes:

- **PASS**: a source-independent state-gating mechanism class can generically reproduce the previously missing hierarchical properties; this warrants deeper historical comparison, not decipherment.
- **PARTIAL**: gating repairs some dimensions (e.g. section density or line clustering) but not the full held-out panel.
- **FAIL**: context gating of this frozen form does not rescue Matteo K2; do not invent additional gates in v1.

## Transparency / compute

Before execution the repository must contain:
- this protocol;
- exact executable `state_gated_k2_v1.py`;
- exact target bundle `state_gated_targets_v1.json`.

Execution must use immutable commit URLs. Raw JSON output, job IDs, source hashes, failed runs and final no-running-HF-job status must be archived.
