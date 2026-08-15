# Northern Italian Cipher Entropy Compatibility v0.1 — frozen target stage

Date: 2026-08-15
Status: prospective target-comparison protocol written **after** the external-only entropy transfer map froze, but **before** any target entropy values are computed in this stage.

## Parent external freeze
The parent experiment is `Northern Italian Cipher Entropy Transfer v0.1`, GitHub Actions run `31906169329`, artifact digest:

`sha256:b500336c6688822e4f79ffdf53632bf4c35ef7abe809e581053731c0a8a9669d`

It passed all five preregistered external sanity gates on six source families. The external transfer map and source hashes are immutable inputs to the interpretation of this stage.

The parent audit also found an important limitation: the **direction of glyph-level H1 for diglyph mechanisms is rendering-sensitive** under the two frozen rendering diagnostics. Therefore diglyph-bearing mechanisms cannot yield a positive target attribution from H1/MI1. They are retained only for conservative falsification/sensitivity reporting. This limitation is frozen before target scoring.

## Scientific question
Given the externally measured entropy transfer functions of historically plausible Northern Italian cipher ingredients, is the target's observed symbol-sequence entropy compatible with any mechanism family more strongly than with unenciphered external-language controls?

This stage does **not** fit plaintext, cipher key, language, mechanism parameters, or representation to the target.

## Representation repair and integrity gate
The previous hostile specificity experiment failed representation integrity. This stage therefore uses exactly one primary target representation:

**STA_CODE** — every STA code matching `[A-Z][0-9a-z]` is one atomic written symbol; the digit/lowercase suffix is part of the symbol and may never be split by a generic tokenizer.

Excluded from primary inference:
- raw ASCII EVA, because connected forms can be split by transliteration;
- STA-family reduction, because it is a correlated coarse transformation rather than an independent replication;
- connected AAA, because the previous coverage gate failed badly.

Before any entropy score is produced, target integrity must pass:
1. synthetic `P2A3K1` parses as one word of exactly three STA symbols;
2. on every locus shared by RF-EVA and STA, parsed word counts are identical;
3. every retained STA word is fully consumed by repeated `[A-Z][0-9a-z]` codes after IVTFF comments/alternatives are removed; any unparsed alphanumeric residue triggers failure;
4. no representation vote or majority rule is permitted.

If integrity fails, verdict is `REPRESENTATION_INTEGRITY_FAIL` and no mechanism comparison is interpreted.

## Length matching chosen without target
The external parent run showed that the shortest generated output among all primary mechanisms contained **1339 output symbols** for a 3000-plaintext-character window. Therefore this stage fixes an observed-sequence window of **1200 symbols**, chosen below that external minimum with margin.

All external generated outputs and all target sequences are scored on exactly 1200 observed symbols. This removes the variable-output-length confound from absolute entropy comparison.

No target length or entropy value was used to select 1200.

## Target-compatible metrics
Primary vector, computed on each 1200-symbol window:
- `H0`
- `H1`
- `MI1 = H0-H1`
- `H1_norm = H1/log2(K)`

Secondary only:
- `H2`
- `K`

Expansion and information-rate diagnostics cannot be observed for the target without knowing plaintext length, so they are **not** used for target compatibility.

## External calibration before target access
The workflow first reconstructs the frozen v0.1 generators and exact external source families, verifies parent input hashes / pinned CREMMA commit, and regenerates each primary mechanism with the frozen seeds and parameters.

Each generated sequence is truncated to its first 1200 observed symbols. No target file exists in the workspace during this stage.

Mechanisms are grouped prospectively:
- `ID`: identity/plaintext control
- `HOM`: HOM2, HOM34, HOM34_NULL_1, HOM34_NULL_2p5, HOM34_NULL_5
- `NOM`: NOM25, NOM50, NOM100
- `BIGRAM`: BIGRAM10, BIGRAM20, BIGRAM40
- `DIGLYPH`: BIGRAM20_DIGLYPH25/50/75
- `COMBINED`: COMBINED_LOW/MID/HIGH

`ORTHO_POST1460` is secondary historical sensitivity only and cannot support a pre-1450 attribution.

## Distance model
For each individual mechanism, the four primary entropy metrics are standardised using robust external statistics only:
- centre = median across generated external samples;
- scale = `1.4826 * MAD`; if zero, use external standard deviation; if still zero, metric is unusable for that mechanism.

Distance is RMS robust-z distance across usable metrics.

### Leave-one-source-family-out open-set calibration
For each mechanism and each source family `f`:
- fit centre/scale on the other five source families;
- score the held-out same-mechanism samples from `f`;
- record that family's median held-out distance.

The mechanism's compatibility threshold is the **maximum** of its six held-out-family median distances. This is conservative and fully external.

A mechanism is externally qualified only if:
1. at least four primary metrics have nonzero usable scale;
2. all six source families contribute qualifying 1200-symbol samples;
3. its held-out threshold is finite.

## Multiple-mechanism selection and plaintext null
Selecting the best of several cipher variants can create a false improvement. Therefore the exact same group-minimisation used on the target is calibrated on held-out **identity/plaintext** windows.

For each held-out plaintext source family:
- fit all mechanism models without that family;
- score the held-out identity windows;
- compute `Delta_group = d_ID - min(d_variant in group)` for HOM, NOM, BIGRAM, DIGLYPH and COMBINED.

For each group, the external plaintext-null threshold is the **95th percentile** of these pooled held-out plaintext `Delta_group` values. If sample count is too small for a stable percentile (<30), use the maximum instead.

A target group may be called `CIPHER_ENTROPY_ADVANTAGE` only if its target median `Delta_group` exceeds that externally frozen plaintext-null threshold.

## Positive-control qualification
Before target access, each cipher group must also demonstrate that externally generated members of that group can be distinguished from ID under the same group-min procedure.

For held-out generated cipher samples, define the same `Delta_group`. A group qualifies only when:
- median `Delta_group > 0`, and
- at least 5/6 source-family medians are positive.

A group failing this positive-control gate is `NOT_TESTABLE` on the target.

## Diglyph limitation
Because parent v0.1 found glyph-level H1 rendering sensitivity:
- `DIGLYPH` and `COMBINED` cannot receive a positive historical-attribution verdict from the primary four-metric distance;
- they may be **rejected** if target incompatibility is robust under both frozen renderings;
- if compatible under either rendering they are reported `RENDERING_SENSITIVE_NONRESOLVING`.

No post-target rendering model may be introduced.

## Target scoring
Only after all external gates and thresholds are written to `external_freeze.json` does the workflow download pinned RF-EVA and STA target files and run the representation-integrity gate.

Primary target dataset:
- concatenate STA_CODE symbols in manuscript order;
- form non-overlapping 1200-symbol windows;
- no symbols cross an IVTFF line-order discontinuity introduced by comments/alternatives, but ordinary word/line boundaries are omitted from the entropy stream.

Secondary section dataset:
- concatenate within each frozen section label only;
- form non-overlapping 1200-symbol windows;
- sections with fewer than two windows are descriptive only.

For each qualified non-diglyph group:
- calculate target-window distances to each frozen individual mechanism;
- group distance is the minimum mechanism distance per window;
- calculate `Delta_group = d_ID - d_group` per target window;
- compare the target median against the frozen external plaintext-null threshold.

## Verdicts
For the primary whole-manuscript target:

- `NO_CIPHER_ENTROPY_ADVANTAGE`: no qualified HOM/NOM/BIGRAM group exceeds its external plaintext-null threshold.
- `HISTORICAL_CIPHER_ENTROPY_CANDIDATE`: at least one qualified HOM/NOM/BIGRAM group exceeds its threshold **and** the target median distance to at least one member mechanism is within that mechanism's external LOSO compatibility threshold.
- `ENTROPY_ADVANTAGE_BUT_OUT_OF_DISTRIBUTION`: a group beats the plaintext-null threshold but target remains outside all member mechanism compatibility thresholds.
- `REPRESENTATION_INTEGRITY_FAIL`: integrity gate fails.
- `EXTERNAL_CALIBRATION_FAIL`: any mandatory external calibration gate fails.

Section results cannot upgrade a negative whole-manuscript verdict; they are heterogeneity diagnostics only.

## Firewall / circularity rules
- No target entropy value is present in this protocol or external calibration.
- No source language is selected after target exposure.
- No mechanism parameter changes after target exposure.
- No representation majority vote.
- No Naibbe calibration.
- No f116v or imagery evidence enters the score.
- No lexical matches enter the score.
- The target can falsify the historical mechanism family; it cannot retrospectively redefine it.
