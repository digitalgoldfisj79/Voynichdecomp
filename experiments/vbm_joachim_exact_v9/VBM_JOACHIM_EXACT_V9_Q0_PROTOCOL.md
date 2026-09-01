# VBM Joachim-exact v9 — Q0 preregistration

Date: 2026-09-01
Branch: `experiment/vbm-joachim-exact-v9-20260901`
Namespace: `VBMJOACHIMEXACTV9Q0`
Status: **FROZEN BEFORE NEW VMS STRUCTURAL STATISTICS**

## Motivation

The newly published f115v.26 feasibility example specifies a materially different VBM parser from the earlier v1–v8 implementation. Earlier VBM code split token interiors into small surface units and special-cased only `qo` as a two-character left bridge half. The new example instead treats the entire token interior as one consonant-cluster unit and explicitly parses both `qo` and `ch` as two-character left bridge halves.

This experiment therefore does **not** reopen or overwrite earlier VBM closeouts. It tests a newly clarified minimal model.

The supplied worked example is explicitly a feasibility study, not claimed plaintext. It is used only as a parser fixture and capacity illustration; it is not evidence for German or Bavarian.

## Source-grounded parser v9-minimal

For a normalized lowercase alphabetic Voynich token `w`:

1. Left bridge half `L(w)`:
   - `qo` if `w` begins `qo` and has at least one further character;
   - `ch` if `w` begins `ch` and has at least one further character;
   - otherwise the first character.
2. Right bridge half `R(w)` is the final character.
3. Nucleus `N(w)` is the entire substring strictly between `L(w)` and `R(w)`.
4. Empty nuclei are allowed. Thus a two-character token such as `al` contributes no consonant nucleus and may be consumed entirely by neighbouring bridges.
5. A one-character token is a predeclared extension not demonstrated in the supplied example: primary rule `SINGLE_SHARED` assigns the same character to both left and right halves and an empty nucleus. Its frequency is reported. A secondary `SINGLE_EXCLUDE` sensitivity is descriptive only and cannot rescue Q0.
6. For adjacent tokens `w_i w_{i+1}` on the same transcribed line, the vowel-bridge surface type is `R(w_i)|L(w_{i+1})`.
7. Line-initial `L(w_1)` and line-final `R(w_n)` are open LAAFU halves. They are counted but receive no vowel value in Q0.
8. No bridge crosses a line boundary.
9. Tokens containing characters outside `[a-z]` after normalized transcription splitting are excluded and counted; no character repair is allowed.

No additional multi-character bridge halves (for example `sh`) may be invented after Q0 statistics are seen. A richer parser requires a fresh preregistration based on an external rule supplied independently of target outcomes.

## Worked fixture — must parse exactly

Cipher line:

`dcheedy kchedy lcheey ror al chokedy dol qokeeeos qolkeedy qokar ar`

Expected token triples `(L,N,R)`:

1. `dcheedy -> (d, cheed, y)`
2. `kchedy -> (k, ched, y)`
3. `lcheey -> (l, chee, y)`
4. `ror -> (r, o, r)`
5. `al -> (a, EMPTY, l)`
6. `chokedy -> (ch, oked, y)`
7. `dol -> (d, o, l)`
8. `qokeeeos -> (qo, keeeo, s)`
9. `qolkeedy -> (qo, lkeed, y)`
10. `qokar -> (qo, ka, r)`
11. `ar -> (a, EMPTY, r)`

Expected bridges:

`y|k, y|l, y|r, r|a, l|ch, y|d, l|qo, s|qo, y|qo, r|a`

Fixture mismatch is an automatic Q0 failure and stops the programme.

## Minimal codebook class represented by the worked example

If later decoded under this v9-minimal architecture:

- every bridge surface type maps globally to one of five vowels `{a,e,i,o,u}`;
- homophony is allowed: multiple bridge types may share a vowel;
- every non-empty nucleus surface type maps globally to a consonant string of length 1–5;
- the plaintext consonant alphabet for capacity accounting is the 21 lowercase Latin letters excluding `aeiou`;
- empty nucleus maps to empty plaintext;
- the same surface type must retain the same mapping across training and held-out data under the minimal reusable-key hypothesis.

The 1–5 range is frozen because the supplied example explicitly uses nucleus outputs from length 1 through 5. No contextual, folio-specific, hand-specific, or length-specific alternate codebook is admitted in v9-minimal.

## Q0 question

Before attempting any language model or plaintext inference, is the newly specified parser sufficiently reusable and identifiable to support a global train-only codebook on unseen Voynich material?

Q0 is purely structural. It does not fit German, Bavarian, Italian, Latin, or any plaintext.

## Frozen target integrity

Inherited consumed diagnostic set:

`H1 = f28v f31v f88r f5r f34r f81v`

Inherited genuinely sealed confirmation set:

`C1 = f85r1 f53v f33r f10r f23r f111r`

Q0 must not read or score H1 or C1.

All other ZLZI folios form the Q0 pool. They are split deterministically at folio level:

- `INTERNAL_HOLDOUT` if `SHA256("VBMJOACHIMEXACTV9Q0::" + folio)[0:8] mod 5 == 0`;
- otherwise `TRAIN`.

No reassignment is allowed after statistics are seen.

## Q0 measurements

For TRAIN and INTERNAL_HOLDOUT report:

1. folios, lines, valid tokens;
2. one-character token count;
3. invalid/excluded token count;
4. non-empty and empty nucleus events;
5. bridge events and open edge halves;
6. unique nucleus and bridge surface types;
7. singleton and frequency-of-frequency spectra;
8. event-weighted and type-weighted held-out coverage by TRAIN for nuclei and bridges;
9. unseen held-out event counts;
10. top 30 nucleus and bridge types with frequencies;
11. codebook capacity and lower-bound description cost.

### Codebook capacity accounting

Bridge assignment space:

`B_cap = K_bridge * log2(5)` bits.

Each nucleus has

`sum_{l=1..5} 21^l`

possible consonant-string values, so raw mapping-space capacity is:

`N_cap = K_nucleus * log2(sum_{l=1..5} 21^l)` bits.

For later MDL fitting, the frozen prefix-code cost of a chosen nucleus value of length `l` is:

`log2(5) + l*log2(21)` bits.

Q0 also reports an optimistic minimum reusable-key cost by assigning `l=1` to every nucleus. This deliberately favours VBM; type identifiers and parser-description overhead are not charged. If even this optimistic accounting is prohibitive, that is strong negative evidence.

### Worked-example capacity audit

Using exactly the values supplied in the feasibility example, report:

- unique bridge mappings and consistency of repeats;
- unique nucleus mappings and consistency of repeats;
- sum of mapped consonant lengths;
- bridge mapping bits;
- nucleus mapping bits under the frozen prefix code;
- total key bits before any type-name/header overhead;
- bits per produced plaintext character for this one-line fit.

This is explicitly a model-capacity illustration, not a significance test.

## Binding Q0 gate

All must pass:

1. worked fixture exact parse = 100%;
2. INTERNAL_HOLDOUT contains at least 500 non-empty nucleus events and 500 bridge events;
3. TRAIN covers at least **90% of INTERNAL_HOLDOUT nucleus occurrences**;
4. TRAIN covers at least **97% of INTERNAL_HOLDOUT bridge occurrences**.

The occurrence thresholds are frozen before Q0 target statistics. Type coverage itself is diagnostic and has no threshold.

If Q0 fails, stop v9-minimal before language fitting. The permitted conclusion is that the minimal globally reusable whole-nucleus codebook is structurally too unstable for a held-out decipherment test under these rules. A richer context-dependent VBM is not thereby disproven, but it is more flexible and cannot be introduced as a rescue without an independently specified external rule and a new preregistration.

If Q0 passes, Q1 may be designed and preregistered **without reading H1/C1**. Q1 must first qualify a synthetic known-answer decoder/MDL instrument on reusable-key positives and fresh-key/non-language negatives. Only after Q1 qualification may H1 be scored. C1 remains sealed until all H1 gates pass.

## Stop rules

- No language model in Q0.
- No plaintext optimization in Q0.
- No Joachim value assignment is used except for the worked-example capacity audit.
- No new bridge-half grammar after seeing Q0.
- No threshold relaxation.
- No H1 or C1 access.
- Negative results are binding and retained.
