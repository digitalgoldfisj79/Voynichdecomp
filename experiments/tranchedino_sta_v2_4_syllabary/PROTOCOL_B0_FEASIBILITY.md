# Tranchedino × STA v2.4 — Stage B0 source/representation feasibility

Date frozen: 2026-08-09
Namespace: `TRANCHSTA24B0`
Status: **SOURCE-ONLY FEASIBILITY; NO SOLVER AND NO VOYNICH FIT AUTHORISED**
Historical template: `HISTORICAL_TEMPLATE_F134V_F135R.{md,json}`

## 1. Question

Before building another blind solver, determine whether the primary-source f.134v–135r one-sign syllabary mechanism produces a finite active surface inventory that is representable in the already frozen RF/full-STA symbolic layer at Paduan control length.

This stage may inspect only the historical key sheet and genuine Paduan source. It must not score Voynich language likelihood, fit a key to Voynich, or inspect any decoded Voynich string.

## 2. Fixed source

Reuse the recovered historical Paduan line source from v2.3:

- archive SHA-256 `ddae949a2d4ff13714204f3751feaf9e836333ef57a45def77c803cd87fc7b61`;
- `paduan_lines.csv` SHA-256 `c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6`;
- chronological training/held-out cut at page 183;
- frozen 19-letter normalisation `abcdefghilmnopqrstu`, with `j→i`, `v/w→u`, `y→i`, `x/z→s`.

Use held-out source only for the finite-cipher occupancy simulations. Training frequencies may be used only to report source frequencies, not to select a target representation.

## 3. Fixed K166 semantic slot inventory

The 19-letter-compatible f.134v–135r model contains 166 historical slots:

- 43 alphabetic homophone signs: 3 each for `a e i o u`, 2 for each other retained letter;
- 8 geminate signs: `bb cc ff mm nn rr tt ss`;
- 64 one-sign syllabic entries from the frozen historical template;
- 7 null signs;
- 44 lexical/nomenclator signs.

Only nine lexical strings are presently palaeographically secure enough for automatic source matching:
`dinari galee nave grippo che perche como unde quando`.

The other 35 lexical/nomenclator slots remain real historical key capacity but are **latent in B0**. They may not be assigned substitute modern/frequent words merely to increase occupancy.

## 4. Frozen source generator

Each control is a 12,000-normalised-letter window assembled in line order from held-out Paduan pages. Four deterministic windows are selected by SHA-256 ordering of `TRANCHSTA24B0window::<page>` and cyclic concatenation as necessary. Word and line boundaries are available to the generator; word spaces are not emitted to the cipher surface.

Encoding order within each word:

1. if the whole normalised word is one of the nine secure lexical entries, emit its one lexical sign;
2. otherwise scan left-to-right;
3. if the next plaintext span matches a frozen syllabary unit, use the longest match (`qua/que/qui/quo` before any shorter alternative) and emit its syllable sign with probability `p_syll`; if the Bernoulli draw fails, continue to steps 4–5 without consuming the span;
4. if the next two characters form one of the eight frozen geminates, emit its geminate sign and consume two characters;
5. otherwise emit one of the historical alphabetic homophones uniformly and consume one character;
6. after every substantive emitted sign, insert at most one null with probability `p_null`, choosing uniformly among the seven null signs.

Every historical slot has a stable semantic identity inside a control, then all 166 surface labels are independently randomly permuted. Label values therefore contain no class information.

## 5. Nuisance grid

Historical usage frequencies for syllabic and null signs are not stated by the key sheet. They are therefore treated as calibration nuisance strata rather than fitted parameters.

Freeze the 12-cell grid:

- `p_syll ∈ {0.25, 0.50, 0.75, 1.00}`;
- `p_null ∈ {0.00, 0.03, 0.10}`.

Run the four deterministic held-out windows in every cell: 48 source-only controls total.

No cell may be dropped after results are seen.

## 6. Measurements

For every control archive:

- cipher-event count;
- plaintext-letter / cipher-event expansion ratio;
- number of distinct active surface signs overall and by class;
- occurrence counts by class;
- distinct syllable signs observed / 64;
- distinct alphabetic signs observed / 43;
- distinct geminate signs observed / 8;
- distinct null signs observed / 7 where `p_null>0`;
- secure lexical-code occurrences and identities;
- empirical surface entropy.

No recovery score exists in B0 because no solver exists in B0.

## 7. Full-STA representation census

Binding RF1b source remains SHA-256
`81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`, with 157,254 parsed full-STA characters and 166 observed member types under the frozen parser.

For each observed historical active-inventory size `K_active`, compute occurrence coverage of the `K_active` most frequent RF full-STA member types.

Primary full-STA feasibility gate:

- every one of the 48 controls must have `K_active <= 166` (structural sanity);
- every control's RF top-`K_active` occurrence coverage must be `>=0.995`.

This is only an inventory/coverage test. It does not assign any historical semantic slot to an RF member.

## 8. Connected-aaa audit

The prior official `STA-aaa.bit` conversion audit found 150 distinct usable aaa signatures from 161 usable full-STA members; top-92 full-STA members collapsed to 88 signatures.

B0 records whether each control has `K_active <=150`, but this is **diagnostic only**. Raw connected aaa does not preserve the boundary of an originating full-STA member when one member expands to multiple aaa units. Therefore no one-sign historical mapping to connected aaa is authorised in v2.4 B0.

A later aaa arm would require its own frozen segmentation/representation protocol.

## 9. Advancement

B0 passes for the primary full-STA arm only if all 48 controls satisfy the full-STA feasibility gate.

If B0 passes, the next permitted step is **B1 positive-control solver design** for the one-sign / variable-plaintext-expansion mechanism. B1 must prove fresh-key recovery and component recovery on synthetic Paduan controls before any Voynich target can be scored.

If B0 fails, no solver is built for this template.

Regardless of result, the exact K166 equality between the historical compatible key sheet and RF's 166 observed full-STA types remains descriptive and is not itself evidence for a historical relationship.
