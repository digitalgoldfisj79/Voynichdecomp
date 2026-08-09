# v1.1 Addendum — EVA connected-unit falsification

Frozen 2026-08-09 after identifying that connected EVA sequences contribute essentially the entire induced within-word German-vs-French transition advantage, and before fitting any atomic-unit M19 key.

## Motivation

The v1.0 analysis treated each literal lowercase character in the slim transliteration as an independent cipher symbol. EVA documentation states that `ch`, `sh`, `cfh`, `ckh`, `cph`, and `cth` are special connected sequences for which connectivity is implicit. On C10 ZLZI, greedily collapsing these sequences reduces 103,954 literal characters to 93,485 connected/basic units and expands the surface-unit inventory from 25 literal characters to 31 units.

A cipher interpretation must therefore survive the alternative representation in which these connected sequences are atomic surface units. This is a representation-sensitivity/falsification test, not a claim that EVA's connected sequences must be single cryptographic symbols historically.

## Tokenization

Within each Voynich word, greedily match longest-first:

`cfh`, `ckh`, `cph`, `cth`, `ch`, `sh`

All other literal characters remain singleton units. Word boundaries are unchanged. No other compounds are introduced.

## Atomic M19 model

Keep the exact 19 BnF unmarked numerical values and the same five-table plaintext emission law. The observed surface alphabet now has 31 units. For the conservative homophone parameterization, every numerical value must have at least one surface unit and exactly 12 values have a second surface homophone (31 = 19 + 12); no value has >2 forms.

## Positive-control gate

Generate 31-form synthetic M19 ciphertexts for the same six v0.9 qualification languages: Latin, Italian, German, French, Arabic, Spanish. Use the v0.9 fresh LM partitions. Training/held-out lengths remain 45,000/39,000 plaintext letters.

Fit with a generalized legal-map annealer (31 surface units, 19 values, 12 duplicated values), two independent fits per true language. Required before Voynich scoring:

- 6/6 correct language rank;
- minimum true-language held-out margin >= 0.05 nats/letter;
- median weighted numerical-map recovery >= 0.95;
- minimum recovery >= 0.85;
- minimum independent-fit agreement >= 0.90.

If this fails, the atomic implementation is `INSTRUMENT NOT QUALIFIED` and cannot falsify v1.0.

## Voynich test

Reuse exactly the v0.9 folio partition logic to obtain T09 (key-fitting folios), H09 (held-out discovery folios), and C10 (all remaining 122 folios). Only tokenization changes.

Fit a separate atomic M19 key under each of the eight language models on T09. Rank on H09 by exact forward likelihood. Require independent-fit agreement >=0.90 and language margin >=0.05 for a primary atomic signal.

Regardless of H09 outcome, report the best stable map's fixed C10 ranking as an explanatory sensitivity diagnostic, clearly labelled post hoc relative to the already-observed C10 literal-character result.

## Interpretation

- If German remains stable and first after atomic tokenization, the v1.0 signal is not explained by EVA connected-sequence decomposition.
- If German loses stability/rank or the margin collapses, classify the v1.0 German effect as `EVA REPRESENTATION SENSITIVE`.
- If no language gives a stable atomic map, do not search additional tokenizations in this programme; stop with representation sensitivity established.
