# Wagenseil 1699 Wieduwilt — frozen OCR fidelity sample

## Status

`FROZEN_BEFORE_FULL_TRANSCRIPTION_OR_OCR_DISCREPANCY_VIEWING`

Source object: BSB `bsb10903876`.

Eligible frozen Latin-type image set was fixed beforehand in `WAGENSEIL1699_EXTERNAL_FREEZE.md`: 77 pages comprising image suffixes `00261,00263,...,00393` plus `00395`–`00404`.

## Deterministic selection

Selection string:

`bsb10903876|WAGENSEIL1699_WIEDUWILT|77_LATIN_PAGES|2026-09-12|fidelity_sample_v01`

SHA-256:

`946b91fa23bc86dbebbec3eec03910119a7e68edd7126f28fd4d5fa6231b8c05`

Python RNG integer seed from the first 16 hex digits:

`10694802243648784091`

Six pages selected without replacement and sorted after selection:

| image suffix | printed page |
|---|---:|
| `00273` | 171 |
| `00289` | 187 |
| `00337` | 235 |
| `00381` | 279 |
| `00389` | 287 |
| `00391` | 289 |

The printed-page mapping here uses the mechanically established +102 offset in the alternating p.159–291 run.

## Frozen audit rule

After one fixed full-transcription pipeline has produced all 77 pages, these six pages must be independently compared against the pixels before any L score is admitted.

Record at minimum:

- substantive omitted/inserted words or lines;
- word segmentation disagreements;
- literal character disagreements affecting the later a-z atom stream;
- hyphenation/line-break handling;
- damaged/uncertain print;
- page/order mismatch;
- punctuation/typographic differences separately.

No dictionary, modern-spelling, YIVO or linguistic-plausibility repair is allowed. Any systematic atom-relevant OCR defect blocks use until its effect is explicitly bounded; the six pages cannot be replaced because another page is easier to read.
