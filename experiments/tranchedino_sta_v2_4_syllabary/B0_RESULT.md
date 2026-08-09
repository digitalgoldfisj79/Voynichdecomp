# Tranchedino × STA v2.4 — Stage B0 result

Date: 2026-08-09
Protocol: `PROTOCOL_B0_FEASIBILITY.md`
Status: **STAGE B0 PASS — B1 POSITIVE-CONTROL DESIGN AUTHORISED; NO VOYNICH FIT AUTHORISED**
Scientific payload SHA-256: `348c50a46dd53e1a297bf395850272192048456312d7da99d1f3ed03719a9923`

## 1. Exact source reproduction

The frozen historical Paduan source was reproduced under the same 19-letter normalisation as the preceding Tranchedino × STA stages:

- chronological cut page: 183;
- training: 4,119 lines / 172,347 retained letters;
- held out: 1,423 lines / 54,750 retained letters.

Four deterministic held-out 12,000-letter windows selected by the frozen SHA-256 rule begin on pages:

`189, 244, 240, 235`.

No Voynich data entered this stage.

## 2. Forty-eight-control nuisance grid

The frozen grid was fully executed:

- `p_syll ∈ {0.25, 0.50, 0.75, 1.00}`;
- `p_null ∈ {0.00, 0.03, 0.10}`;
- four deterministic held-out windows per cell;
- 48 controls total.

No cell was dropped.

### Aggregate ranges

| quantity | minimum | median | maximum |
|---|---:|---:|---:|
| active surface inventory K | **112** | **122** | **124** |
| plaintext letters / cipher event | 0.99461 | 1.21562 | 1.56433 |
| cipher events per 12,000 letters | 7,671 | 9,871.5 | 12,065 |
| surface entropy, bits | 5.65255 | 6.22998 | 6.57700 |

All 43 historical alphabetic homophone signs appeared in every control.

Syllabary exposure was strong:

- distinct syllable signs observed: minimum **59/64**, median **64/64**, maximum **64/64**;
- syllable occurrences: minimum **875**, median **2,442**, maximum **4,096**.

Geminate exposure:

- **7/8** geminate signs occurred in every control;
- `bb` was absent in all four deterministic windows;
- `cc, ff, mm, nn, rr, tt, ss` were present.

Null exposure behaved as expected:

- zero null identities when `p_null=0`;
- all seven null signs observed when `p_null>0`.

The nine securely transcribed lexical entries were sparse, as expected from the actual source rather than a synthetic frequent-word substitute:

- distinct secure lexical identities observed: 2–3, median 3;
- secure lexical-code occurrences: 5–30, median 23;
- only `che`, `como`, and `quando` occurred somewhere in the four selected windows.

The remaining 35 historical lexical/nomenclator slots remained latent and were not assigned invented plaintext words.

## 3. Stratum summary

| p_syll | p_null | K active range | median expansion | minimum distinct syllables | median syllable occurrences |
|---:|---:|---:|---:|---:|---:|
| .25 | .00 | 112–116 | 1.10467 | 59 | 951.5 |
| .25 | .03 | 121–124 | 1.07072 | 61 | — |
| .25 | .10 | 120–123 | 1.00214 | 60 | — |
| .50 | .00 | 114–117 | 1.21495 | 61 | 1,929 |
| .50 | .03 | 122–124 | 1.17388 | 62 | — |
| .50 | .10 | 121–124 | 1.10759 | 61 | — |
| .75 | .00 | 115–117 | 1.34800 | 62 | — |
| .75 | .03 | 122–124 | 1.30920 | 62 | — |
| .75 | .10 | 122–124 | 1.21661 | 62 | — |
| 1.00 | .00 | 115–117 | 1.51087 | 62 | — |
| 1.00 | .03 | 122–124 | 1.46675 | 62 | — |
| 1.00 | .10 | 122–124 | 1.37444 | 62 | — |

Dashes indicate values not needed for the binding gate; complete row-level results are deterministically reproducible with committed `b0_feasibility.py`.

## 4. RF/full-STA representation gate

A fresh bounded Hugging Face census was run after first confirming no jobs were active.

HF job: `6a78a1afda2af92a634f04a8`

The job reproduced the binding RF1b SHA-256 and the 157,254-character / 166-member census, then evaluated exact top-K occurrence coverage over the complete observed B0 range:

| K | RF occurrence coverage |
|---:|---:|
| 112 | 0.9996566065 |
| 113 | 0.9996629656 |
| 114 | 0.9996693248 |
| 115 | 0.9996756839 |
| 116 | 0.9996820431 |
| 117 | 0.9996884022 |
| 118 | 0.9996947613 |
| 119 | 0.9997011205 |
| 120 | 0.9997074796 |
| 121 | 0.9997138388 |
| 122 | 0.9997201979 |
| 123 | 0.9997265570 |
| 124 | 0.9997329162 |

Therefore all 48 controls exceed the frozen `>=0.995` full-STA coverage threshold by a wide margin.

Formal full-STA result: **48/48 PASS**.

## 5. Connected-aaa diagnostic

All B0 active inventories satisfy `K_active <=124 <150`, so pure distinct-sign cardinality does not rule out the 150-sign connected-aaa signature inventory.

This is **not** an aaa mapping qualification. The official STA→aaa conversion may expand one full-STA member into multiple aaa units, and raw aaa does not carry the originating-member boundary. A connected-aaa target arm therefore remains prohibited without its own segmentation protocol.

## 6. Interpretation

The primary-source f.134v–135r mechanism survives the pre-solver test:

1. its finite active inventory at Paduan control length is not implausibly larger than the RF/full-STA symbolic inventory;
2. essentially all RF occurrences can be represented at the required K;
3. the 64-entry historical syllabary is sufficiently occupied to make positive-control identifiability testable rather than vacuous;
4. the sparse diplomatic/nomenclator tail remains a nuisance, not a reason to invent frequent lexical substitutes.

The exact descriptive equality between the historical 19-letter-compatible key-sheet capacity K=166 and RF's 166 observed full-STA member types remains **non-evidential** and is not used in any gate.

## Verdict

**STAGE B0 PASS.**

This authorises only B1 synthetic/Paduan positive-control calibration for the one-visible-sign / variable-plaintext-expansion historical mechanism. It does not authorise a Voynich language score, target map, or decoded plaintext.
