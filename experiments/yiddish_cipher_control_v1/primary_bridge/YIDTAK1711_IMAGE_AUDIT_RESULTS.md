# YIDTAK1711 six-leaf primary-script audit — results

## RETRACTED FINDINGS

1. **RETRACTED:** “the dedicated HUC probe never ran.” It did run earlier and failed on the obsolete `mss.huc.edu` endpoint. That was a transport failure, not evidence that the source was unavailable.
2. **RETRACTED:** the first `3r/p009` audit count of **13 segmentation errors / 60 total discrepancies**. Cause: raw OOXML extraction dropped Word soft-line-break elements and concatenated words across lines. The comparator was re-extracted with `python-docx`, preserving line breaks. Those segmentation counts must not be used.
3. **RETRACTED:** the broad framing “the DOCX systematically adds rafe everywhere.” `6v/p016` is a counterexample: independently checked representative rafe loci match the source there. The supported claim is page-/locus-dependent editorial or transcriptional divergence, not universal rafe insertion.
4. **RETRACTED / INSTRUMENT FAILURE:** GOT-OCR 2.0 produced digit-like garbage on these pages and is not evidential. Florence-2 OCR returned only `1` on `7r`. Neither output is used.
5. **RETRACTED / NOT USED:** an exhaustive GPT-4o pass on `7r` generated internally contradictory and obviously spurious discrepancies (including claiming text was omitted when it was present in the supplied comparator). Only the earlier focused header adjudication is retained.

## Frozen design

- Frozen pre-view sample: `1v`, `3r`, `3v`, `6v`, `7r`, `8r`.
- Mechanical source mapping committed before viewing:
  - `1v` -> `0006_p006.jpg`
  - `3r` -> `0009_p009.jpg`
  - `3v` -> `0010_p010.jpg`
  - `6v` -> `0016_p016.jpg`
  - `7r` -> `0017_p017.jpg`
  - `8r` -> `0019_p019.jpg`
- Comparator: exact Zenodo file `Takanot 1711 - YidTakNL.docx`, SHA-256 `b949f2c089072a757fedeaacca5108f9c120981bd0f92b1479d73950531f0d46`.
- Word line breaks were preserved in the corrected comparator extraction.
- Frozen failure rule: a material omission, page mismatch, or systematic transcription/normalization defect relevant to source-script atomization blocks promotion to a primary-script bridge. Small literal disagreements are reported and bounded, not silently repaired.

## Conservative page-level adjudication

The table records only discrepancies retained after focused independent visual checks. It deliberately does **not** claim exhaustive character-error counts, because general-purpose vision models were unreliable when asked for exhaustive OCR.

| Side | Source image | Page/order | Whole-word/phrase omission or insertion | Independently retained atom-relevant discrepancies | Result |
|---|---|---|---|---|---|
| `1v` | `0006_p006.jpg` | match | none detected | comparator `מן` vs source `אן`; repeated comparator `התֿמנותֿ` vs source `התאנות`; comparator `וממטיבֿ` vs source `ומאטיב`; multiple comparator rafe marks absent in source | **FAIL** |
| `3r` | `0009_p009.jpg` | match | none detected | repeated comparator `התֿמנותֿ` vs source `התמנות`; `בביתֿ הכנסתֿ` vs source without rafes; `בפתֿיחתֿ` vs source without rafes; comparator `אנ׳` vs source `אונ׳` | **FAIL** |
| `3v` | `0010_p010.jpg` | match | none detected | comparator `מין` vs source `קיין`; comparator `אורטר` vs source `אורדר`; multiple comparator rafe marks absent in source at `פֿאר/פֿאלגניש`, `גלייכֿן`, `בביתֿ הכנסתֿ`, `תפֿילה`, `פֿון`, `זאכֿין`, `פֿאר ריכֿטן` | **FAIL** |
| `6v` | `0016_p016.jpg` | match | none detected | comparator `בלאשטין` vs source `בלאשטן`; representative rafe loci independently checked and found to match | **FAIL literal equality** |
| `7r` | `0017_p017.jpg` | match | none established by the focused adjudication | comparator header `חקנות הקהילה` vs source `תקנות הקהילה`; focused independent check did not establish a systematic rafe mismatch on this page | **FAIL literal equality** |
| `8r` | `0019_p019.jpg` | match | none detected | comparator-added rafes independently confirmed at several loci including `גיחתֿמנת`, `חיובֿים`, `שבתֿ`, `מתנתֿ`, `חופתֿו`, `רבֿ`, `רבֿיעי`, `רופֿין`; comparator `צווייטא` vs source `צווייטה`; comparator `רעגירנדי` vs source `רעגירנדר` | **FAIL** |

## Bounding statement

All **6 of 6 precommitted leaves** contain at least one independently retained literal source/transcription disagreement. This is a deterministic audit result, not a stochastic effect-size comparison, so there is no meaningful null standard deviation to report. The result is bounded conservatively: it establishes failure of exact/bijective source-script inheritance on the frozen sample, but it does **not** estimate the manuscript-wide character error rate and does **not** imply the YidTakNL edition is generally poor.

The strongest falsification of a “universal rafe-normalization” explanation is `6v`, where representative rafe loci match. Therefore the supported interpretation is **heterogeneous editorial/transcription divergence**, including both rafe normalization on several audited leaves and isolated base-letter disagreements.

## Bridge decision

**PRIMARY-SCRIPT BRIDGE: FAIL for literal/bijective inheritance.**

The late normalized Penn/Yiddish representation cannot inherit “primary-script qualified” status from this audit. The 1711 YidTakNL edition remains useful as a historical transcription, but it cannot be treated as a lossless atom-by-atom surrogate for the printed source without an explicit error model or a new independently verified diplomatic transcription layer.

This result does **not** reopen or alter C0–C5 M0 validation on the normalized representation. It changes the representation claim only.

Voynich/C7 remains sealed.

## Instrument notes

- General-purpose vision models were used only after the six source pages were committed in Git.
- Exhaustive OCR outputs that became internally inconsistent were discarded rather than averaged into the result.
- Focused cross-model adjudication was used for the retained key discrepancies on all six leaves.

## Controlling state after audit

`C0_C5_M0_PASS__C6_LATE_NORMALIZED_PASS__PRIMARY_BRIDGE_FAIL__LANGUAGE_L_OPEN__C7_SEALED`
