# VBM v143 addendum: restore Currier A and full-manuscript scope
14 October 2026 · **SCOPE CORRECTION / FIRST A+B SURFACE AUDIT; NOT MECHANISM IDENTIFICATION**

## RETRACTED AND NARROWED FINDINGS (top of running record)

- **NARROWED:** v142/v143 B-only glyph-support failures cannot be extrapolated to Currier A. The original cell and glyph inventory was fitted from an author-supplied Currier B REVA transcription; A had not been evaluated.
- **NARROWED:** v143's exact German/Hartlieb decoder and state-carry effect remain synthetic engineering successes, not tests of A or B historical state transmission.
- **UNCHANGED:** v143's historical identification remains NO-GO; the B-only fit and synthetic carry detection do not establish Voynich plaintext.
- **No new numeric finding has been retracted** in this addendum.

## Admission and data

Primary public transcription: Zandbergen–Landini `ZL3b-n.txt`, version 13 May 2025, IVTFF 2.0, mirrored at [GitHub](https://github.com/Aspect-Research/voynich-autoexploration/blob/master/data/transcriptions/eva_zl3b.txt), blob `2a4533ab9bdfa85db9bad602d590978953055df1`. A companion `currier_herbal_transfer.js` implements the parsing and resampling; download the primary source at that URL to rerun. The public source was parsed using the connected GitHub fetch service. A local copy of the 411,671-byte source is **not** included in this package.

Only P (running text) loci in folios labelled `$L=A` or `$L=B`; excludes labels, circles, radii, tokens with transcription uncertainty or markup fragments. For the table below, local sections use `$I=H` Herbal, `$I=B` Biological, `$I=P` Pharmaceutical, `$I=S` Stars, `$I=T` Text/other, `$I=C` Cosmological; these category letters follow source metadata and should not be casually equated to more granular research section labels. The parser treats comma and dot as separators and excludes tokens intersecting `<?>` and the written `<->` split marker. Specific treatment of uncertain breaks is a representation sensitivity requiring further examination.

## Actual usable running-text coverage

| Group | Usable certain words | Folios |
|---|---:|---:|
| Currier A total | 9,902 | 114 |
| Currier B total | 22,178 | 83 |
| Herbal A | 6,781 | 95 |
| Herbal B | 3,037 | 32 |
| Pharmaceutical A | 2,195 | 16 |
| Biological B | 6,025 | 19 |
| Stars B | 10,628 | 23 |

A and B have different section and scribal-hand mixtures. In Herbal specifically A is hand 1 throughout; Herbal B is hands 2, 3 and 5. **No within-hand A/B comparison can distinguish their effects**. This reproduces the critical confounding warning from the previous Super-Grammar work.

## Section-matched transfer feasibility assay, not an attribution test

Within Herbal, leave a whole physical quire out as test material. From the other quires draw **1,500 training tokens** separately from the same Currier class and from the opposite class; exclude the test quire in both training pools. Repeat 100 deterministic train subsamples per test quire, pool word-level errors by test-word count. Score: fraction of test words containing at least one *raw-character trigram* absent from the relevant training inventory. This does **not** test v143 carry state or plaintext.

| Test section | Representation | Within-class unsupported | Other-class unsupported | Descriptive gap |
|---|---|---:|---:|---:|
| Herbal A | EVA | 8.55% | 13.83% | +5.28 points |
| Herbal A | REVA compound-atom reversal | 8.66% | 13.86% | +5.20 points |
| Herbal B | EVA | 7.53% | 24.93% | +17.39 points |
| Herbal B | REVA compound-atom reversal | 7.57% | 26.54% | +18.97 points |

Matched-null SD: **not available**. Random resamples are for description and do *not* qualify as nulls. The disparities are descriptive and may depend on hand, textual ecology, quire, conditional vocabulary, or transcription. Consequently we do not infer a Currier-specific generating mechanism.

## Implications for continuing v143-derived research

1. **Common grammar, conditional parameters:** Preserve v142/v143's compositional and cross-word-permutation architecture as a *synthetic hypothesis*. Introduce separate section/hand-conditioned surface diagnostics without allowing an unrestricted A/B retuning to count as historical evidence.
2. **No mixing sections:** Primary tests within Herbal A, Herbal B, Pharmaceutical A, Biological B and Stars B; running text only. Labels are evaluated separately. Section/hand/quire stratification controls evaluation; Currier remains a descriptive coordinate, not a presumed cause.
3. **Representation discipline:** Public EVA and author's bespoke reversed-EVA are not interchangeable without documented atomisation, reversal, uncertain-space handling and a verified correspondence. Confirm B replication against the author's reference before any new mechanistic comparison.
4. **Historical-identification gate:** A frozen v143-derived observable must be calculable without plaintext and must beat capacity-matched autonomous state generators under hard controls in *both* A and B strata. An improvement merely from permitting a new free table for A is insufficient. If no observable separates the mechanisms, stop.
5. **Cross-regime transfer:** After within-class validation, freeze constraints and test A→B and B→A transfer separately, preserving quire holdout and preventing fit on target. Interpretation must remain confound-aware.

**Decision:** Extend the data universe now. The current test establishes the need for two-sided surface admission, but neither v143 nor a replacement mechanism is promoted. No Currier-A historical carry test has been passed.
