# T2 copying-template control and cross-transcription bound — 2026-10-19

## RETRACTED / NOT PROMOTABLE — READ FIRST
1. **NO validated content-specific residual:** the source-association advantage over a directly memory-matched copying estimator is below the preregistered effect/power cutoffs under an explicit matched-block null. It is not a decipherment or identified source signal.
2. **Derived RF headline not transcription-invariant:** the observed comparison `+0.007333` bits/token on the derived RF table shrinks to approximately `+0.0020` on independent Takahashi (TTLI) and `+0.00186` on Zandbergen–Landini (ZLZI), with matched-null excesses *negative* in both.
3. **No full four-rule qualification:** archived exact FORM oracle exists, but T2 does NOT run or qualify a complete frozen SELECT+FORM+CONNECT+LINE_ENTRY generator. Do not present its context approximation as one.
4. **No inference from observational source/copying twins:** even a future positive association might have an exactly equivalent production-template account; see prior IDENT-GATE exact twins.

## Source, folds and prereg
- [Prospective T2 protocol](COPY_MATCH_T2_PREREG_20261019.md), commit `ab003f8382dcc8dcd68056be1f0926dbc22a03ab` before scoring.
- [Executable T2 scorer](copy_match_t2_score_20261019.js), archived [full null vector/results](results/copy_match_t2_real_20261019.json).
- [Prospective cross-transcription protocol](COPY_MATCH_T2_XTRANS_PREREG_20261019.md), commit `7ec431a26bbb1095a0b660988ecfdac8f3306254` before layer scores.
- [Same scorer with only cohort assertion relaxed](copy_match_t2_xtrans_score_20261019.js), archived [TTLI](results/copy_match_t2_ttli_20261019.json) and [ZLZI](results/copy_match_t2_zlzi_20261019.json).
- All checks: **the same 80 folio sides**, canonical Herbal-A, original Currier A / Davis hand 1, 7 whole-quire folds A–G, first 15 page tokens visible but not scored. No plant names or meanings.
- Both models share identical training-page late-token category histograms, first-15 anchor extraction, 97 output categories, edit-distance-one kernel, Dirichlet-35 smoothing and 0.2 extra component weight. They differ only in aggregation/conditioning of complete copy-donor blocks vs per-anchor cooccurrence profiles. Nominal storage matched; effective nonparametric model flexibility is not rigorously equalized, so this is **not** a qualified full matched-generator negative.
- 199 reproducible within-training-quire late-production-block permutations (length groups of four), same heldout pages and M1 for both models.

## Primary numbers
| Cohort / source | Tokens in source | Scored targets | Copy model bits/token | Association model bits/token | Raw association−copy gain | Matched null mean | Null SD | Effect−null | z | upper p | Quire signs | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Original stored RF-derived table | 4,580 | 3,389 | 4.013742 | 4.006409 | +0.007333 | +0.005882 | 0.001302 | **+0.001451** | 1.115 | .145 | 7/7 | FAIL |
| TTLI, strictly running P0 | 6,245 | 5,045 | 3.894946 | 3.892938 | +0.002008 | +0.003045 | 0.000788 | **−0.001037** | −1.316 | .920 | 6/7 | FAIL |
| ZLZI, strictly running P0 | 6,506 | 5,306 | 3.890050 | 3.888188 | +0.001862 | +0.002919 | 0.000776 | **−0.001057** | −1.362 | .925 | 4/7 | FAIL |

As preregistered, the gate requires ALL: raw gain≥0.02 bits/token, null effect≥2 SD, upper p≤.05 and positive≥5/7 quires. **0/3 eligible variants pass.**

### Additional effect-size bounds (descriptive, post-results)
Reweighting each independent transcription layer's seven observed fold gains to the SAME *original RF* quire-level target-token proportions yields raw gains **+0.0019573** bits/token for TTLI and **+0.0015400** for ZLZI. This bounds one confound (different quire distributions), but it does NOT equalize within-folio token positions or vocabulary, and it lacks a reweighted matched-null estimate. Therefore this is an untested diagnostic, not a significance claim.

Original RF leave-one-quire-out *raw* gain ranges +0.006518–0.007750 bits/token; both layer-specific raw gains are smaller. This does not establish statistically significant differences between transcriptions, because neither an independent matched difference null nor causal mechanism controls were run.

## Transcription provenance
The repository has a primary-source-linked `voynich_transcriptions_slim.json` dataset from OrcusLabs, with source manifest:
- TTLI/Takahashi from `LSI_ivtff_0d.txt`, listed SHA256 `3f3f2af18cde10efe75c582f49b07b651c3397022fcbfa5854fecc424c121afa`.
- ZLZI/Zandbergen–Landini from `ZL_ivtff_2b.txt`, listed SHA256 `c7ffff9e1f3ecbec174e234c04f056b2bec14f8d722726c456f108e2c7060db5`.
These are verified as *manifest entries* in the source archive; original IVTFF bytes were NOT independently downloaded/rehashed in this run. They are different transcriptions of the SAME manuscript, not independent physical manuscript samples or certainly independent editorial judgments.

The stored research-agent table `vms_directional_token_v01` is `relkind='r'`, with no SQL provenance comment. Its upstream transcript is **not established**. The previously examined `voynich_semantic_notation_v1.rf_occurrences` is visibly derived from another RF-style source and is NOT a TTLI/ZLZI replication. We do not silently label the RF table as one of the sourced witnesses.

## Formal/model audit
- **Circularity**: No botanical naming/translation, no claimed meaning; selection of the same 80 pages is from a previous analysis (not fresh confirmatory sample).
- **Leakage**: Seven entire physical quires held out, including both sides of all bifolia; per-fold vocab/anchors derived only from six training quires. Test-page first 15 tokens observable at prediction time. Note original archive's full-corpus trained FORM socket must NOT be used as if fold-independent in later tests; either rebuild FORM with train-only counts or explicitly treat full-corpus socket as a nonconfirmatory oracle.
- **Confounds**: Different page/line tokenization and layout markers, original-derived RF versus TTLI/ZLZI sampling, uncertain external source provenance; cross-layer score differences are not solely glyph-reading differences.
- **Matched nulls**: Within-quire block swaps preserve complete late-token histograms but not early-to-late template alignments. More sophisticated packet-level copying can still explain raw improvements.
- **Control fairness**: Same training memories and priors; parameter-effective degrees of freedom and alternative estimator behavior are not mathematically equated. Neither model can exclusively identify source semantics.
- **Measurement degeneracy**: OTHER category collapses unseen tail; 97-way vocabulary is train-fold specific. AUC/dictionary hits deliberately not measured.
- **Representation dependence**: TTLI and ZLZI give weaker raw advantage and negative matched-null effects. On all three data representations, no gate passes.
- **Decision fragility**: Outcome FAIL irrespective of whether we rely on original derived or authentic separate transcriptions, so no source-identification claim. Exploratory quire weighting cannot alter registered decision.
- **Audit completeness**: Preregs, exact code, data query, metadata, full 199 nulls, and outputs on branch. Full synthetic source-vs-production recovery (≥20 independent families, ≥90% sensitivity and ≤5% false positives) **NOT run**. Fully integrated frozen 4-rule generator **NOT installed**.

## Mechanism test status
**Result**: `FAIL / NONIDENTIFIABLE`. Under the specific observable feature families, heldout long-range association is not separated from the matched production null, including across independent transcription source files. Do not infer that no linguistic/source content exists. The direct comparison falsifies the narrow claim that this *particular predictor and null* expose reliable additional information. More grammar-aware inference is blocked by the previously failed Q1B/IDENT qualification and incomplete full 4-rule synthetic generator.
