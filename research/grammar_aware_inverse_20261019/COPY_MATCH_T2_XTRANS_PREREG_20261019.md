# T2 XTRANS1: ZLZI versus TTLI transcription-sensitivity preregistration — 2026-10-19

Registered before any T2 score was calculated on the two alternative layers. The original T2 used research-agent derived `vms_directional_token_v01` with unverified upstream transcription; both T2 variant models failed to distinguish source information from matched copying. This test is an EXPLORATORY TRANSCRIPTION BOUND, not a new independent discovery: the same 80 folio sides were selected for the original analysis.

## Independently sourced readings
Pinned archive `digitalgoldfisj79/Voynichdecomp`, `voynich_transcriptions_slim.json` derived from the OrcusLabs/voynich.science multi-source compilation. Sources recorded in that file:
- TTLI: Takeshi Takahashi, `LSI_ivtff_0d`, source SHA256 `3f3f2af18cde10efe75c582f49b07b651c3397022fcbfa5854fecc424c121afa`.
- ZLZI: Zandbergen–Landini, `ZL_ivtff_2b`, source SHA256 `c7ffff9e1f3ecbec174e234c04f056b2bec14f8d722726c456f108e2c7060db5`.
These are distinct transcription source files, not necessarily independent paleographical decisions. Preserve them separately. Read-only mapping of the original 80 folios to quire from the original source cohort. Do not add pages to increase power. Select lines whose encoded layout marker is EXACTLY one of `@P0,+P0,=P0,*P0` (standard running text; exclude all labeled L, circular/spatial and uncertain other paragraph types), enforce pure [a-z]+ case-insensitively, lowercase, order (numeric line prefix, literal line key), set position inside each line. Reindex each complete line as paragraph index so the old line-entry approximation works; the model sees all tokens on that page in order. **No equality of token counts is required**: the two transcriptions differ.

## Frozen model
Reuse T2 engine `copy_match_t2_score_20261019.js` EXACTLY apart from relaxing original-source population assertion, which becomes `require 80 page cohort, seven quires A–G`. Same 96+OTHER top-vocabulary choice, early first 15 tokens per side not scored, 199 production-block permutations, PRNG seed 20261019, whole-quire holdouts, 1-edit match, priors, all decisions unchanged. Test TTLI and ZLZI separately with no tuning or transfer of learned weights.

## Readouts/decision
Primary `assoc_over_copy_bits`, matched null excess, null SD, permutation p, seven quire fold signs, and full sample counts. Do not claim replicated evidence unless each independently exceeds the original preregistered thresholds (raw >=0.02; excess >=2 null SD; p<=0.05; >=5/7 positive). Even simultaneous success would only confirm sensitivity to transcription, not source-vs-copy mechanism: both models share observational memory and observational twins can exist.

## Further reservations
- Mixed text stream may differ from database-derived RF paragraph sampling; compare internal **TTLI vs ZLZI** first; do not pool their cross-entropies with original RF numbers.
- Confirm corpus's parent archive checksum and source format before declaring full provenance success. Embedded source SHA metadata is not an independent bytewise re-hash of original IVTFF files.
- Exact FORM/CONNECT/LINE_ENTRY remains NOT installed; these are still observational approximations.
