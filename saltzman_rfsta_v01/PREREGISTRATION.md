# VOYNICH-RF-STA-REPLAY-v0.1 — preregistration

**Status:** frozen before RF/STA/aaa Voynich target scoring.

## Correction being made
The upstream ASC source-generalisation handover made representation robustness binding: later Voynich comparisons must survive RF-member / STA-family / connected-`aaa` and must not rely on ASCII decomposition artefacts. Phase 9 v0.1 instead used `Paper/Cipher_paper/enriched_records.pkl`; that run is retained as a non-binding auxiliary consequence test and is not used to select anything here.

## Fixed Voynich sources
- RF1b SHA-256 `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`
- bitrans.c SHA-256 `3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a`
- STA-aaa.bit SHA-256 `622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64`
- regenerated RF1b aaa SHA-256 must equal `c14f43c731f46274f35b604356c6bb96a1186e0836aa9aa2b518666cce854167` before scoring.

RF1b already uses full STA members. Therefore the three correlated projections are: full RF/STA member, STA-family collapse, and connected aaa. They are a robustness hierarchy, not three independent replications.

## Frozen parsing
Physical/text line records are read from RF1b. `<->` is a hard interruption. Periods are RF word boundaries. Only words consisting wholly of full two-character STA members present in the verified STA-aaa table are retained. A non-clean word breaks adjacency. Exactly the same retained word loci are used in all three representations. Each analytical unit is injectively mapped to one Unicode codepoint before the unchanged ED1 scorer.

For connected aaa, the verified table is applied member-by-member inside each original RF word. `~` creates a new analytical unit; colon-connected strokes remain one unit. This preserves the original RF word segmentation rather than allowing the stroke conversion to redefine words.

## Target audit
For each representation, run 20 frozen scorer seeds, 200 null permutations per seed. The representation-specific target is the median Q4 across those seeds.

Binding representation robustness requires, in every representation:
- median ED1_N0 >= 1.10;
- median E1_N0 in [0.90,1.10];
- median ED1_N0 > ED1_N1 > ED1_N3;
- median ED1_N3 in [0.95,1.10];
- at least 80% of the 20 target null-seed Q4 rows satisfy those same structural conditions.

A qualitative attenuation diagnostic is recorded separately and cannot rescue failure of the binding gate.

## Frozen mechanism replay
No synthetic simulation is rerun. Reuse exact Phase-8 artifact `asc-mechanism-cipher-order-v01-final` from workflow run `31800905368`, which contains all 190 documents' median Q vectors. Arms are fixed before the RF/STA targets are opened:
- baseline: `CIPHER_ONLY`;
- canonical: `FIXED_LINE_RESET__POST`;
- sensitivity: `FIXED_CONTINUOUS__POST`.

ATOMIC/LITERAL remains a synthetic-side robustness pair. For each Voynich representation, document and arm, robust distance is the worse of the two synthetic representations.

Primary outcome is ED1 d3 transfer to each newly measured target. Gain = d3(cipher only) - d3(canonical). A robust directional gain requires the lower 95% bootstrap CI of median gain to exceed zero in all three Voynich representations. A robust ED1 match additionally requires canonical median robust d3 <=0.15 in all three.

E1 and full Q4 are secondary but binding for any claim of a *full* Q match. Zero/non-positive E1 is never imputed.

## Adjudication
The exact labels and conditions are frozen in `protocol.json`. If the Voynich target itself fails representation robustness, no positive Voynich-mechanism claim is permitted even if a synthetic arm happens to fit one projection. No mechanism retuning follows failure.
