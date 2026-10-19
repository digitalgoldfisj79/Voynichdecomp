# Voynich Super-Grammar v03.1 — Correction Candidate

**Date:** 2026-09-28  
**Candidate ID:** `supergrammar_v031_candidate_20260928`  
**Run ID:** `supergrammar_v031_20260928`  
**Status:** `CANDIDATE — PENDING EXTERNAL ADVERSARIAL REVIEW` (not frozen; not exported as certified)  
**Corrects:** `supergrammar_v03_release_20260920` (manifest SHA256 `c4bbace849cf1cc79ff3999f02a2031ddc888bc3aa222a78a2174c3b6253027e`), which remains frozen and unmodified.  
**Certification standard:** `FORTUNE_STYLE_FAIL_CLOSED_V1` plus one added obligation, `EXTERNAL_ADVERSARIAL_REVIEW`.

## Corrections introduced by v03.1 (read first)

1. **Line-order bug in the v03 certifier.** `research/supergrammar_v03_certifier_20260920.py` line 137 used `re.match(r"(\\d+)", label)`, which never matches a line label. Every line received number 0 and the junction stage ordered lines lexicographically ("1", "10", …, "19", "2"). In ZLZI, 646 of 3,893 LINE_BREAK pairs (16.6%) joined physically non-adjacent lines, and all of f115r was assigned to hand S2. Fixed in `supergrammar_v031_certifier_20260928.py`. The bug **understated** SGK05.
2. **SGT07 downgraded, CERTIFIED → CERTIFIED_BOUNDED.** With SPACE training volume matched to LINE_BREAK volume (per fold and section, 10 seeds), the metric does not resolve the worst cell: median worst-cell ratio **1.13** (range 0.88–1.46). The failure is confined to JSLI, which covers 7,315 tokens on 91 folios (21% of ZLZI). The five full-coverage layers pass every cell in every seed.
3. **Hard-zero magnitudes restated.** `ci`, `dh`, `lh` were reported at ratios 32.2 / 41.6 / 27.1 in raw EVA, where the bench glyph is split into `c` + `h` strokes. Under the same null in an explicit atomic frame (`ch sh cth ckh cph cfh` as single units) they are **2.91 / 3.95 / 2.70**. All 19 zeros hold in both frames.
4. **Evidence rows reconciled.** v03 hard-zero evidence rows stored the older `forbidden_bigram_v01` null (e.g. `pn` 2.99) while the release document reported the certifier replay (`pn` 3.55). v03.1 stores the certifier values for both frames.
5. **SGK02B gets an evidence row.** It was certified in v03 with zero evidence rows.
6. **Count language withdrawn.** "32 certified claim certificates" and "543 obligations" are not reported as measures of support. The 19 hard-zero patterns express, by inspection (untested), about four regularities. v03 obligations were closed in batches per claim group with one shared note.
7. **Undisclosed contradiction added.** SGK04 absolute SPACE edge gain resolves **negative** in JSLI at α=4: −0.03594 / 0.01665 = −2.16. This was present in v03 and not stated.

## Authority and scope

This candidate corrects the frozen v03 kernel. It adds no new grammar. Every result below comes from held-out bifolium folds re-run by separate scripts **within the same research pipeline**. No result has been replicated outside that pipeline or by an independent researcher. That is why the `EXTERNAL_ADVERSARIAL_REVIEW` obligation is open on every claim and nothing here is exported as certified.

**Structure.** The kernel holds 9 theorem wrappers over 13 atomic lemmas and 19 hard-zero bigram patterns. By inspection, the 19 patterns reduce to four candidate regularities, none of them tested as a grouping:
- `n` is preceded by `i` (dn, kn, ln, pn, tn, yn);
- no two adjacent gallows (kk, kp, pp, pt, tp, tt);
- gallows are not followed by `m`, `l`, `r` (km, tm, pl, tr);
- composition of the bench glyph (ci, dh, lh).

**Representations.** All six transliterations (ZLZI, ZLZB, TTLI, JSLI, VDRB, TTIA) are EVA-family. "Representation-robust" below means robust across these six only, not across non-EVA segmentations such as STA.

This is not a decipherment, semantic model, historical production proof, cipher identification, or coverage of labels/circular/radial loci.

## Theorem kernel (candidate status; all certificates OPEN on external review)

### SGT01 — Local form grammar — CERTIFIED_BOUNDED (candidate)

Running-text token forms require at least second-order within-token context, and obey 19 graphotactic zero constraints across six EVA-family transliterations.

- **Order-2 context.** The weakest order1→order2 cell across six layers, both directions and four smoothing values is effect **0.231214 bits/event**, null SD **0.040103**, ratio **5.77** (48/48 cells).
- **Hard zeros, raw EVA.** All 19 are zero in all six layers. The weakest ratio is **3.55** (`pn`).
- **Hard zeros, atomic frame A1** (same null). All 19 are zero in all six layers. The weakest ratio is **2.70** (`lh`).
- Second-order dependency is a property of ordinary writing systems generally. This theorem describes the VMS; it does not distinguish it.

### SGT04 — Repertoire bound — CERTIFIED_BOUNDED (candidate; carried from v03, unchanged)

The registered explicit preselected page-family codebook is not part of the architecture.

- No-codebook A4 is adequate in **4/5** frozen folds and inside empirical d95 in **5/5**. Explicit-codebook A3 is adequate in **0/5**.
- Mean joint distance is **1.140162** without the codebook and **3.060952** with it.
- This rejects that registered model class, not every possible page-scale controller.

### SGT05 — Boundary ED1 bound — CERTIFIED_BOUNDED (candidate)

The metric does not resolve independent ED1 avoidance at physical line breaks.

- Hostile page+vertical-quartile shuffle: effect **−0.001822**, null SD **0.001704**, ratio magnitude **1.07**, movable coverage **98.46%**.
- The result is unchanged under corrected line order.

### SGT06 — Currier identifiability bound — CERTIFIED_BOUNDED (candidate; carried from v03, unchanged)

The metric does not resolve Currier as independent of section and Davis hand.

- Currier-versus-hand effect **0.005258 bits/token**, null SD **0.004419**, ratio **1.18**.
- No section+Davis-hand stratum contains both Currier A and B.

### SGT07 — SPACE versus LINE_BREAK attenuation — CERTIFIED_BOUNDED (candidate; was CERTIFIED)

The local edge dependency is stronger across within-line SPACE than across physical LINE_BREAK, in the five full-coverage transliterations.

- **Full volume, corrected line order.** Weakest cell is JSLI α=4: effect **0.12958**, null SD **0.04474**, ratio **2.90** (24/24 cells). The five full-coverage layers range **5.40–5.71**.
- **Within-paragraph line breaks only** (119 cross-paragraph junctions removed): weakest **2.89**, 24/24 cells.
- **Volume-matched** (SPACE training subsampled to LINE_BREAK volume, 10 seeds). The metric does not resolve the worst cell: median worst ratio **1.13**.
  - ZLZI, ZLZB, TTLI, VDRB and TTIA pass 40/40 seed×α cells each.
  - JSLI passes 14/40.
- LINE_BREAK held-out gain is **negative** at α≤16 in every layer. The attenuation is closer to "SPACE carries edge information, LINE_BREAK carries none detectable" than to two dependencies of different strength.

### SGT08 — Order-3 is not core — CERTIFIED_BOUNDED (candidate)

Third-order within-token context is excluded because its gain is smoothing- and direction-dependent.

- **23 of 48** cells fail the gate, with failures in all six layers.
- Worst cell is VDRB α=4 LTR: **−0.04482 / 0.00786 = −5.70**. At low smoothing, order-3 context worsens prediction.
- ZLZI α=16: LTR **−0.000987 / 0.002475 = 0.40**; RTL **+0.003715 / 0.002560 = 1.45**.

### SGT09 — Current-parent R64 recurrence — CERTIFIED_BOUNDED (candidate; carried from v03, unchanged)

Under the SG1.1 no-codebook parent, a banded same-page exact-reuse channel over lags 2–64 improves the **targeted** lag-2–5 recurrence geometry.

- Banded R64 versus no reuse: effect **0.005672**, null SD **0.002538**, ratio **2.23**, 5/5 folds.
- Banded versus uniform: **0.002084 / 0.000938 = 2.22**.
- Total joint improvement over the mutation-off parent is unresolved (**1.59**).
- Analytical caveat, untested: a reuse channel improving reuse statistics is partly built into the comparison.
- The metric does not resolve lag-1 exclusion: **−0.000328 / 0.000211 = 1.55**.

### SGT10 — Conditional graphotactics — CERTIFIED_BOUNDED (candidate; carried from v03, unchanged)

- **q-conditioned k/t selection.** Primary ZLZI effect **0.023703 bits/event**, null SD **0.004965**, ratio **4.77**.
  - Four of five alternates resolve positive.
  - JSLI is positive but unresolved: **0.000917 / 0.002157 = 0.43**.
- **i-run-length-dependent termination.** Both context bounds resolve in all six layers; weakest replay ratio **3.52**.

### SGT11 — Junction morphology — CERTIFIED_BOUNDED (candidate; carried from v03, dependency SGK05 now BOUNDED)

"Morphology" here means the previous token's first glyph, last two glyphs and length (`morph(prev) = (prev[:1], prev[-2:], len(prev))`). It is a naive character partition, not a morphological analysis.

- Primary ZLZI effect **0.007355 bits/token**, null SD **0.001914**, ratio **3.84**.
- Four of five alternates resolve positive. JSLI is unresolved.
- Universal exact previous-token closure is false. TTLI retains **0.001036 / 0.000382 = 2.72**, and the frozen exact-closure criterion passes 2/6 layers.

## Unchanged from v03

- **Superseded wrappers:** `SGT02_JUNCTION_GRAMMAR`, `SGT03_SHORT_MEMORY`.
- **Excluded uncertified candidates:** SGK10, SGK11, SGK12, SGK13, SGK14.
- **Retractions:** the v03 ledger's nine retractions stand. v03.1 adds the seven corrections at the top of this document.

## Verifier ledger

| verifier | script SHA256 | result SHA256 |
|---|---|---|
| v03.1 correction certifier (this candidate) | `c65bcdc2ba88a0cf3692ccf84f5b3e1f1e3327484025c18ec066c968818f2993` | `fe0bd39b55447fb12ff7c207ae4279d88995d8453991102b081aa0aff52070a1` |
| v03 closure/junction replay (SGT11 inputs) | `da33220eff309db753877bbe0618a787344257f74fa2398016ab4e10604180a7` | `774819b6b2f8cfb4d8957011ad0a418ed9b2545ce5f10b2bf57f4f2fb09d0fc4` |
| v03 conditional graphotactics (SGT10) | `5615eb45b72aa5aba763d9df38d2516f8b3dd361f2b477e6e8070f831354ccda` | `15ae35e720ace4dff1475140cdd1eeb737b43f841cd648d573191c3ac6fa8e5a` |
| v03 current-parent R64 (SGT09) | `f55b309d5801fefa98991d79ec991d20e61aaef06d2ca7cd6ec38f14735d2306` | `a282e222a8f83c20091e39d5eca85215f4ec07b4dfc79902d5b8543b95a7d0e8` |
| v03 certifier (superseded by v03.1 for SGT01/05/07/08) | `d0dcd9a24ad3157ee2ea9cdc4cfce7844499afc02e91612244b4b9c84a799b15` | `6b235bd6002d8f51823181b6766061056063384332a6cdc3262bb5d0f9d4b41a` |

Reproduce the v03.1 certifier locally in under two minutes:

`python research/supergrammar_v031_certifier_20260928.py --corpus voynich_transcriptions_slim.json`

## Canonical source hashes (unchanged)

- Corpus SHA256: `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`
- Canonical event rows SHA256: `74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f`
- Canonical fold assignment SHA256: `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`
- Hash conventions:
  - Corpus/rows/folds/result hashes are SHA256 of Python `json.dumps(sort_keys=True, separators=(",",":"))`.
  - The v03 release-manifest hash is SHA256 of Postgres `release_manifest::jsonb::text`.

## Obligation ledger changes

- One added required obligation, `EXTERNAL_ADVERSARIAL_REVIEW`: a context-free adversarial attempt on the claim and data (rule 20), recorded before freeze. It is OPEN on all v03.1 claims.
- Every v03.1 obligation carries a claim-specific note.
- Quantitative obligations are linked to evidence rows in `vms_supergrammar_obligation_evidence_v03`, which was empty in v03.

## Path to freeze

1. External adversarial review of this document and the result pickle, with no supporting context.
2. Record its outcome as the `EXTERNAL_ADVERSARIAL_REVIEW` obligation per claim.
3. Scope the export view by release before any v03.1 certificate turns CERTIFIED. Otherwise it merges into the v03 export surface.
4. Freeze under a new release ID with a new manifest hash. Mark `supergrammar_v03_release_20260920` SUPERSEDED.
