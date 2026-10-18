# RETRACTIONS AND SUPERSESSIONS (READ FIRST)

**R6 (2026-10-17): cold demolition of PREREG 34–37, with the key numbers re-verified.**
- **(a) PREREG 34/34b "his ordering stability is not inflated by text noise" is MISLEADING.** The all-pairs metric was dominated by trivially decisive cross-group pairs.
  - Same-group pairs (126), share decisive: his page 38.9%, line bootstrap 23.0% (ratio 0.59).
  - Pairs within 0.05 of each other (106): his 19.8%, line bootstrap 8.5% (ratio 0.43).
  - Correct statement: the coarse order is robust; the fine order within groups is much less certain than his page shows.
- **(b) PREREG 37 "his order disagrees with physical adjacency" is WITHDRAWN.**
  - The anchor set mixed Pelling's ORIGINAL nesting (vellum flaws) with paint transfers in the PRESENT binding, which are disputed as later paint.
  - The power simulation assumed the answer.
  - Vellum-flaw pairs alone: mean |rank diff| 3.0 vs null 8.33, p 0.035 (3 pairs; leans toward agreement).
  - Status: UNRESOLVED.
- **(c) PREREG 35 secondary "word information beyond glyph style": NOT ESTABLISHED.** It is circular in principle, because his positions are a function of the transition distances.
- **(d) PREREG 36 "no within-scribe respelling": DEMOTED** to "no near-complete replacements detected". Partial respelling is untested.

**R5 (2026-10-17):** PREREG 13 "text cannot resolve individual neighbours" applied to OUR gradient (rank SD 3.8). His order has within-scribe-1 rank SD of about 1.1 under line resampling. The R5 claim that the anchor test is therefore powered is withdrawn with R6(b).

**R4 (2026-10-15):** "Generators not compared" was wrong; about 24 generators had been run in other programmes. The space-versus-line-break claim from one CREMMA cell is superseded by XD1: Nuremberg reproduces the effect, so it is not VMS-specific.

**R3 (2026-10-14):** the posted same-face effect fails size equalisation (z 1.54). The MRM conjugate confidence interval was not a valid test; the corrected excess over null is −0.025, z 7.0. Still standing: conjugate coherence z 6.61 and PERMANOVA sheet z 5.31.

**R2 (2026-10-14):** the PREREG 9 "SUPPORTS" verdict is withdrawn, because the split-leaf statistic passed a shuffled negative control (it was detecting size).

**R1 (2026-10-12):** MDS variance shares were wrong because the denominator used only 4 eigenvalues. No verdict depended on them.

**PREREG 15 note (2026-10-17, Ed):** f116v is NOT stained. The positive control in the edge-soiling test was therefore invalid, and the VOID verdict was for the wrong reason. The test needs a re-run with a valid control chosen in advance.

---

# Horseshoe / bifolium / device / npcompl33t-order programme, 12–18 Oct 2026 (Claude Code session 011WNaWAafdUDyDRMHCZA7pe)

The full record is `RUNNING_RESULTS_horseshoe.md` (~156 KB; PREREG 1–38 plus a stage map, every number with its provenance). This doc is the condensed state.

## Strand A — horseshoe and bifolium structure (12–14 Oct; PR #34 package, PR #35 evidence folder; both merged)
- **Text MDS dim1 is the Currier axis** (A/B AUC 0.9986). The Nomic word-map arc is word size. Page-level image MDS shows no horseshoe. P60 90–95%.
- **Bifolium coherence beyond scribe:** text within quire × scribe, z 11.4; within section × scribe × Currier, z 13.9 (7.4% closer). Sheet > opening. PERMANOVA sheet effect z 6.27 raw / 5.31 rarefied (3% of variance). P60 75%.
- **Same face of a flat sheet:** FAILS size equalisation (R3).
- **Gradient within Herbal-A scribe 1 (PREREG 12):** z 2.52. Partly the Currier A→B axis (z 2.66). Unrelated to the present binding.
- **Text ordering methods on known-order texts (PREREG 14):** at VMS size, at best about 1 in 5 true neighbours are recovered.
- **Interior chain stages (PREREG 16):** 1.5a is indistinguishable from a mixture of earlier and later pages on the full chain (z 0.28). Other interior stages exceed mixing.

## Strand B — npcompl33t's chain and respelling (15–16 Oct)
- **PREREG 17:** memoryless unit substitution of real Latin/Greek never reproduces the SGT12 repeat geometry (lag-1 0.14–0.54 vs VMS 1.02). FALSIFIED. P60 85%.
- **PREREG 18:**
  - chedy rises along the chain (ρ .88).
  - chy~chey anti-correlation unresolved (ρ −.63).
  - 1a-vs-Bio asymmetry is a size effect.
  - 1a-vs-Stars asymmetry survives rarefaction (z 4.5).
- **PREREG 19:** context inheritance VOID (the self-match positive control failed).
- **PREREG 20:** "information preserved" (IP) does not discriminate (real 0.8625 vs scrambled 0.8637). FALSIFIED.
- **PREREG 21:** the respelling excess fails to replicate in Takahashi (Stouffer −3.12). Only the A→B steps (1.5a→2a, 2a→3a) are positive, and those coincide with scribe change.
- **PREREG 22 (his optimizer, Chain code, generic edges):** VMS real R_LP 0.580 vs page-shuffled 0.596 (SD 0.015). FALSIFIES ordered respelling.

## Strand C — generators and devices (Brian Cham's volvelle; Ed's multi-arm and 3-dial ideas) (16 Oct)
- **PREREG 24–26:** volvelle vs self-citation on the repeat profile. The volvelle is best in raw fit but loses decisively once refined: locked TEST D 5102 vs 263. With wear, rings build non-words (attested share 0.355).
- **PREREG 27:** two interleaved streams lose to one stream. VMS edit similarity peaks at lag 2; length pairing is a lag-1 effect.
- **PREREG 28:**
  - Length pairing is ~86% junction + page.
  - New held-out word types are 84.4% one-glyph edits of known words (null 19.3%). Latin: 33% one-edit, 52% recombination.
  - A German (ReM) reference is still needed.
- **PREREG 29 (multi-arm device):** beats the copier on the repeat profile, but overproduces novel words (31–34% vs 14%). The verdict falsifies an overall advantage.
- **PREREG 30 (realistic ring-device coverage):** best case 81% of held-out tokens, 37% of novel types, decoys 5%. Latin coverage is far worse. Falsifies realistic device coverage.
- **PREREG 31 (Stolfi crust/mantle/core on 3 dials, read across a diameter):** falsified. The paired dials are too rigid (novel types 8%). The left/right crust coupling is real (MI z 105), but not resolved against a naive first-2/last-2 split (ratio 1.35).
- **PREREG 32 (salvage):**
  - 5 independent Stolfi dials × 36 + one-glyph slip: tokens 97.7%, novel 84.7%, decoys 30.6%. Meets the bar.
  - BUT "known word + one slip" does the same (97.9% / 84.4% / 18.5%).
  - Device is sufficient, not necessary.
- **PREREG 33 (dial turn vs copy slip):** primary VOID. Secondary and post-hoc measures: local relations are flat over lags 1–2, which is copy-like, not a sequential dial. The multi-arm device was not tested here.

## Strand D — npcompl33t's chronology-stability page (17–18 Oct)
- His ElPiGraph ensemble keeps the MDS FIXED and only drops 10% of points.
- **Text resampling of his own code** (gate: Spearman 0.998; per-line ZL counts sum exactly to his):
  - all-pairs decisiveness ~90% vs his 92%;
  - same-group pairs 23% vs 39% (R6a).
- **Combined-uncertainty stage map** (200 replicate sets, 1,600 fits; text + his point masks + 8 starts + λ .01/.05). 7 settled stages, against 13 on his page:
  1. 1a (10)
  2. 1b (8)
  3. ONE unordered block of 20 (1.5a, 2a, 3a, 3b, 5a, 6)
  4. 3c (4)
  5. T:107-112
  6. 4c (2)
  7. bio 4a (5)
  - P60 80–85%.
- **Order vs vocabulary (PREREG 35):** within section × scribe, bifolia closer in his order share more words (ρ −0.69, z −9; TTLI same). This is largely glyph style. Across quires: z −8.5.
- **Order-split novelty (PREREG 38):** random splits flatter forward novelty by ~4–5 points. Direction unresolved.
- **Bio pinned at the curve end:** all pool50 bio bifolia sit at 1.000; 13 of 20 bio pages at 1.000. No internal Q13 order is resolved.
- **Mauro / Layfield & Davis 2026 (Digital Medievalist 19, "Singulion Structure"):**
  - Their method: LSA on ZL v2a, exhaustive best-of-permutation search, no null reported.
  - Their own top-2 Q13 orders disagree almost completely.
  - Q20 agreement is two text-similarity methods on the same text, not independent. His 3c/4c split leaves a 1/48 internal-match chance, and his internal Q20 positions are unsettled.
  - 105-first and 103/116-last are robust.
- **Physical anchors:**
  - Pelling's clean original-state list is f2v→f3r, the vellum-flaw chain f35-36-9-10|15-16-37-38, and Q13B 84-75 around 78-81 (Davis agrees on the 78-81 centre).
  - Only 5 pairs, and already unblinded, so no confirmatory test is possible.
  - Route forward: Davis's upper-margin water-stain shapes (first 7 quires, pre-misordering). Images are available: 162 IIIF pages are on disk and the Yale manifest is reachable.

## Super Grammar dependency (17 Oct)
- **SG23_REMOTE_STATE_REUSE** ("rather than continuous page drift") was tested only against drift in PHYSICAL order. Drift along an order unrelated to the binding is live (cross-quire vocabulary follows his order, z −8.5). That check is circular, so it needs a 1-D order vs K-state held-out comparison. Interpretation OPEN; the recurrence statement stands.
- **Claims register v03:** the `certification_status` column is stale; `certificate_status` is correct. The export was fixed in PR #36 (merged 2026-10-15). The Supabase v03 table is NOT changed, because it is a release with certificate md5s.

## Open items
1. Re-run PREREG 15 with a valid second stained control.
2. Water-stain-shape anchors from images.
3. Respelling test calibrated on partial replacements.
4. SG23 1-D vs K-state comparison.
5. ReM German reference for novelty.
6. Multi-arm device through PREREG 33.
7. Fresh-chat demolition of the stage map.

## Where the files are (as of 2026-10-18)
- **Cloud workspace (EPHEMERAL, reclaimed when the session ends):** `/home/claude/horseshoe/`
  - `RUNNING_RESULTS_horseshoe.md`
  - `scripts/s1–s55*.py`
  - `s31_out` … `s55_out` (results JSON and stage pickles)
  - `post/` (notes to npcompl33t, figures)
  - `wear/img` (162 IIIF page photos)
  - `handoff/`
  - External clones in `/home/claude/ext` (voynich-clusters, voynich-optimizer, Vulgate).
- **GitHub digitalgoldfisj79/Voynichdecomp:**
  - `research/bifolium_order_tests` (PR #34, merged)
  - `voynich_structure_evidence/` (PR #35 merged; PR #36 certification tables, merged 2026-10-15)
  - Scripts s31–s55, the running file and the stage map are NOT yet in the repo.
- **Supabase handoff_docs:** this doc, plus `voynich_npcompl33t_order_stage_map_20261018` (stage membership).
