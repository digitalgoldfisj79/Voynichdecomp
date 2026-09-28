# RUNNING RESULTS — Super-Grammar v03 release review (2026-09-28)

## RETRACTED / DOWNGRADED FINDINGS (this review)
1. [MINE, corrected] "18 hard-zero certificates" -> 19 (counted from certifier HARD_ZEROS tuple). 19 of 32 certified claims.
2. [MINE, corrected] M1 wording: evidence IS linked at claim level (31/32 claims); only the obligation-level link table is empty.
3. [MINE, falsified expectation] I predicted the line-order bug inflated SGK05. It deflated it (2.13 -> 2.90).
4. [RELEASE, recommended downgrade] SGK05 / SGT07 CERTIFIED -> CERTIFIED_BOUNDED: pre-registered T4 volume-matching falsifier fired (median worst-cell ratio 1.13; JSLI only).
5. [RELEASE, recommended restatement] ci/dh/lh hard-zero magnitudes (ratios 25.8-39.3) are raw-EVA stroke-split artefacts; atomic-frame ratios 3.0-3.3.

## Scope
Object under review: Supabase project `research-agent` (ymaqlcfjmdwncdbjprmw), tables `vms_supergrammar_*_v03`,
release row `supergrammar_v03_release_20260920`, manifest_sha256 `c4bbace849cf1cc79ff3999f02a2031ddc888bc3aa222a78a2174c3b6253027e`,
frozen 2026-09-20 08:43:39 UTC. 9 theorems, 32 certified claims, 2 failed, 5 excluded open kernel claims, 9 retractions.

## PRE-REGISTRATION (written before any audit query beyond the release row)

### A. Machinery integrity — BLOCKER if any fires
- A1 FALSIFY release: any theorem in the release manifest whose REQUIRED/NEGATIVE_CONSTRAINT dependency is not CERTIFIED/CERTIFIED_BOUNDED in certificates_v03.
- A2 FALSIFY: any certified claim with a REQUIRED obligation in OPEN or FAIL.
- A3 FALSIFY: stored validation_v03 PASS rows that do not reproduce when I recompute the same invariant independently in SQL.
- A4 FALSIFY: manifest counts (theorem_count 9, certified_claim_count 32, failed 2, excluded 5, retractions 9) do not match the live tables.
- A5 FALSIFY: manifest hash cannot be reproduced from the stored manifest JSON under any documented canonicalisation (flag as UNVERIFIABLE, not failed, if canonicalisation is undocumented).
- A6 FALSIFY: live tables modified after frozen_at (2026-09-20 08:43 UTC) without a new release row.
- CONFIRM: all six pass.

### B. Scientific content — per certified theorem
- B1 Effect-size discipline (user rule 16): every quantitative claim underlying a certified theorem stores effect AND null SD; ratio >= 2. FAIL = certified on ratio < 2 or with no null SD.
- B2 Naive baseline (user rule 19): any theorem that uses PGCS-flavoured features must have a naive first/last-n partition comparator. FAIL = PGCS-feature theorem with no naive baseline in evidence.
- B3 Circularity: were thresholds/folds tuned on the same data used to certify? FAIL = certification evidence drawn from a run flagged for tuning logic (e.g. sgv03_closure_v1 INVALIDATED for HYPERPARAMETER_TUNING_LOGIC).
- B4 External check (rule 20/21/26): has any certified theorem been re-derived in a fresh context or externally? If none, the release must say "not externally replicated". FAIL = release language implying established status without that caveat.
- B5 Language audit: certified statements that are unfalsifiable or phrase a bound as a positive finding.

### C. Primary-data spot check
- C1 Re-derive at least one certified headline number from the Voynichdecomp repo transliterations.
  FALSIFY: my re-derivation differs in sign or by more than the stored null SD.
  CONFIRM: same sign and within one stored null SD.

## LOG

### 2026-09-28 — A4 counts (SQL, live tables)
PASS. certificates: CERTIFIED 2 + CERTIFIED_BOUNDED 30 = 32 (manifest 32); FAILED 2 (2); retractions 9 (9); certified_theorems view 9 (9).
Other buckets: OPEN 55, REJECTED 8, RETRACTED 1, SUPERSEDED 5. claims_v03 holds 109 rows: 103 base run + 3 extension_gallows_20260924 + 3 v04_realization_library_20260924.

### A6 post-freeze modification
PARTIAL FAIL (hygiene, not science). validation_v03 check 6 `repository_release_landed` checked_at 08:46:17 > frozen_at 08:43:39; manifest embeds only 5 validation checks. Two post-release runs (20260924) write into the SAME v03 tables (claims/evidence/obligations) with run_ids distinct from the release; they have 102 obligations with checked_at NULL and no certificates. Frozen rows themselves not modified (max timestamps ≤ 08:40:50 for release run).

### A1 dependency closure (independent SQL recompute)
PASS. 0 violations across 32 theorem→lemma edges. The kernel table's status vocabulary ('SATISFIED') differs from certificate vocabulary — cosmetic.

### A2 certified-claim obligations
PASS on its own terms: 543/543 required obligations PASS for the 32 certified claims.
BUT (MACHINERY FINDING M1): vms_supergrammar_obligation_evidence_v03 = 0 rows. No obligation is linked to an evidence row.
MACHINERY FINDING M2: obligation notes are batch-templated. Per obligation code, 32 claims share ~9 distinct notes; within a claim one sentence closes all 17 obligations (e.g. " Clean-room v1 and source audit close this obligation."). "543 obligations discharged" ≈ ~9 claim-group decisions × 17 labels. Costume-claim risk if the release advertises obligation counts.
MACHINERY FINDING M3: SGK02B_ORDER3_NOT_ROBUST is certified with evidence_count = 0; its AUDIT_COMPLETENESS and EFFECT_BOUND obligations are PASS. The numbers exist (manifest retraction text: ZLZI α=16 LTR −0.00098664/0.00247465; RTL +0.00371542/0.00255975 = 1.45) but are not an evidence row. Self-contradiction with the obligation definition.

### B1 effect/null-SD per certified claim (stored evidence, NOT yet re-derived)
| claim | cert | worst-case effect / null SD | ratio | note |
|---|---|---|---|---|
| SGK01 order2 | CERTIFIED | 0.2312 / 0.0401 | 5.77 | worst of 6 layers×2 dirs×4 α |
| SGK05 SPACE>LINE_BREAK | CERTIFIED | 0.1013 / 0.0475 | 2.13 | NARROW: 10% larger null SD → 1.94 |
| SGK04 SPACE edge | BOUNDED | 0.0372 / 0.0152 | 2.44 | ZLZI only; not universal across layers |
| SGK06 q→k/t | BOUNDED | 0.0237 / 0.0050 | 4.77 | 4/5 alt layers |
| SGK07 i-run termination | BOUNDED | 0.0754 / 0.0127 | 5.92 | all layers |
| SGK08 prev-token morphology | BOUNDED | 0.00735 / 0.00191 | 3.84 | rule-19 naive baseline: TO CHECK |
| SGK13B R64 banded reuse | BOUNDED | 0.00567 / 0.00254 | 2.23 | NARROW; targeted coordinates only; total joint z=1.59 unresolved |
| SGK03_HZ_* (18 hard zeros) | BOUNDED | expected-count effects 7.2–1013 | 2.99–40.5 | stored caveat: same-corpus discovery selection |
| SGK09B, SGK13C, SGK16, SGK17 | BOUNDED | ratios 2.72, 1.55, 1.07, 1.18 | — | negative/bounding lemmas; ratio<2 appropriate to their negative statements |
| SGK15 codebook rejected | BOUNDED | null / null | — | adequacy gate, no effect row (mean joint 1.140 vs 3.061 lives only in manifest text) |
| SGK02B order3 not robust | BOUNDED | — | — | NO EVIDENCE ROW (M3) |

CONTENT FINDING C-HZ (analytical claim — needs primary-data check before relying): the 19 hard-zero certificates [CORRECTED from 18 → 19 after counting certifier HARD_ZEROS tuple; 19 of 32 certified claims] are not 18 independent regularities. By inspection they group into: n-after-non-i {dn,kn,ln,pn,tn,yn}; gallows-gallows {kk,kp,pp,pt,tp,tt}; gallows+m/l/r {km,tm,pl,tr}; h-after-non-bench {dh,lh}; ci. At least {ci, dh, lh} look like consequences of EVA splitting the bench glyph into c+h strokes — true by transliteration convention, undefined in STA. CIRCULARITY obligation is PASS while the evidence row itself says discovery selection was in the same corpus.

### C-HZ confirmed from Supabase (vms_supergrammar_forbidden_bigram_v01, atomic vs raw frame)
Certified ratios for ci/dh/lh are RAW-EVA: 34.6 / 39.3 / 25.8 (expected 914 / 1013 / 501).
Same table, ATOMIC frame (bench glyph merged): ratios 3.3 / 3.2 / 3.0 (expected 8.3 / 9.3 / 8.6); ci has 105 atomic opportunities vs 12,060 raw.
=> the headline magnitude for these three is a transliteration-convention artefact (~10x inflation); the zero itself survives at ratio ~3.
Other 16 hard zeros: raw and atomic ratios agree within ~20% (e.g. kk 31.0 vs 24.3, tt 17.0 vs 14.2).
All 6 replication layers (ZLZI, ZLZB, TTLI, JSLI, VDRB, TTIA) are EVA-family — REPRESENTATION_DEPENDENCE PASS does not cover non-EVA segmentation.

### IMPLEMENTATION FINDING I1 — line-order bug in certifier (VERIFIED)
research/supergrammar_v03_certifier_20260920.py line 137: re.match(r"(\\d+)", label) matches a literal backslash; returns None for "12".
=> line_no = 0 for every line; build_junction_pairs sorts by (0,label) = lexicographic ("1","10",...,"19","2","20",...).
206/235 pages have >=10 lines. LINE_BREAK pairs therefore include non-adjacent lines (1->10, 19->2, 2->20, 29->3, 31->4...).
Also davis_hand(f115r, 0) assigns all f115r lines to S2.
Scope: affects SGK05 (SPACE>LINE_BREAK, CERTIFIED) and SGK04 base-model hand conditioning (f115r only). NOT within-token claims.
boundary_ed1_stage sorts by x["line"] (all 0) -> stable -> JSON order, which is physical for 234/235 pages: SGK16 essentially unaffected.
closure_certifier line 109 uses the correct r"(\d+)" — SGK08/SGK09B unaffected.
Script sha256 on main == manifest registered sha256 for all 4 release scripts (provenance holds; the bug is in the registered code).

## PRE-REGISTRATION — SGK05 bounding tests (written BEFORE running)
Criterion inherited from the certifier: SGK05 certified iff SPACE_MINUS_LINEBREAK effect>0 and ratio>=2 in ALL 6 layers x 4 alphas.
- T1 REPRODUCTION: unmodified junction_stage. CONFIRM: worst ratio = 2.1336 +/- 0.001 and worst effect 0.10134 +/- 0.0005. FALSIFY: otherwise (then my harness is wrong and T2-T4 are void).
- T2 BUG FIX: physical line order (r"(\d+)"), otherwise identical. FALSIFY SGK05 certification: any cell ratio<2 or effect<=0. CONFIRM: all 24 cells >=2.
- T3 PARAGRAPH CONFOUND (on T2 ordering): LINE_BREAK restricted to within-paragraph junctions (right line u starts '+'); cross-paragraph ('@') reported separately. FALSIFY "physical-line attenuation": any cell ratio<2. CONFIRM: all >=2.
- T4 SAMPLE-SIZE CONFOUND (on T2 ordering): in each training fold, SPACE training pairs subsampled without replacement to the LINE_BREAK training count, stratified by section; test sets untouched; 10 seeds. FALSIFY "SPACE>LINE_BREAK is not a data-volume artefact": median-over-seeds worst-cell ratio <2. CONFIRM: >=2.
Expectation stated in advance (unverified): T2 raises LINE_BREAK gain modestly (true adjacency restored) and lowers SGK05 margin; I do not know whether it crosses 2. P(T2 falsifies)=40%, P(T4 falsifies)=50%.

## RESULTS — SGK05 bounding tests (scripts/sgk05_bounding.py, out/sgk05_bounding.pkl, out/sgk05_summary.md, out/sgk05_t4_by_alpha.md)
Adjacency audit: 646 / 3,893 certifier LINE_BREAK pairs (16.6%) join physically non-adjacent lines.
- T1 REPRODUCTION — CONFIRMED. Worst cell JSLI α=4: effect 0.10134 / null SD 0.04750 = 2.134 (stored 2.1336). Harness valid.
- T2 BUG FIX — CONFIRM criterion met (did NOT falsify). Worst cell JSLI α=4: 0.12958 / 0.04474 = 2.896; 24/24 cells ≥2. The bug UNDERSTATED SGK05. My advance expectation (bug inflates it) was wrong.
- T3 PARAGRAPH CONFOUND — CONFIRM criterion met. 119 cross-paragraph junctions removed (3,893→3,774); worst 0.12891 / 0.04454 = 2.894; 24/24.
- T4 SAMPLE-SIZE CONFOUND — FALSIFIED (pre-registered criterion fired). Median over 10 seeds of worst-cell ratio = 1.134 (min 0.597, max 1.903). The metric does not resolve SPACE>LINE_BREAK in the worst cell once SPACE training volume is matched.
  Localisation (descriptive, after the verdict): failure is confined to JSLI. The other 5 layers pass all 4 α in 10/10 seeds (median ratios 3.65–5.51). JSLI medians 1.58 / 2.13 / 2.01 / 1.15 at α 4/16/64/256.
  JSLI coverage: 1,013 P-lines, 7,315 tokens, 91 folios (21% of ZLZI's 34,087 tokens). JSLI is also the non-resolving layer for SGK06 and SGK08.
  Also observed: LINE_BREAK held-out gain is NEGATIVE at α≤16 in every layer (adding previous-line final glyph hurts prediction). "SPACE > LINE_BREAK" is mostly "SPACE positive, LINE_BREAK null-or-overfit", not two positive dependencies of different strength.
DISPOSITION RECOMMENDED: SGK05 CERTIFIED → CERTIFIED_BOUNDED, scope "resolves in 5/6 layers under volume matching; JSLI (21% coverage) does not". SGT07 inherits BOUNDED. Fix line-137 regex and re-issue numbers (T2 values).
P(survives 60d): SGK05 as BOUNDED 5/6-layer claim = 85%; as currently CERTIFIED all-layer claim = 20%.

### A5 manifest hash
PASS: manifest_sha256 == sha256(release_manifest::jsonb::text) (Postgres jsonb text form). Corpus/rows/folds hashes use Python compact sort_keys JSON. Two canonicalisations — undocumented in the release doc.

### Hard-zero replay recompute (scripts/hardzero_and_coverage.py)
Certifier code on primary data: min ratio 3.551 (pn: expected 10.44, null SD 2.941) — MATCHES release doc line 23.
DB evidence rows for SGK03_HZ_* store forbidden_bigram_v01 values instead (pn 7.236/2.423 = 2.99; ci 914.4/26.44 = 34.6) while labelled "primary raw row plus independent positional-null replay". Doc and DB disagree for all 19. The two nulls differ by 20-40% in expected count (yn 11.8 vs 18.3; ci 914 vs 744). No pattern falls below 2 under either null in raw EVA. The replay has no atomic-frame arm.

### B2 naive baseline (rule 19)
SGT11 "morphology" = morph(prev) = (prev[:1], prev[-2:], len(prev)) in closure certifier line 123-125. It IS a naive partition, so rule 19 is satisfied by construction; no PGCS-slot claim exists in the v03 kernel. The word "morphology" overstates it.

### B4 external replication
None. "Independent replay"/"clean-room" = separate script on held-out bifolium folds, same pipeline/agent family. No fresh-context demolition recorded (rule 20). This review is NOT blind — I read the handoff docs first.

### B5 language audit (release doc)
- L13 "32 certified ... claim certificates": costume — 19 are bigram zeros that collapse by inspection to ~4 regularities (n needs preceding i; no gallows-gallows; gallows not before m/n/r/l; EVA bench composition ci/dh/lh).
- L21 "representation-robust": six EVA-family transliterations only.
- L47-51 SGT07 CERTIFIED: see T4.
- L79 "morphology": naive edge/length features.
- "Independent" throughout: same-pipeline held-out re-run.
- L135 "All frozen release invariants pass": true, but obligations are batch-closed per claim group (M2) and the obligation-evidence link table is empty (M1).
- SGT01 statement repeats SGT08's order-3 clause.

### OVERALL VERDICT
Machinery: fail-closed graph is internally consistent (A1, A2, A4, A5 pass); provenance holds (script hashes match). Not release-ready as-is: 4 blockers (SGK05 status + regex fix; hard-zero evidence/doc mismatch + atomic restatement; SGK02B evidence row; costume counts), plus language fixes.

### P(survives 60d)
| finding | P |
|---|---|
| I1 line-order regex bug exists in registered certifier | 99% |
| M1-M3 machinery findings | 95% |
| SGK05 survives as BOUNDED (5/6 layers) | 85% |
| SGK05 survives as CERTIFIED (all 24 cells) | 20% |
| ci/dh/lh magnitudes are EVA artefacts (atomic ratio ~3) | 90% |
| 19 hard zeros reduce to ~4 regularities (by inspection, not tested) | 70% |
| SGT01 order-2 (ratio 5.77) | 90% |
| SGT09 R64 (ratio 2.23, targeted coordinates) | 50% |
| SGT10 conditional graphotactics | 80% |
| SGT11 prev-token edge/length residual | 70% |
| Negative bounds SGT04/05/06/08 | 85% |

# v03.1 CORRECTION RUN — 2026-09-28 (Ed: "Do it")
Script: research/supergrammar_v031_certifier_20260928.py (copy of registered v03 certifier + changes C1–C6 listed in its header).
Atomic frame A1 (declared, not reverse-engineered from v01): ch sh cth ckh cph cfh -> single units, longest first. ZLZI residual standalone c=98 (raw 11,962), h=193 (raw 16,232).

## PRE-REGISTRATION (before running v03.1)
- R1 char stage re-run is a REPRODUCTION (line order irrelevant to within-token). CONFIRM: ORDER1_TO_ORDER2 worst = 5.7655 ± 0.001; ORDER2_TO_ORDER3 fails pass_all. FALSIFY: otherwise → stop, harness drift.
- R2 SGK05 full volume, fixed order: expect T2 values (worst 2.896). FALSIFY SGK05-as-BOUNDED if any of 5 full-coverage layers (not JSLI) has a cell <2.
- R3 SGK05 volume-matched (10 seeds): pre-registered falsifier already fired in T4; this re-run must reproduce median worst 1.134 ± 0.01. If it does, SGK05 status = CERTIFIED_BOUNDED with scope "5 full-coverage layers; JSLI unresolved".
- R4 atomic hard zeros (NEW TEST): per pattern, FALSIFY "hard zero in atomic frame" if ZLZI atomic observed > 0, any alt layer observed > 0, or ratio < 2. CONFIRM if all three hold. Prediction (unverified): ci/dh/lh drop to ratio 2–4; gallows and n-patterns change < 30%. P(at least one of ci/dh/lh < 2) = 35%.
- R5 boundary ED1 with physical order: FALSIFY SGK16 bound if hostile ratio >= 2 (i.e. independent avoidance appears).

## v03.1 RESULTS (out/sgv031_certifier.pkl, result_sha256 fe0bd39b55447fb12ff7c207ae4279d88995d8453991102b081aa0aff52070a1)
- R1 CONFIRMED: ORDER1_TO_ORDER2 worst JSLI α=4 LTR 0.231214/0.040103 = 5.7655 (48/48). ORDER2_TO_ORDER3 fails: 25/48 cells pass; failing cells in all 6 layers; worst VDRB α=4 LTR −0.04482/0.00786 = −5.70 (order-3 HURTS at low smoothing). ZLZI α=16: LTR −0.00098664/0.00247465 = 0.40; RTL 0.00371542/0.00255975 = 1.45 — identical to the v03 manifest text → SGK02B now has a reproducible evidence source.
- R2 CONFIRMED (SGK05 as BOUNDED survives): fixed order, full volume, 24/24 cells ≥2; worst JSLI α=4 0.12958/0.04474 = 2.896; five full-coverage layers 5.40–5.71.
- R3 FALSIFIER FIRED AGAIN (reproduced within tolerance): volume-matched median worst 1.127 (T4 1.134; |Δ|=0.007 < 0.01; sampling order differs: sorted sections). Min 0.882, max 1.463. By layer over 10 seeds × 4 α: ZLZI/ZLZB/TTLI/VDRB/TTIA 40/40; JSLI 14/40. → SGK05 CERTIFIED_BOUNDED, scope "5 full-coverage layers".
- R4 CONFIRMED for all 19 (atomic frame A1, same null): all zero in all 6 layers; min ratio 2.70 (lh). ci 32.23→2.91, dh 41.59→3.95, lh 27.11→2.70 (×0.09–0.10). Remaining 16 change by ×0.81–1.14. My 35% prediction that one of ci/dh/lh would drop below 2 did not happen.
- R5 CONFIRMED bound: ED1 hostile −0.001822/0.001704 = −1.069 (movable 98.46%); identical to v03 (JSON order was already physical).
- Side finding (was already in v03, not disclosed in v03 doc): SGK04 SPACE edge gain is RESOLVED NEGATIVE in JSLI α=4: −0.03594/0.01665 = −2.16. ZLZI worst 0.03810/0.01543 = 2.469 (v03 stored 2.4386; change from the f115r hand fix).

## v03.1 APPLIED — 2026-09-28 (state at end of session)
- Repo: branch `supergrammar-v031-candidate`, commit 630c17a (+ brief commit), DRAFT PR #32 — NOT merged. Registered v03 scripts untouched.
- Supabase (insert-only, run `supergrammar_v031_20260928`): 33 claims; 55 evidence rows (47 v031 + 8 carried); 594 obligations (33×18); 330 obligation→evidence links; 33 certificates (32 OPEN on EXTERNAL_ADVERSARIAL_REVIEW, 1 FAILED); 9 theorems (OPEN; SGT07 target CERTIFIED_BOUNDED); 33 dependencies; retractions −9..−3; verifier sgv031_certifier_local_v1; validation checks 7–10 all PASS (no leak into v03 export view; open obligation blocks certification; EFFECT_BOUND linked; frozen v03 counts and manifest sha unchanged). New obligation type EXTERNAL_ADVERSARIAL_REVIEW (audit_order 18).
- handoff_docs key `voynich_supergrammar_v031_candidate_20260928`, content sha256 ab3cf6a0... == repo release doc.
- One failed attempt, rolled back cleanly: first evidence insert wrote the generated column effect_over_null_sd (transaction aborted, 0 rows).
- Correction to my change-set estimate: evidence rows 55, not ~62.
- A6 finding softened: run_id-keyed sharing of the v03 tables is by design; the actual risk is that vms_supergrammar_certified_theorems_v03 filters by status only, so a future CERTIFIED v031 row would merge into the v03 export surface. Must be scoped before freeze.
- NEXT (rule 20): research/DEMOLITION_BRIEF_supergrammar_v031_20260928.md → fresh chat, no context. Record per-claim outcome as EXTERNAL_ADVERSARIAL_REVIEW.
