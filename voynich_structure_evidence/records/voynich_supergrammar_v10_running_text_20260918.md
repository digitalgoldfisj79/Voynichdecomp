# Voynich Super-Grammar 1.0 — frozen running-text architecture and closure

- key: `voynich_supergrammar_v10_running_text_20260918`
- status: superseded
- created: 2026-09-18T15:09:12.615745+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---


# VOYNICH SUPER-GRAMMAR 1.0 — RUNNING TEXT FREEZE

Date: 2026-09-18
Status: FROZEN_RUNNING_TEXT_V1
Canonical source run: supergrammar_v01_20260918
Supersedes development handoff: voynich_supergrammar_v01_20260918

## Verdict

The major observable grammar of Voynich running text is now held-out closed within the tested structural benchmark.

Frozen held-out loss: 10.197297969462493 bits/token on 34,416 tokens, versus 21.727607757597347 bits/token for the root character model.

The current architecture survives:
- folio-block held-out evaluation;
- raw and compound-atomic representation bounds;
- alternate transcription / word-division replication for the relevant small transition layer;
- systematic parent mediation;
- explicit hard-zero upper-bound tests;
- flexible order-6 and 256-token context competitors;
- nested exact-token transition closure.

The final nested exact-token child is switched off in all five folds (lambda=1e9) and changes codelength by only -2.3195e-9 bits/token. There is therefore no resolved exact previous-token transition residual after SG45.

## Frozen running-text layers

1. Representation/segmentation firewall: editorial separators removed; raw and atomic frames retained; uncertain boundaries not silently equated with words.
2. Regional state: strong page/physical-leaf localisation and reusable remote subword regimes.
3. Repertoire: stable vocabulary-frequency reservoir plus active local repertoire.
4. Physical line frame: line-start/end position is a genuine conditioning coordinate, though partly redundant after richer downstream state is included.
5. Token formation: higher-order token-internal context through order 3; 19 cross-transcription/atomic hard zeros; 15 additional robust avoidances; q-conditioned k/t selection; i-run-length-dependent termination.
6. Surface choice: exact base-preserving alternants are selected by local ecology; d is a local surface realisation, not a routing operator.
7. Short memory: component/subword persistence and medium-range exact recurrence.
8. SPACE junction: strong immediately adjacent edge scaffold plus a small set of representation-robust higher-order exceptions.
9. SG45 previous-token morphology state: minimal raw coordinate first character + final two characters + length; representation-stable and transcription-robust.
10. Exact recurrence cache: distances 2-64, small but held-out resolved.
11. LINE_BREAK: separate strongly attenuated regime; m|p avoidance is the sole transcription-robust individual cell and remains CONDITIONAL.
12. Paragraph level: no individual cell is identifiable with only 82 boundaries; no universal hard reset is licensed.

## Explicitly not independent grammar mechanisms

- global junction dissimilation;
- ED1/copy-neighbour enrichment after line/edge/length parent matching;
- named v2.3 lexemes as individual grammar nodes;
- general exact previous-token bigram state;
- hundreds of old cross-line cells;
- an omnibus direct 2x2 cross-space code;
- global rightward d routing.

## Closure evidence

v0.1:
ROOT 21.727608 -> structural FULL 11.172661 bits/token.
Higher internal context, junction, hard zeros, local subword state and exact R64 each have resolved held-out ablation penalties.
Longer order6/history256 competitors are worse.

v0.2:
Adding the stable vocabulary reservoir lowers loss to 10.200110.
An independently retuned order6/history256 model remains worse at 10.226570.

v0.3:
A tiny exact-token transition residual appears: +0.0029709 bits/token.

v0.4:
That residual is absorbed by previous-token morphology. first2+last2+length slightly outperforms exact identity; exact-vs-family difference unresolved. 97.5% of the raw residual is within ordinary SPACE and 82.2% follows previous tokens with training frequency >=100.

v0.4b:
Minimal raw coordinate is first character + final two characters + length. Compound-atomic P2S2+length gives the same effect within error. Coordinate is not an exact-identity alias by the frozen >=80% criterion.

v0.4c:
Both raw-minimal and atomic morphology states replicate positively in all five alternate transcription layers and resolve in 4/5, with no contradiction. SG45 promoted CORE.

v0.5:
Nested exact identity beyond SG45 selects parent-only lambda=1e9 in 5/5 folds. EXACT_IDENTITY_CLOSED.

## Scope bounds

This freeze covers ordinary running text only.
Labels, radial/circular text, other special loci, uncertain segmentation alternatives and manuscript-level semantics are separate programmes.
Predictive closure does not tell us whether the system is cipher, notation, generated surface text, meaningful language, or hybrid.

# CURRIER A/B NESTED RESIDUAL TEST — PREREGISTERED BEFORE OUTCOMES

Question:
Does Currier A/B carry held-out running-text information beyond Super-Grammar 1.0, and is any such compression independent of Davis hand rather than a coarse label for the same section/hand production partition?

Support audit frozen before outcome computation:
On folios carrying both Currier and Davis metadata, there is ZERO section + Davis-hand stratum containing both Currier A and Currier B.
Observed section/hand support:
- Herbal-A: hand 1 = A; hands 2/5 = B.
- Herbal-B: hand 1 = A; hand 3 = B.
- Balneological: hand 2 = B.
- Pharmaceutical: hand 1 = A.
- Stars: hand 3 = B.
- text-only: hand 3 = A.
Therefore an observational claim of Currier causality independent of BOTH section and hand is not identifiable from this manuscript. The predictive test below can determine whether Currier is a useful residual register/compression coordinate, and whether hand explains at least as much.

Evaluation universe:
Only ordinary running-text folios carrying Currier A/B + Davis-hand metadata in the canonical metadata table. Special loci without Currier classification are excluded. Every physical quire is indivisible.

Outer validation:
5 deterministic folds of whole QUIRES, not pages, keyed by currier-audit-outer. All labeled pages in a quire are held out together. Inner tuning uses 4 deterministic quire folds keyed by currier-audit-inner.

Frozen parent:
Super-Grammar 1.0 structure through SG45:
- raw order-3 internal grammar + hard-zero mask;
- local subword64;
- section vocabulary reservoir;
- SPACE/LINE_BREAK boundary state;
- SG45 previous-token morphology parent = section + boundary + previous raw first character + previous raw final-two characters + previous raw length;
- exact R64 cache.
Use the stable frozen hyperparameters alpha=16, beta=64, vocabulary mixture v=0.60, SG45 shrinkage lambda=1024, cache w=0.05. This audit asks for incremental register information; it does not re-open grammar tuning.

Nested normalized whole-token children:
S_CONTROL: key = section.
CURRIER: key = section + Currier A/B.
HAND: key = section + Davis hand.
HAND_CURRIER: key = section + Davis hand + Currier.

For each child:
P_child(token|key) = (count_key(token) + lambda * P_SG45(token)) / (N_key + lambda).
Tune lambda on inner QUIRE folds from {16,64,256,1024,4096,16384,1e9}; 1e9 is an operational off-switch.
After lambda selection re-tune exact R64 cache weight from {0,.05,.10,.20,.35} on inner training only.

Primary contrasts:
1. CURRIER - S_CONTROL held-out codelength.
2. HAND - S_CONTROL.
3. CURRIER - HAND.
4. HAND_CURRIER - HAND.

Inference:
Paired held-out token codelength with 5,000 deterministic QUIRE-block sign flips. Report effect and null SD together. If abs(effect-null_mean)/nullSD <2: THE METRIC DOES NOT RESOLVE THIS.

Identifiability gate:
Because no section+hand stratum contains both A and B, HAND_CURRIER should be context-identical to HAND on the observed support. If so, Currier-after-hand is structurally non-identifiable, not evidence for zero causal effect.

Decision:
CURRIER_INDEPENDENT_REGISTER is NOT licensable unless a valid section+hand overlap exists; current support audit already blocks that causal/independent interpretation.
CURRIER_USEFUL_COORDINATE if CURRIER beats S_CONTROL by >0.01 bits/token with z>=2 and is not materially worse than HAND (HAND minus CURRIER <=0.005 bits/token or unresolved).
HAND_DOMINATES if HAND beats CURRIER by >0.005 bits/token with z>=2.
CURRIER_REDUNDANT if CURRIER vs S_CONTROL is unresolved/nonpositive, or its gain is fully matched/exceeded by HAND while Currier-after-hand is non-identifiable.
MIXED otherwise.

Interpretation:
A positive Currier result means A/B is a useful coarse register label for predictive compression. It does NOT establish two grammars, two languages, chronology, or a Currier-specific production cause.

## Pre-outcome localisation addendum
Using the fixed held-out predictions from the primary models, decompose CURRIER-vs-S_CONTROL and HAND-vs-S_CONTROL gain by manuscript section without re-tuning. Only sections containing both Currier A and B are considered potentially discriminating; single-Currier sections are reported as descriptive compression only. Herbal-A is expected to dominate the usable same-section support, but Currier remains confounded there with Davis hand. This decomposition cannot rescue the independent-register identifiability gate.

## Currier audit implementation correction — first run invalidated before inference
The first Currier-audit execution used hash(q+salt) mod 5 directly on the 12 labeled physical quires. That deterministic hash happened to allocate ZERO quires to outer fold 4. The preregistered five-fold whole-quire validation requirement therefore failed. All first-run model outcomes are discarded and are not evidence.

Correction, before accepted inference:
- keep the same corpus, parent, child definitions, tuning grids, thresholds and quire-block requirement;
- assign outer folds by sorting the 12 quire IDs on the frozen hash key and distributing them round-robin across folds 0..4, guaranteeing five nonempty folds without using any model outcome;
- assign inner quire folds analogously with the independent currier-audit-inner hash and four-way round-robin;
- require all five outer folds nonempty and HAND_CURRIER to reproduce HAND exactly before accepting results.
No statistical threshold or model definition changes.

# CURRIER PRIMARY NESTED WHOLE-TOKEN AUDIT — ACCEPTED RESULT

Universe: 28,229 canonical running-text tokens on 176 Currier-labelled folios spanning 12 physical quires. Five whole-quire outer folds are nonempty after the preregistered fold-allocation correction.

All four nested exact-token children independently choose the operational off-switch lambda=1e9 in every outer fold:
- S_CONTROL: 5/5 off.
- CURRIER (section+Currier): 5/5 off.
- HAND (section+Davis hand): 5/5 off.
- HAND_CURRIER: 5/5 off.

With lambda=1e9 implemented as the exact parent probability, S_CONTROL, CURRIER, HAND and HAND_CURRIER have IDENTICAL held-out loss: 10.782536184910514 bits/token.
Therefore:
- CURRIER vs S_CONTROL incremental gain = exactly 0.
- HAND vs S_CONTROL = exactly 0.
- CURRIER vs HAND = exactly 0.
- HAND_CURRIER vs HAND = exactly 0.
The generic retuned-cache control versus the fixed parent changes loss by +0.05795 bits/token in the favourable direction, but quire-block null SD is 0.06877, z=0.86: THE METRIC DOES NOT RESOLVE THIS.

Identifiability:
No observed section+Davis-hand cell contains both Currier A and Currier B. Currier-after-hand is therefore not merely statistically null; it is structurally non-identifiable on this manuscript.

Primary decision: CURRIER does not earn an independent whole-token register child on top of Super-Grammar 1.0. Davis hand does not earn one either under the same test.

# CURRIER LOWER-DIMENSIONAL CHARACTER RESIDUAL PROBE — PREREGISTERED BEFORE OUTCOMES

Reason:
The exact-token child is deliberately high-dimensional and could miss a Currier-specific graphotactic shift that generalises to unseen token forms. Run a lower-dimensional normalized character correction before finalising SG12.

Same universe and whole-quire folds as the accepted primary audit. Frozen parent remains Super-Grammar 1.0 through SG45 with alpha=16, beta=64, vocabulary mixture v=.60, SG45 lambda=1024 and exact-R64 w=.05.

For each character event, compute the frozen pre-vocabulary character probability after local subword state and hard-zero renormalisation.

Nested character correction counts:
START events context:
- S_CONTROL: section + boundary type + physical line-position class + previous final raw character.
- CURRIER: same + Currier A/B.
- HAND: same + Davis hand.
- HAND_CURRIER: same + hand + Currier.

INTERNAL events context:
- S_CONTROL: section + previous up-to-two raw characters inside the token.
- CURRIER: same + Currier.
- HAND: same + Davis hand.
- HAND_CURRIER: same + hand + Currier.

For an event target y:
P_child(y|context)=(count(context,y)+lambda*P_frozen_char(y))/(N_context+lambda).
The child is normalized because counts sum to N and the event-specific frozen parent distribution sums to one.

Use ONE shared lambda per model across START+INTERNAL events, tuned on inner whole-quire folds from {16,64,256,1024,4096,16384,1e9}, where 1e9 is an exact off-switch.
After corrected character probabilities are multiplied into a token probability, apply the frozen section vocabulary mixture, SG45 previous-token morphology child and exact R64 cache unchanged. No child is allowed to re-tune those downstream layers.

Primary contrasts:
CURRIER vs S_CONTROL; HAND vs S_CONTROL; CURRIER vs HAND; HAND_CURRIER vs HAND.
5,000 whole-quire sign flips.

Decision:
If CURRIER chooses lambda=1e9 in >=4/5 folds and has unresolved/nonpositive gain versus S_CONTROL, classify CURRIER_REDUNDANT.
If Currier resolves but HAND matches/exceeds it and Currier-after-hand remains non-identifiable, retain COORDINATE only.
No independent Currier-register claim is licensed without section+hand overlap regardless of predictive outcome.


# CURRIER LOWER-DIMENSIONAL CHARACTER RESIDUAL — ACCEPTED RESULT 2026-09-18

Same 28,229-token, 12-quire universe and corrected five whole-quire folds as the accepted primary audit.

Held-out losses:
- S_CONTROL: 10.8426465667 bits/token.
- CURRIER: 10.7915701230.
- HAND: 10.7968282922.
- HAND_CURRIER: 10.7968282922.

Primary contrasts:
- Currier vs section-only: gain +0.0510764437 bits/token; quire-block null SD 0.0215953653; z=2.3596; p_upper=0.00180.
- Davis hand vs section-only: gain +0.0458182745; null SD 0.0190561177; z=2.4092; p_upper=0.00100.
- Currier vs Davis hand: gain +0.0052581692 in Currier's favour; null SD 0.0044194789; z=1.1827. THE METRIC DOES NOT RESOLVE THIS.
- HAND_CURRIER vs HAND: exactly identical on observed support because no section+hand stratum contains both Currier A and B.

Interpretation:
The whole-token child audit showed no Currier-specific exact-vocabulary/register layer: all token children switched off in 5/5 folds.
The lower-dimensional character probe does show a real A/B-associated graphotactic signal beyond section alone.
But Davis hand carries essentially the same amount of held-out information, and Currier versus hand is unresolved.
Since no section+hand stratum contains both A and B, the manuscript cannot identify an independent Currier effect after hand.

Final SG12 decision:
CURRIER A/B REMAINS A COORDINATE, NOT A SEPARATE GRAMMAR SWITCH.
Currier was detecting a real coarse clustering of graphotactic/production state, but this audit does not support two independent Voynich grammars.

Section-localisation caution:
Herbal-A carries the main potentially discriminating same-section A/B support, but there A/B is hand-confounded (hand1=A; hands2/5=B). Gains reported in sections with only one Currier class can arise from the globally tuned shrinkage parameter and are not interpreted as Currier information.

Persistent objects:
- public.vms_supergrammar_currier_audit_models_v01
- public.vms_supergrammar_currier_audit_contrasts_v01
- public.vms_supergrammar_currier_audit_folds_v01
- public.vms_supergrammar_currier_support_v01
- public.vms_supergrammar_currier_char_models_v01
- public.vms_supergrammar_currier_char_contrasts_v01
- public.vms_supergrammar_currier_char_folds_v01
- public.vms_supergrammar_currier_char_sections_v01

# NPCOMPL33T MULTI-REGIME PARENT-MEDIATION TEST — PREREGISTERED BEFORE OUTCOMES

Motivation:
Voynich Ninja thread 6084 proposes reproducible page-level regimes beyond Currier A/B: early Herbal A; late Herbal A + pharmaceutical; Herbal B plus a Stars/Recipes sub-regime; Biological/Balneological; Celestial/hand-4 material; and f58 as a unique outlier. The author explicitly says Herbal B and Stars/Recipes do not clear the bar for separate "languages", so the primary learned register merges them.

Goal:
Test whether those descriptive clusters carry held-out INTERNAL graphotactic information beyond frozen Super-Grammar 1.0, and whether they outperform Davis hand as a predictive coordinate. This is a parent-mediation test of production/register state, not a language-classification test.

Operational page ecology, frozen before outcomes:
- EARLY_A: canonical running-text Herbal-A pages in physical quires 1-7 whose folio is not in the established Currier-B herbal set {26,31,33,34,39,40,41,43,46,48,50,55}.
- LATE_A_PHARMA: canonical Herbal-B/Pharmaceutical pages on folios 87-102 excluding f94/f95; this covers late A/herbal + pharmaceutical ecology.
- B_HERBAL: the established Currier-B herbal folios {26,31,33,34,39,40,41,43,46,48,50,55} plus f94/f95.
- B_STARS: canonical Stars/Recipes pages f103-f116.
- BIO: canonical Balneological pages f75-f84.
- CELESTIAL: canonical Astronomical/Cosmological/Zodiac pages f67-f73, all LFD hand 4.
- f58r/v is NOT used as a learnable held-out register because it occupies only one physical quire and is excluded from the 34,416-token frozen running-text corpus. It receives a separate external-transfer/outlier test.
- Mixed/transitional quire-8 pages f57/f65/f66 and Rosettes f85/f86 are excluded from the primary regime fit.

Primary register:
REGIME5 = {EARLY_A, LATE_A_PHARMA, B_CORE=(B_HERBAL+B_STARS), BIO, CELESTIAL}.
Secondary refinement:
REGIME6_SPLITB = same, but split B_CORE into B_HERBAL and B_STARS.

LFD hand control:
Use Davis page assignments: hand 4 for f67-f73; hand 3 for f94/f95 and f103-f116; hand 2 for the established B-herbal folios and f75-f84; hand 5 for f41/f48; hand 1 for remaining EARLY_A and LATE_A_PHARMA pages. Mixed f115r is treated as hand 3 at page level, matching the dominant page hand.

Target:
INTERNAL token formation only. No token-start, SPACE, line-position or exact-token vocabulary target is scored. This deliberately matches the thread's within-word-transition object and avoids contaminating celestial/radial material with arbitrary text order.

Frozen parent:
Super-Grammar 1.0 raw internal character model through order 3, with section and physical line-position conditioning and the 19 hard-zero constraints. Local subword state is retained exactly as in the closure model. alpha=16, beta=64. Only internal next-character events plus END are scored.

Nested normalized corrections:
HAND: context = Davis hand + previous up-to-two raw characters within token.
REGIME5: context = five-state regime + previous up-to-two raw characters.
REGIME6_SPLITB: context = six-state regime + previous up-to-two raw characters.
Each correction shrinks to the frozen parent event probability:
P_child(y|context)=(count(context,y)+lambda*P_parent(y))/(N_context+lambda).
One shared lambda per model, tuned on training-only inner physical-quire folds from {16,64,256,1024,4096,16384,1e9}. 1e9 is an operational off-switch.

Validation:
Entire physical quires are indivisible. Use deterministic hash-sort round-robin five-fold outer and four-fold inner validation over the eligible physical quires. Every fold must be nonempty. Report held-out bits per INTERNAL event and bits per token. Use 5,000 physical-quire sign-flip draws for paired contrasts.

Primary contrasts:
REGIME5 vs PARENT.
HAND vs PARENT.
REGIME5 vs HAND.
REGIME6_SPLITB vs REGIME5.

Interpretation gates:
- MULTI_REGIME_USEFUL_COORDINATE if REGIME5 beats PARENT by >0.01 bits/token-equivalent with z>=2.
- REGIME_BEATS_HAND only if REGIME5 improves over HAND by >0.005 bits/token-equivalent with z>=2.
- B_SPLIT_SUPPORTED only if REGIME6_SPLITB improves over REGIME5 by >0.005 bits/token-equivalent with z>=2.
- Otherwise retain the simpler coordinate.
No claim of multiple languages follows from predictive gain.

f58 external transfer:
After primary outcomes are frozen, build regime-specific within-token transition profiles from ALL eligible training pages, with f58 absent. Score f58r/v without fitting any parameter to them and compare their per-event cross-entropy/JS position against EARLY_A, LATE_A_PHARMA, B_HERBAL, B_STARS, BIO and CELESTIAL. Also compare f58 to the held-out page-loss distribution of LFD hand 3. This is an outlier/localisation test only; a single folio cannot define a learnable register under whole-quire validation.

# NPCOMPL33T MULTI-REGIME AUDIT — RESULTS 2026-09-18

Operationalisation was frozen before outcomes from the thread/RZ ecology. This is an audit of the published regime descriptions, not an exact reproduction of unpublished page-cluster assignments.

Primary learned universe:
33,411 tokens / 167,116 internal continuation+END events after excluding mixed quire-8 pages and Rosettes from the learned register fit. Entire physical quires were held out.

Held-out internal codelength, expressed per token for readability:
- frozen PARENT: 13.34912933 bits/token-equivalent.
- Davis HAND child: 11.17911263.
- REGIME5 = {EARLY_A, LATE_A_PHARMA, merged B_CORE, BIO, CELESTIAL}: 10.10168304.
- REGIME6_SPLITB, splitting B_HERBAL from B_STARS: 12.53082439.

However the effect is highly heterogeneous by physical quire:
- REGIME5 vs PARENT gain +3.24744629 bits/token-equivalent; physical-quire sign-flip null SD 2.46244026; z=1.3296. THE METRIC DOES NOT RESOLVE THIS.
- HAND vs PARENT +2.17001670; null SD 1.40599295; z=1.5411. THE METRIC DOES NOT RESOLVE THIS.
- REGIME5 vs HAND +1.07742959; null SD 2.80476128; z=0.3756. THE METRIC DOES NOT RESOLVE THIS.
- splitting Herbal-B from Stars/Recipes is worse by 2.42914135 bits/token-equivalent, but null SD 2.43410995; z=-1.0200. THE METRIC DOES NOT RESOLVE THIS.

Why the variance is large:
- BIO occupies only physical quire 13, so a true BIO-specific child cannot be learned when that entire quire is held out. Davis hand 2 transfers from Herbal-B and improves BIO by +7.09474 bits/token-equivalent; the BIO register itself necessarily reverts to parent.
- B_STARS occupies essentially one physical quire. When quire 18 is held out, the merged B_CORE state can transfer from Herbal-B and gains +7.60201 bits/token-equivalent, while a separately split B_STARS state has no training support.
- CELESTIAL is fully confounded with Davis hand 4 in this operational corpus; REGIME and HAND make the same local prediction.
- LATE_A_PHARMA is predicted better by hand 1 than by its regime label in the held-out decomposition (-0.93111 bits/token-equivalent for REGIME5 vs HAND).
- EARLY_A has only a small residual difference between regime and hand (+0.02842 bits/token-equivalent).

Frozen primary decision:
The proposed multi-regime ecology is DESCRIPTIVELY REAL but does not clear the preregistered whole-quire predictive gate as an independent grammar/register layer. No evidence supports splitting Herbal-B from Stars/Recipes as a separate learnable grammar state. The result is dominated by manuscript design confounding: several claimed regimes are also single-quire or single-hand production blocks.

## f58 external/outlier replication

f58 is not a learnable register under whole-quire validation: it is one folio in one physical quire and is outside the frozen 34,416-token running-text corpus. It was therefore tested only as an external transition-profile outlier without fitting any parameter to f58.

Using context-weighted within-token JS divergence and the same odd/even-line internal-variation denominator described in the thread, on 211 eligible pages:
RAW:
- f58r rank 1 / 211; ratio 1.53113.
- f58v rank 5 / 211; ratio 1.26409.
ATOMIC compound-EVA:
- f58r rank 1 / 211; ratio 1.58856.
- f58v rank 2 / 211; ratio 1.42529.

This independently reproduces the core f58 claim qualitatively and makes it representation-robust.

In every frame and on both sides, the nearest external pages are celestial:
- f58r -> f72r1.
- f58v -> f72r2.

Group-centroid JS is also smallest for CELESTIAL for both f58r and f58v in raw and atomic frames.
Raw f58r: CELESTIAL 0.06176 versus same-hand-3 centroid about 0.10084.
Raw f58v: CELESTIAL 0.06808 versus same-hand-3 centroid about 0.08069.
Atomic f58r: CELESTIAL 0.07603 versus same-hand-3 centroid about 0.12117.
Atomic f58v: CELESTIAL 0.07976 versus same-hand-3 centroid about 0.09986.

Interpretation:
f58 really is exceptional, but the new result cuts against treating it as a self-contained seventh/sixth language. Its within-token transition ecology is closer to hand-4 celestial material than to the rest of Davis hand 3. That makes f58 a particularly useful natural experiment for TASK/REGISTER versus SCRIBE, not evidence by itself for a distinct language.

Persistent objects:
- public.vms_supergrammar_npcompl_regime_models_v01
- public.vms_supergrammar_npcompl_regime_contrasts_v01
- public.vms_supergrammar_npcompl_regime_local_v01
- public.vms_supergrammar_f58_transfer_v01
- public.vms_supergrammar_f58_outlier_v01

# FULL LABEL PROGRAMME v0.1 — BLINDED PREREGISTRATION 2026-09-18

Purpose:
Test label text from first principles against frozen running-text Super-Grammar 1.0 without using the known f58/f72 observation. The analysis is explicitly blinded with respect to f58 and f72 until all primary label outcomes and model choices are frozen.

Blinding firewall:
- All f58* and f72* pages are quarantined from EVERY primary fit, tuning step, model comparison, nearest-neighbour search and diagnostic ranking.
- Primary scripts may know only that a fixed quarantine set exists; no f58/f72 result is inspected before PRIMARY_COMPLETE is written.
- After primary results are persisted, the quarantine is unblinded exactly once and scored using frozen models only. No post-unblind retuning is permitted.

Corpus:
Primary transcription ZLZI.
STRICT_LABEL = IVTFF locus classes La, Lc, Lf, Ln, Lp, Ls, Lt, Lz.
L0 is a secondary dislocated-word control, not part of STRICT_LABEL.
Lx is excluded as possible external/marginal writing.
Circular C* and radial R* loci are excluded from label training and reserved for a later geometry-aware programme.
RUNNING = P0/P1/Pb/Pc/Pr/Pt paragraph loci.
Primary strict-label census before quarantine: 683 loci / 771 tokens / 45 pages. After removing f72*: 596 tokens / 39 pages.
Physical quire is the indivisible validation block.

Representations:
- RAW EVA.
- COMPOUND_ATOMIC EVA using frozen compounds ckh, cth, cph, cfh, ch, sh, qo.
- Alternate transcription robustness for ZLZB, TTLI, JSLI, VDRB and TTIA where coverage is usable.

Programme A — running-to-label token-grammar transfer:
Train ONLY on RUNNING tokens outside the held-out physical quire. Score held-out STRICT_LABEL tokens using a normalized token-internal model:
- section-conditioned character root;
- within-token suffix hierarchy through order 3;
- Jeffreys root smoothing and hierarchical shrinkage alpha=16;
- 19 frozen hard-zero bigrams where applicable.
No SPACE, line-position, previous-token, SG45 or exact-R64 dynamics are allowed: this test concerns intrinsic token formation only.
For every held-out label token, compare its codelength with running tokens in the same held-out quire matched on section and exact token length. If exact length is unavailable, nearest length +/-1 is allowed and flagged.
Primary statistic = label minus matched-running bits/token. Quire-block sign flips, 5,000 draws.

Programme B — does LABEL require an additional internal grammar?
Nested LABEL correction is trained on STRICT_LABEL tokens in training quires only and shrinks to the frozen running-parent event probability.
Context = section + previous up-to-two internal symbols/atoms.
Lambda grid {16,64,256,1024,4096,16384,1e9}; 1e9 is an exact off-switch.
Score held-out labels only. Primary contrast LABEL_CHILD versus RUNNING_PARENT.
If gain <=0.01 bits/token or |z|<2, generic label-internal grammar is not resolved.
If gain >0.01 and z>=2, LABEL_REGISTER is supported.

Programme C — subtype mediation:
Compare generic LABEL_CHILD with a subtype child keyed by IVTFF label subtype La/Lc/Lf/Ln/Lp/Ls/Lt/Lz.
Same nested normalization and whole-quire validation.
Subtype is promoted only if it adds >0.01 bits/token with z>=2 over generic label.
Because subtype and section/quire are highly confounded, any single-quire subtype is descriptive only.

Programme D — feature-level decomposition against matched running nulls:
Pre-specified features:
- raw/atomic token length;
- initial symbol/atom;
- final symbol/atom;
- final-two morphology;
- q incidence and q->k/t continuation where defined;
- maximal i-run length and terminator distribution;
- hard-zero violation rate;
- order-1, order-2 and order-3 internal codelength;
- exact-token running-frequency bin {0,1,2-4,5-19,20-99,>=100}.
For each feature compare labels to section+length-matched running tokens with quire-preserving permutation/bootstrap. These are decomposition tests, not separate promotion opportunities.

Programme E — vocabulary/repertoire:
Using running text outside each held-out quire only:
- fraction of label tokens/types attested in running;
- running frequency distribution of label tokens;
- depletion/enrichment of globally common running forms;
- exact label repeats across pages/sections.
No semantic meaning is inferred.

Programme F — multi-token label loci:
Only loci containing >=2 tokens are used.
Test whether adjacent token edge states exhibit the frozen running SPACE-junction scaffold relative to within-locus token-order permutations and matched same-length running junctions.
This is exploratory unless >=100 adjacent label pairs survive.

Programme G — cross-transcription robustness:
Repeat the primary transfer and generic label-child tests on ZLZB, TTLI, JSLI, VDRB and TTIA.
ROBUST_LABEL_REGISTER requires positive resolved label-child gain in >=4/5 usable alternates with no resolved contradiction.
ROBUST_SHARED_TOKEN_GRAMMAR requires label-vs-matched-running transfer gap <=0.10 bits/token or unresolved in >=4/5 usable alternates.

Frozen primary decisions:
SHARED_TOKEN_GRAMMAR if Programme A transfer gap is unresolved or <=0.10 bits/token AND Programme B generic label-child gain is unresolved or <=0.01.
LABEL_REGISTER if Programme B gain >0.01 bits/token with z>=2 and cross-transcription robustness passes.
SUBTYPE_REGIMES only if Programme C resolves and at least two distinct subtype families contribute outside single-quire support.
MIXED otherwise.

UNBLIND protocol, only after PRIMARY_COMPLETE:
1. Score quarantined f72 labels under frozen running-parent and frozen label-child models. Establish whether f72 is typical label material or itself exceptional.
2. Score quarantined f58 running tokens under frozen running-parent and label-child models. Compute per-page LABEL_LLR = loss_running_parent - loss_label_child; rank f58r/v among all ordinary running pages scored without retraining.
3. Compare f58 to frozen label subtype centroids in RAW and ATOMIC internal-transition space. No parameter may be fitted to f58 or f72.
4. Ask whether f58 is label-like generally, celestial-label-like specifically, or neither.

## Label programme v0.1 support amendment before outcomes
A section/locus support audit shows that, after the f58/f72 quarantine, strict labels with substantial same-section running-text support are Astronomical (84 label tokens / 271 running), Balneological (115 / 6180) and Pharmaceutical (233 / 2275). Cosmological has only 7 labels, Herbal-A 4. Zodiac has 153 strict labels but ZERO P-type running tokens in the same section.
Therefore:
- PRIMARY_LABEL_REGISTER inference (Programmes A/B) is restricted to sections with >=20 strict-label tokens AND >=100 P-type running tokens: Astronomical, Balneological, Pharmaceutical.
- ALL_LABELS results, including Zodiac, are reported descriptively/transfer-only.
- Zodiac cannot by itself establish a label-versus-running register because section and locus are perfectly confounded there.
No outcome has been computed; thresholds and models are unchanged.

## Label programme v0.1 pre-outcome identifiability correction
The support audit also reveals that a generic LABEL child keyed by section+suffix would be unlearnable under whole-quire validation for Astronomical and Balneological, because each has label support in only one physical quire. That would test section-specific memorisation, not a cross-section label mechanism.

Before any outcome computation, Programme B/C contexts are therefore corrected as follows:
- GENERIC_LABEL child context = START or previous up-to-two internal symbols/atoms, WITHOUT section. It still shrinks event-by-event to the section-conditioned running parent. Thus it can learn a locus-specific residual across sections while preserving the parent's section ecology.
- SUBTYPE child context = label subtype + START/suffix2, also shrinking to the section-conditioned parent.
- A separate SECTION_LABEL diagnostic may be reported but cannot drive promotion.
All other frozen thresholds, quire blocking and blinding rules remain unchanged.

# FULL LABEL PROGRAMME v0.1 — PRIMARY BLINDED RESULTS FROZEN BEFORE UNBLIND

Blinding was maintained: f58* and f72* were absent from all primary fitting, tuning, diagnostics and rankings.

Strict-label corpus after quarantine:
596 label tokens on 39 pages. Primary inferential support, requiring same-section running text, comprises 432 label tokens in Astronomical, Balneological and Pharmaceutical material across only four independent physical quires.

Primary raw EVA:
- running-parent transfer gap label minus matched running = +3.44043 bits/token; null SD 2.45188; z=1.386. THE METRIC DOES NOT RESOLVE THIS.
- generic cross-section LABEL child gain over running parent = +4.46108 bits/token; null SD 2.43568; z=1.817. THE METRIC DOES NOT RESOLVE THIS.
- every one of the four primary physical quires has positive generic-label gain and every fold selects strongest tested correction lambda=16.
- subtype child relative to generic label child = -3.65869 bits/token; null SD 2.30674; z=-1.592. THE METRIC DOES NOT RESOLVE THIS; subtype does not generalise under whole-quire holdout.

Compound-atomic:
- matched transfer gap +2.78191 bits/token; null SD 2.01769; z=1.365: THE METRIC DOES NOT RESOLVE THIS.
- generic LABEL child gain +3.88115; null SD 2.11471; z=1.830: THE METRIC DOES NOT RESOLVE THIS.
- subtype increment -3.05381; null SD 1.97368; z=-1.545: THE METRIC DOES NOT RESOLVE THIS.

All-label descriptive result:
Including Zodiac and other sections that lack same-section running support, generic LABEL child gain resolves in raw (+5.48176, SD 2.23916, z=2.447) and atomic (+4.87040, SD 1.93413, z=2.506), but this cannot establish locus causality because Zodiac is section-locus confounded.

Cross-transcription:
The generic label correction is positive and very similar in every alternate layer:
ZLZB +4.455 (z=1.82), TTLI +4.425 (1.84), JSLI +4.482 (1.67), VDRB +4.400 (1.87), TTIA +4.459 (1.84). None individually resolves at z>=2 because the same four physical blocks bound the inference. Matched running-transfer gaps are likewise positive but unresolved in all alternates. Therefore the preregistered ROBUST_LABEL_REGISTER gate fails.

Feature decomposition, same-section + length-matched running controls:
Labels differ strongly in initial and terminal morphology and are depleted in q-bearing forms and long i-runs. Initial/final-two effects survive compound atomisation. q->k/t choice itself is not different once q occurs. These token-level decomposition p-values are anti-conservative with respect to the four-quire clustering and do NOT override the quire-block primary inference.

Vocabulary/repertoire:
- 596 tokens / 485 types.
- only 342/596 = 57.38% of label tokens are attested in out-of-quire running text; 254/596 are unseen.
- only 9/596 = 1.51% belong to the ten most frequent running-text forms.
- zero violations of the 19 frozen hard-zero bigrams.

L0 dislocated-word control:
Applying the strict-label correction to L0 gives raw +1.65196 bits/token with null SD 1.89762, z=0.899; atomic +1.26280 with SD 1.51725, z=0.837. THE METRIC DOES NOT RESOLVE THIS. The strict-label shift is therefore not simply a universal property of isolated/dislocated words.

Multi-token labels:
65 adjacent pairs in 54 loci, below the preregistered >=100-pair confirmatory threshold. Exploratorily, the frozen running SPACE-edge scaffold improves them by +0.38667 bits/pair with null SD 0.12570, z=3.08. This is suggestive that multi-token labels preserve ordinary local junction compatibility, but remains exploratory.

FROZEN PRIMARY DECISIONS BEFORE UNBLIND:
- SHARED_TOKEN_GRAMMAR gate: PASS, because neither the matched transfer gap nor generic label-child increment resolves at physical-quire level.
- LABEL_REGISTER: FAIL preregistered promotion gate.
- SUBTYPE_REGIMES: FAIL.
- Interpretive status: NO RESOLVED SEPARATE LABEL GRAMMAR; there is nevertheless a large, directionally consistent morphology/repertoire shift that the four-block manuscript design cannot distinguish cleanly from section/task ecology.
PRIMARY_COMPLETE_BLIND is now frozen. Only now may f58/f72 be unblinded.

# Formal terminology correction
Replace the assistant-created term CELESTIAL in SG47/SG48 prose with ACZ (Astronomical+Cosmological+Zodiac manuscript ecology). This is a nomenclature correction only; prior numerical results are unchanged.

# NOMENCLATURE CORRECTION + ACZ CROSS-LOCUS REGISTER TRANSFER v0.1 — PREREGISTERED BEFORE OUTCOMES

Nomenclature:
The earlier analysis used the assistant-created convenience term "CELESTIAL". This is NOT an established Voynich category and is now retired from the formal model.
Neutral replacement:
ACZ = Astronomical + Cosmological + Zodiac manuscript ecology, operationally the f67-f73 region used in the earlier experiments.
All future formal statements use ACZ. Earlier numerical results are unchanged; only the label is corrected.

Question:
Does a manuscript-region/token-production state transfer across locus modes, independently of exact token identity and ordinary running-text sequence dynamics?
The confirmatory crossed comparison is ACZ versus Pharmaceutical (PHARMA), because these are the only two available ecologies with STRICT_LABEL and RUNNING support across at least two physical quires. This is a structural region-state test, not a semantic claim about subject matter.

Primary transcription:
ZLZI. Raw EVA and frozen compound-atomic EVA are co-primary.

Locus definitions:
RUN = P0/P1/Pb/Pc/Pr/Pt.
STRICT_LABEL = La/Lc/Lf/Ln/Lp/Ls/Lt/Lz.
Circular/radial loci are secondary ACZ-only transfer probes and do not drive the confirmatory inference.

Region definitions:
ACZ = pages whose frozen section map is Astronomical, Cosmological or Zodiac.
PHARMA = pages whose frozen section map is Pharmaceutical.
No f58 page belongs to training. f72 is quarantined from all model fitting and tuning and is used only after the primary cross-locus result is frozen.

Physical support:
RUN ACZ: Q09/Q10.
LABEL ACZ: Q09/Q10/Q11/Q12.
RUN PHARMA: Q15/Q17.
LABEL PHARMA: Q15/Q17.
Thus RUN->LABEL has six independently held-out target quires (Q09,Q10,Q11,Q12,Q15,Q17).
LABEL->RUN has four target quires (Q09,Q10,Q15,Q17).

Model:
No exact token identity is used.
For each source locus separately, fit a pooled common hierarchical token-internal model on source-locus tokens from ACZ+PHARMA outside the target quire:
- START plus internal next-symbol/atom events including END;
- suffix hierarchy up to order 3;
- Jeffreys root smoothing 0.5;
- alpha=16 hierarchical shrinkage;
- frozen hard-zero legality where applicable.

Then fit ACZ and PHARMA normalized region children over the same event contexts:
P_region(y|context) = (count_region(context,y) + 16 * P_common(y|context)) / (N_region(context) + 16).
The shrinkage 16 is frozen; it is not tuned on target data.

Primary cross-locus transfers:
A. RUN->LABEL: train region children using RUN only; score held-out STRICT_LABEL tokens as ACZ vs PHARMA.
B. LABEL->RUN: train using STRICT_LABEL only, excluding f72; score held-out RUN tokens.

Primary statistic:
TRUE_REGION_ADVANTAGE = loss_wrong_region - loss_true_region, in bits/token.
Positive means region identity is recoverable across the locus change.
Inference is by exact physical-quire sign-flip enumeration because there are only six or four target blocks. Also report token-weighted and equal-quire means.

Length-controlled sensitivity:
Repeat A/B with exact token length supplied as a conditioning variable in every event context. This removes token-length distribution as a source of region discrimination.
The region signal is considered deeper than length only if the sign remains positive in >=5/6 RUN->LABEL target quires in raw and atomic frames.

Same-locus oracle sensitivity:
For each target locus, train the corresponding same-locus region classifier on other quires and compare its TRUE_REGION_ADVANTAGE with the cross-locus model. Report the fraction of same-locus discriminative information recovered cross-locus. This is descriptive and cannot override the block-level primary inference.

Alternate transcription robustness:
Repeat RUN->LABEL and LABEL->RUN in ZLZB, TTLI, JSLI, VDRB and TTIA using identical frozen definitions where the relevant source/target tokens exist. No model or threshold changes are allowed after ZLZI outcomes.

Frozen decision:
ACZ_REGION_TRANSFERS_ACROSS_LOCUS if:
- RUN->LABEL TRUE_REGION_ADVANTAGE is positive in >=5/6 physical quires in BOTH raw and atomic;
- exact sign-flip p_upper <= .05 in BOTH;
- length-controlled RUN->LABEL remains positive in >=5/6 quires in BOTH frames;
- LABEL->RUN is positive in >=3/4 physical quires in BOTH frames (four blocks cannot attain two-sided p<.05 by exact sign flips).
Otherwise HOLD_MIXED.
This licenses only a cross-locus manuscript-region/register coordinate, not semantic subject matter or causal "task".

Frozen post-primary probes:
1. f72: with f72 still absent from all training, score its LABEL and circular tokens under RUN-trained ACZ-vs-PHARMA classifiers.
2. f58: score f58 running tokens under LABEL-trained ACZ-vs-PHARMA classifiers with f72 excluded; also score under RUN-trained classifiers. No fitting to f58 is permitted.
3. Rank f58 among ordinary running pages by ACZ-vs-PHARMA log-likelihood ratio.
If f58 prefers ACZ under BOTH label-trained and running-trained models, record ACZ_AFFINITY_CROSS_LOCUS; if only one source locus does, HOLD.

Interpretive limit:
Even a positive result cannot distinguish region/task/content from other variables locked to the f67-f73 production ecology (including hand and codicological unit). It tests whether the state crosses locus, not its historical cause.

# ACZ CROSS-LOCUS DIRECT TRANSFER v0.1 — ACCEPTED NEGATIVE RESULT

The strongest version of the upstream-region hypothesis fails.

ZLZI, raw:
RUN->LABEL true-region advantage = -0.21494 bits/token; exact six-quire sign-flip p_upper=.796875.
LABEL->RUN = -0.13812; four-quire p_upper=.6875.
Length-controlled RUN->LABEL = -0.08562; p_upper=.6875.

ZLZI, compound-atomic:
RUN->LABEL = -0.20135; p_upper=.78125.
LABEL->RUN = -0.38097; p_upper=.875.
Length-controlled RUN->LABEL = -0.15390; p_upper=.890625.

The failure is not transcription-specific. ZLZB, TTLI, JSLI, VDRB and TTIA all reproduce NEGATIVE RUN->LABEL and LABEL->RUN true-region advantages in raw and atomic frames. No alternate layer approaches the preregistered transfer gate.

Therefore an absolute ACZ-vs-PHARMA token profile learned in RUN does NOT directly classify LABEL tokens, and vice versa. The strongest claim that region profile simply transfers unchanged across locus is rejected.

Important model diagnosis:
Same-locus region classifiers are generally positive while cross-locus classifiers become negative. Thus the negative result is not merely absence of ACZ/PHARMA differentiation; locus changes the baseline token distribution enough that absolute profile transfer is inappropriate.

# ACZ x LOCUS FACTORISATION v0.2 — PREREGISTERED AFTER v0.1, BEFORE v0.2 OUTCOMES

Motivation:
The proposed hierarchy was REGION/TASK -> shared token grammar -> LOCUS modulation. Such a hierarchy does NOT require absolute RUN and LABEL profiles to be identical. It predicts that REGION and LOCUS effects should be approximately FACTORABLE: after modelling the generic locus shift, a common region effect should improve both loci, with little residual REGION x LOCUS interaction.

Universe:
Same quarantined ZLZI ACZ-vs-PHARMA RUN+STRICT_LABEL corpus and six physical quires Q09,Q10,Q11,Q12,Q15,Q17. f58/f72 remain excluded until v0.2 model choice is frozen. Raw and compound-atomic are co-primary.

Event model:
Token-internal START + continuation + END events, suffix context up to order 3, no exact token identity and no running sequence dynamics.

For each held-out physical quire, fit on the other five:
P0(y|context): pooled common distribution.
P_L(y|context,locus): locus child RUN/LABEL, Dirichlet-shrunk to P0.
P_R(y|context,region): region child ACZ/PHARMA, shrunk to P0.
ADDITIVE model:
P_ADD(y|ctx,locus,region) proportional to P_L(y|ctx,locus) * P_R(y|ctx,region) / P0(y|ctx), then renormalized.
INTERACTION model:
P_INT(y|ctx,locus,region) = full locus x region counts shrunk to P_ADD.

All child shrinkages use lambda=64, frozen before outcomes. A sensitivity grid lambda={16,256} is reported but cannot change the primary decision.
Frozen hard-zero legality is applied in raw EVA.

Primary contrasts, held-out codelength:
1. ADDITIVE vs LOCUS_ONLY: does REGION add after generic locus?
2. ADDITIVE vs REGION_ONLY: does LOCUS add after region?
3. INTERACTION vs ADDITIVE: is a locus-specific region interaction still required?
Exact six-quire sign-flip inference; report token-weighted and equal-quire effects.

Length-controlled sensitivity:
Repeat with exact token length supplied in every event context.

Decision:
FACTORABLE_CROSS_LOCUS_REGION if in BOTH raw and atomic:
- ADDITIVE beats LOCUS_ONLY by >0.01 bits/token with exact p_upper<=.05;
- INTERACTION does not beat ADDITIVE by >0.01 with p_upper<=.05;
and the region increment remains positive under exact-length conditioning in both frames.
LOCUS_SPECIFIC_REGION if INTERACTION beats ADDITIVE by >0.01 with p_upper<=.05 in both frames.
NO_REGION_AFTER_LOCUS if ADDITIVE vs LOCUS_ONLY is nonpositive in both frames.
MIXED otherwise.

Only after the v0.2 primary decision is frozen:
- score f72 label/circular material under the factorised model;
- score f58 under the factorised region factor and rank it among running pages.
No post-unblind retuning.

# ACZ x LOCUS FACTORISATION v0.2 — PRIMARY RESULT FROZEN

The direct-transfer failure is not the whole story, but the factorised hypothesis is only conditionally supported.

STANDARD raw:
LOCUS_ONLY loss 12.63426 bits/token.
REGION_ONLY 12.75978.
ADDITIVE 12.98410.
INTERACTION 12.78974.
REGION_AFTER_LOCUS = -0.34984 bits/token; exact p_upper=.953125. Region hurts once generic locus is modelled.
INTERACTION_AFTER_ADDITIVE = +0.19437; p_upper=.0625.

STANDARD atomic:
LOCUS_ONLY 12.62595.
REGION_ONLY 12.75693.
ADDITIVE 12.98692.
INTERACTION 12.78859.
REGION_AFTER_LOCUS = -0.36097; p_upper=.953125.
INTERACTION_AFTER_ADDITIVE = +0.19834; p_upper=.046875.

Thus the preregistered FACTORABLE_CROSS_LOCUS_REGION decision FAILS in the standard model.

However, conditioning on the OBSERVED EXACT TOKEN LENGTH reverses the result:

LENGTH-CONTROLLED raw:
REGION_AFTER_LOCUS = +0.10982 bits/token; exact p_upper=.015625; all 6/6 physical quires positive.
LOCUS_AFTER_REGION = +0.29375; p=.015625; all 6/6 positive.
INTERACTION_AFTER_ADDITIVE = +0.13146; p=.03125, with 5/6 positive.

LENGTH-CONTROLLED atomic:
REGION_AFTER_LOCUS = +0.08581; p=.03125; 5/6 positive (Q09 slight negative).
LOCUS_AFTER_REGION = +0.26481; p=.015625; all 6/6 positive.
INTERACTION_AFTER_ADDITIVE = +0.10298; p=.140625.

Frozen v0.2 decision: MIXED.
Interpretation bound:
There is no support for a simple invariant region profile. Conditional on token length, both LOCUS and REGION carry cross-validated information, but a residual interaction remains at least in raw EVA. This suggests token length/termination may be a major locus-by-region coupling variable rather than a nuisance to ignore.

# ACZ x LOCUS v0.3 — LENGTH/TERMINATION DECOMPOSITION + ROBUSTNESS, PREREGISTERED BEFORE OUTCOMES

Questions:
1. Is the standard-model failure driven principally by token LENGTH / END probability?
2. Does the positive length-conditioned REGION_AFTER_LOCUS effect replicate across alternate transcriptions/word divisions?
3. After freezing v0.2, where do f58/f72 fall under the factorised ACZ-vs-PHARMA coordinate?

A. LENGTH model:
Predict token length itself using normalized categorical models:
COMMON; LOCUS_ONLY; REGION_ONLY; ADDITIVE locus+region; INTERACTION locus x region.
Training and testing use the same six physical-quire outer folds. Length categories are exact observed raw/atomic lengths, with an >=12 overflow bin if necessary.
Primary contrasts: REGION_AFTER_LOCUS and INTERACTION_AFTER_ADDITIVE.

B. NON-END internal model:
Repeat v0.2 STANDARD but score START + symbol/atom continuation events while EXCLUDING END. This removes direct termination/length prediction without supplying length as a covariate.
If REGION_AFTER_LOCUS becomes positive while the full standard model is negative, termination is implicated.

C. Alternate transcription replication of the LENGTH-CONTROLLED factor model:
ZLZB, TTLI, JSLI, VDRB, TTIA; same lambda=64 and physical-quire definitions.
Record REGION_AFTER_LOCUS and INTERACTION_AFTER_ADDITIVE.
ROBUST_LENGTH_CONDITIONAL_REGION if REGION_AFTER_LOCUS is positive in >=4/5 alternates in raw and atomic with no resolved negative contradiction.

D. Frozen probes:
After A-C model definitions are fixed, score f72 LABEL and circular material and f58 running material under v0.2 ADDITIVE ACZ-vs-PHARMA predictions. No fitting to f58/f72.
For f58 report standard and length-controlled ACZ advantage and page rank among all running pages.
For f72 report ACZ advantage by locus.
No semantic interpretation.

## ACZ x LOCUS v0.3 implementation audit — INVALID RUN, discarded before interpretation

The first v0.3 decomposition implementation accidentally allowed ACZ CIRCULAR/RADIAL rows into the A-C fitting universe because the generic row loader admitted all recognized loci. This violated the preregistered "same six RUN+STRICT_LABEL physical quires as v0.2" requirement and explains why the supposedly identical length-conditioned model failed to reproduce v0.2 exactly.

ALL first-run v0.3 A/B/C outcomes, including alternate-transcription numbers, are INVALID and discarded.

Correction before accepted rerun:
- A LENGTH, B NON-END and C alternate length-conditioned replication are restricted strictly to RUN + STRICT_LABEL, identical to v0.2.
- f72 circular material remains only in the separately preregistered frozen probe D.
- the corrected v0.3 length-conditioned ZLZI model MUST reproduce the frozen v0.2 length-conditioned codelengths and contrasts exactly before any new A/B/C result is accepted.
No threshold/model/hyperparameter changes.

# FULL LABEL UNBLIND + ACZ CROSS-LOCUS REGISTER PROGRAM — FINAL RESULT 2026-09-18

Terminology:
The assistant-created term CELESTIAL is retired. ACZ means only the Astronomical+Cosmological+Zodiac manuscript ecology used operationally in these tests. No semantic inference is intended.

## Blinded label result, now durably recorded
Strict labels do not resolve as a separate grammar under whole-quire inference. They obey the shared hard legality and broad internal grammar while shifting morphology/repertoire. f72, held out from all primary fitting, is strongly label-mode under the frozen label correction. f58 is NOT generically label-like.

## ACZ direct cross-locus transfer
Absolute ACZ-vs-PHARMA profiles do not transfer from RUN to LABEL or vice versa. ZLZI raw RUN->LABEL = -0.21494 bits/token and atomic = -0.20135. All five alternate transcription layers reproduce the same negative direction. REJECT absolute invariant-profile transfer.

## Factorial ACZ x LOCUS model
Without conditioning on token length:
REGION_AFTER_LOCUS is negative in raw (-0.34984) and atomic (-0.36097). A simple strict hierarchy REGION -> LOCUS therefore fails.

When exact token length is supplied:
raw REGION_AFTER_LOCUS = +0.109822 bits/token, exact six-quire p=.015625, 6/6 blocks positive.
atomic = +0.0858074, p=.03125, 5/6 positive.
This length-conditioned region component replicates positive in 5/5 alternate transcriptions in BOTH raw and atomic frames. Raw exact p<=.05 in 5/5 alternates; atomic in 4/5. No resolved negative contradiction.

But REGION x LOCUS interaction remains:
raw interaction gain +0.131457, p=.03125.
atomic +0.102979, p=.140625.
Across alternates the interaction is positive in every raw/atomic layer and often resolves.

## What length is doing
Token length itself is NOT a strong ACZ-after-locus discriminator:
raw length REGION_AFTER_LOCUS +0.00330, p=.171875.
atomic -0.00369, p=.78125.
Removing END/termination does NOT restore factorability: the non-END model still has negative REGION_AFTER_LOCUS (-0.33973 raw; -0.35190 atomic).
Therefore the effect is not simply a different length distribution or termination rule. Exact length acts as a stratifying/template-state coordinate: within a given token length, regional internal morphology becomes more transferable across locus. Marginalising over lengths produces a Simpson-like reversal.

## Frozen probes
f72, absent from training:
factorised ACZ advantage remains positive for labels in all raw/atomic standard and length-controlled models. Circular f72 has strong positive ACZ advantage in standard models and only small/near-zero length-controlled advantage.

f58:
Under the factorised model, f58r has positive ACZ-vs-PHARMA advantage in every frame:
raw +0.8786 standard / +0.5214 length-controlled;
atomic +0.8849 / +0.5246.
It ranks 15/206 raw and 10/206 atomic running pages by standard ACZ affinity, and 17/206 and 16/206 length-controlled.
f58v is positive but weaker: +0.5531/+0.3068 raw and +0.5089/+0.3374 atomic; ranks roughly 43-54/206.

Source-specific frozen probes sharpen this:
f58r prefers ACZ under BOTH RUN-trained and LABEL-trained models in raw/atomic and standard/length-controlled variants.
f58v prefers ACZ under both source loci in standard models, but its length-controlled RUN-trained score turns slightly PHARMA-like while LABEL-trained remains ACZ-like.
Thus f58r is a robust ACZ-affine running page; f58v is weaker/mixed.

## Final structural conclusion
The evidence does NOT support:
REGION/TASK state -> unchanged token profile -> LOCUS.

It supports a more coupled model:
SHARED TOKEN GRAMMAR
+ TOKEN-LENGTH/TEMPLATE STATE
+ REGIONAL/REGISTER STATE
+ LOCUS STATE
+ REGION x LOCUS MODULATION.

A region-associated component of internal morphology is real and cross-locus once token-length/template state is controlled, but locus modifies that regional component. ACZ is therefore a useful cross-locus coordinate, not an independent grammar and not a semantic subject label.

This refines the f58/f72 result:
f58 is not label-like. f58r in particular is ACZ-affine, and that affinity can be learned from label as well as running material. The observed f58-f72 proximity is one manifestation of a regional/register component shared across loci, superimposed on locus-specific morphology.

# SG51 FOLLOW-UP: LENGTH-AS-TEMPLATE PROXY v0.1 — PREREGISTERED BEFORE OUTCOMES

Motivation:
Earlier programmes already established that token length is a real surface coordinate (SG35), a powerful confound for near-copy/ED1 effects (SG30), and part of the CORE SG45 previous-token morphology state. The new ACZ x LOCUS result is different: ACZ regional information transfers across locus only after the CURRENT token's exact final length is supplied. Test whether exact length is merely an observable proxy for a more mechanistic token-template/family coordinate already latent in the Super-Grammar.

Scope:
Same ACZ-vs-Pharmaceutical RUN+STRICT_LABEL universe, six physical quires Q09,Q10,Q11,Q12,Q15,Q17, same f58/f72 quarantine, same whole-quire CV, same raw and compound-atomic co-primary frames as SG51. No sequential running-text information or exact token identity is used.

Important interpretation:
All template coordinates below are retrospective token descriptors, just as exact final length is. They are diagnostic explanatory coordinates, not a claim that a historical scribe knew the final descriptor before writing.

Frozen event model:
Same factor model as ACZ x LOCUS v0.2:
COMMON -> LOCUS child; COMMON -> REGION child; multiplicative ADDITIVE combination; LOCUS x REGION INTERACTION child.
Jeffreys root; alpha=16 suffix hierarchy; lambda=64 factor shrinkage; frozen raw hard-zero legality.
Score START + internal continuation + END.

Candidate template strata, computed without exact total length:

T0 NONE:
No token-level stratum. Reproduces SG51 standard model.

T1 TERMINAL2:
Final two raw symbols / final two compound-atoms, with shorter tokens represented exactly. Tests terminal morphology alone.

T2 EDGE_GEOMETRY:
First symbol/atom + final two symbols/atoms. This is the current-token analogue of the SG45 edge coordinate, explicitly WITHOUT length.

T3 V23_AFFIX:
Longest matching v2.3 supported prefix (length 1-4, support>=50) + longest matching v2.3 supported suffix (length 1-4, support>=50), using the frozen authoritative rulebook. Ties choose longer then lexicographic. No total length or exact core identity.

T4 SLOT_SKELETON:
Map units to coarse established morphology classes and collapse adjacent identical classes:
Q={q,qo}; G={k,t,p,f,ckh,cth,cph,cfh}; B={ch,sh}; I={i}; E={e}; D={d}; A={a,o}; T={y,m,n,r,l,s}; C={c,h}; X=other.
The resulting class sequence is the template skeleton; repeated runs collapse, so exact token length is not encoded by run count.

T5 FAMILY_TEMPLATE:
(V23_AFFIX + SLOT_SKELETON of the residual core after removing the chosen prefix and suffix). This is the strongest preregistered family/template proxy and still excludes total length and exact core spelling.

Benchmark:
LENGTH = exact raw/atomic token length, reproducing SG51 length-conditioned result.

Nested absorption tests:
For each T1-T5 also fit TEMPLATE_PLUS_LENGTH = (template coordinate, exact length).

Primary outcome for each stratum:
REGION_AFTER_LOCUS = LOCUS_ONLY loss - ADDITIVE loss, exact six-quire sign-flip inference.
INTERACTION_AFTER_ADDITIVE also reported.

Template replacement gate:
A structural template REPLACES_LENGTH if, in BOTH raw and atomic:
1. REGION_AFTER_LOCUS >0.01 bits/token with exact p_upper<=.05;
2. positive in >=5/6 physical quires;
3. its ADDITIVE held-out loss is within 0.02 bits/token of LENGTH ADDITIVE, OR the paired TEMPLATE-vs-LENGTH difference is unresolved;
4. adding exact length to the template improves ADDITIVE held-out loss by <=0.01 bits/token OR THE METRIC DOES NOT RESOLVE THIS.

PARTIAL_PROXY if conditions 1-2 pass but length still adds resolved >0.01.
FAIL if conditions 1-2 fail.

Reciprocal test:
Add each template coordinate to LENGTH. If TEMPLATE_PLUS_LENGTH beats LENGTH by >0.01 bits/token with exact p<=.05, that template contains structure beyond length.

Complexity audit:
For every stratum/frame report number of observed classes, median tokens/class, singleton-token mass, and class mutual information with exact length. High-cardinality/singleton coordinates cannot be interpreted as mechanistic even if predictive.

Control:
HASH_CARDINALITY control: deterministically hash exact token spelling into the same number of classes as FAMILY_TEMPLATE, using a frozen SHA256 map. This is NOT a meaningful template; it tests whether arbitrary token partitioning of comparable cardinality can create the ACZ effect. Promotion requires FAMILY_TEMPLATE to beat or materially outperform this control.

Cross-transcription:
Only if a structural template reaches REPLACES_LENGTH or PARTIAL_PROXY in ZLZI, repeat that template and LENGTH benchmark on ZLZB, TTLI, JSLI, VDRB, TTIA before promotion.

Decision:
- LENGTH_IS_TEMPLATE_PROXY if at least one non-hash structural template meets REPLACES_LENGTH in raw+atomic and survives alternate transcription.
- LENGTH_PARTLY_PROXY if best template is PARTIAL_PROXY or reciprocal length still adds information.
- LENGTH_REMAINS_INDEPENDENT_STATE if no structural template passes while LENGTH retains SG51 replication.
No semantic interpretation follows.

## SG51 length-proxy v0.1 deterministic implementation clarification — before outcomes
- V23_AFFIX candidate sets are extracted directly from the frozen v2.3 rulebook: kind=prefix/suffix, support>=50, pattern raw length 1-4. No outcome-based pruning or modern promotion status is used.
- In RAW, longest match is by raw-symbol length then lexicographic. In ATOMIC, rule patterns are atomised with the frozen compound map and longest match is by atom count, then raw-symbol length, then lexicographic.
- FAMILY_TEMPLATE removes the chosen prefix first; the suffix is then chosen as the longest supported suffix that fits wholly within the remaining units (no prefix/suffix overlap). Residual core may be empty.
- SLOT_SKELETON is computed after atomisation in ATOMIC and directly on raw symbols/recognized raw compounds in RAW; adjacent identical classes collapse.
- HASH_CARDINALITY maps each exact token spelling by SHA256(token + "|length-proxy-control-v01") modulo K, where K equals the number of FAMILY_TEMPLATE classes observed in that outer-fold training set. The control therefore matches nominal class count but has no morphological design.
- All class dictionaries are learned/defined from outer-training data only where training support is required; unseen test strata back off to the unstratified parent rather than being assigned from test frequencies.

## SG51 length-proxy v0.1 ZLZI complexity gate — candidate selection before alternate-transcription outcomes
ZLZI has now been scored under the preregistered coordinates.

Complexity audit consequence:
- V23_AFFIX and FAMILY_TEMPLATE are too high-cardinality for mechanistic interpretation here (about 1,370-1,430 classes on 3,243 tokens; median 1 token/class; singleton token mass ~30-33%). FAMILY_TEMPLATE also carries essentially the full entropy of exact length (MI with length equals the observed length entropy), so it behaves as an alias rather than an explanatory compression.
- HASH_CARDINALITY also produces a positive REGION_AFTER_LOCUS sign, showing that arbitrary exact-token partitioning can manufacture part of the stratification effect; however its held-out ADDITIVE codelength is dramatically worse than the structural coordinates.
- TERMINAL2, EDGE_GEOMETRY and SLOT_SKELETON satisfy the ZLZI replacement gate with materially lower singleton mass and are the only candidates advanced to alternate-transcription replication.
No alternate-transcription outcome has yet been computed for these candidates.

Advance:
1. TERMINAL2 + TERMINAL2_PLUS_LENGTH.
2. EDGE_GEOMETRY + EDGE_GEOMETRY_PLUS_LENGTH.
3. SLOT_SKELETON + SLOT_SKELETON_PLUS_LENGTH.
4. LENGTH benchmark.
Promotion requires directionally stable REGION_AFTER_LOCUS and no recurring resolved benefit from adding exact LENGTH.

# SG51 length-proxy v0.1 — ZLZI + alternate-transcription accepted result, frozen before minimal localisation

The preregistered result is decisive at the level of necessity: exact token length is NOT required to reveal the cross-locus ACZ regional component.

ZLZI:
LENGTH benchmark REGION_AFTER_LOCUS:
- raw +0.109822 bits/token, p=.015625, 6/6 blocks positive.
- atomic +0.085807, p=.03125, 5/6.

Low-complexity structural replacements:
TERMINAL2:
- raw +0.137624, p=.015625, 6/6.
- atomic +0.110021, p=.015625, 6/6.
EDGE_GEOMETRY = first unit + final two:
- raw +0.491887, p=.015625, 6/6.
- atomic +0.445535, p=.015625, 6/6.
SLOT_SKELETON:
- raw +0.449119, p=.015625, 6/6.
- atomic +0.425997, p=.015625, 6/6.

Complexity:
TERMINAL2 is especially parsimonious: 104 raw / 123 atomic classes, median 2 tokens/class, singleton token mass about 1-1.4%, MI with exact length only 0.514/0.535 bits versus exact-length entropy about 2.87/2.73 bits.
EDGE_GEOMETRY has 349/440 classes and MI(length) 1.25/1.33 bits.
SLOT_SKELETON has 619 classes, singleton mass 11.9%, MI(length) 2.18/2.15 bits.

V23_AFFIX and FAMILY_TEMPLATE are not accepted as mechanistic explanations despite predictive results: ~1,400 classes, median one token/class, ~30-33% singleton mass; FAMILY_TEMPLATE essentially determines exact length and is therefore an alias.

HASH_CARDINALITY control:
Arbitrary exact-token hashing also creates a positive REGION_AFTER_LOCUS sign, showing that region-gain alone is unsafe under high-cardinality partitioning. But HASH has catastrophically worse held-out ADDITIVE loss (14.41 raw / 13.71 atomic) than structural candidates; FAMILY_TEMPLATE beats HASH on all 6 blocks by +2.924 bits/token raw and +2.373 atomic. The low-cardinality TERMINAL2 result is not explained by nominal class count.

Length nested tests:
For TERMINAL2, EDGE_GEOMETRY and SLOT_SKELETON, adding exact LENGTH does not improve held-out ADDITIVE prediction in either frame; it worsens it, consistent with needless sparse over-stratification. This does NOT prove length has zero latent information; it establishes that exact length is unnecessary once these structural coordinates are supplied.

Alternate transcription replication:
Across ZLZB, TTLI, JSLI, VDRB and TTIA, for each of TERMINAL2, EDGE_GEOMETRY and SLOT_SKELETON:
- REGION_AFTER_LOCUS is positive and exact-sign-flip resolved in 10/10 raw+atomic layer/frame tests.
- every test has 5/6 or 6/6 physical blocks positive.
- exact LENGTH adds positive held-out value in 0/10 tests for each candidate.
- each structural candidate beats the LENGTH benchmark in held-out ADDITIVE codelength in 10/10 tests.
Thus the structural replacement is transcription/word-division robust.

Frozen v0.1 decision:
LENGTH_IS_TEMPLATE_PROXY, with the strongest parsimonious current explanation being terminal morphology (final-two units). SLOT_SKELETON independently supports a broader token-template interpretation.
Bound:
Because TERMINAL2 and EDGE_GEOMETRY are retrospective descriptors that include observed token endings, this result says what STRUCTURAL STRATUM length was proxying; it does not claim a prospective historical writing algorithm.

# SG52 MINIMAL TEMPLATE LOCALISATION v0.1 — PREREGISTERED BEFORE OUTCOMES

Purpose:
Determine whether the replacement is specifically terminal morphology or a more general early/whole-token family state, while reducing descriptor information.

Same ZLZI ACZ-vs-PHARMA RUN+LABEL corpus, six whole-quire folds, raw/atomic, factor model and exact sign inference.

Coordinates:
- INITIAL1 = first raw symbol / compound atom.
- INITIAL2 = first two symbols/atoms.
- TERMINAL1 = final symbol/atom.
- TERMINAL2 = frozen successful benchmark.
- INITIAL1_TERMINAL1.
- TERMINAL2_CLASS = coarse morphology classes of final two units using Q/G/B/I/E/D/A/T/C/X map, retaining order but not exact glyph identity.
- SLOT_PREFIX2_CLASS = coarse classes of first two units.
No total length.

For each coordinate also fit +LENGTH as a nested diagnostic.

Decision:
MINIMAL_TERMINAL1 if TERMINAL1 passes raw+atomic replacement gate.
MINIMAL_TERMINAL2 if TERMINAL1 fails but TERMINAL2 and TERMINAL2_CLASS pass.
GENERAL_EDGE_FAMILY if INITIAL2 or INITIAL1_TERMINAL1 pass comparably and terminal-only is not uniquely favoured.
TERMINAL_IDENTITY_SPECIFIC if exact TERMINAL2 passes but TERMINAL2_CLASS fails.
Alternate transcription is required for any newly minimal coordinate before SG52 promotion.

# SG52 minimal template localisation — ZLZI result, frozen before alternate transcription

The ZLZI localisation is not terminal-only.

RAW:
INITIAL1: REGION_AFTER_LOCUS +0.2316 bits/token, exact p=.015625, 6/6 blocks positive; 16 classes, singleton mass 0.1%, MI(length)=0.274 bits.
INITIAL2: +0.4127, p=.015625, 6/6.
TERMINAL1: -0.0452, p=.53125; fails.
TERMINAL2 exact: +0.1376, p=.015625, 6/6.
INITIAL1+TERMINAL1: +0.5010, p=.015625, 6/6.
TERMINAL2_CLASS: +0.0280, p=.265625; fails.
SLOT_PREFIX2_CLASS: +0.3700, p=.015625, 6/6.

ATOMIC:
INITIAL1: +0.2645, p=.015625, 6/6; 23 classes, singleton mass 0.1%, MI(length)=0.288.
INITIAL2: +0.4604, p=.015625, 6/6.
TERMINAL1: -0.0641, p=.640625; fails.
TERMINAL2 exact: +0.1100, p=.015625, 6/6.
INITIAL1+TERMINAL1: +0.4794, p=.015625, 6/6.
TERMINAL2_CLASS: -0.0028, p=.53125; fails.
SLOT_PREFIX2_CLASS: +0.3887, p=.015625, 6/6.

Adding exact LENGTH to INITIAL1 worsens held-out ADDITIVE loss in both frames.

Frozen localisation decision:
GENERAL_EDGE_FAMILY, with INITIAL1 as the newly minimal successful coordinate. Exact second-terminal identity carries additional information, but coarse terminal morphology does not; the effect is therefore not simply a terminal-category/length phenomenon.
Advance INITIAL1 and INITIAL1_PLUS_LENGTH to the five alternate transcription/word-division layers before promotion. No alternate outcome has yet been computed.

# SG51 LENGTH-PROXY IMPLEMENTATION AUDIT — ORACLE RESULT QUARANTINED BEFORE LEAK-SAFE OUTCOMES

The first structural-replacement implementation correctly followed the preregistered retrospective-descriptor wording, but its held-out codelength comparison is NOT suitable for mechanistic promotion: descriptor-defining units were also among the events being predicted. Examples:
- INITIAL1 conditioned the START event on the observed initial itself;
- TERMINAL2 conditioned its own terminal events on their observed identity;
- EDGE/SLOT descriptors similarly expose parts of the target token.
Exact LENGTH does not expose glyph identity in the same way.
Therefore the v0.1/v0.1-minimal structural results are retained only as an ORACLE LOCALISATION DIAGNOSTIC. The earlier LENGTH_IS_TEMPLATE_PROXY promotion decision is QUARANTINED and not accepted as final mechanism evidence.
No previously reported numeric result is deleted.

# SG52 LEAK-SAFE TEMPLATE MEDIATION v0.2 — PREREGISTERED BEFORE OUTCOMES

Goal:
Test the same hypothesis on IDENTICAL target events that do not contain any descriptor-defining units.

Common target:
For every token of raw/atomic length >=5, score only MIDDLE unit events at positions 3 through L-2 inclusive in one-based notation: zero-based indices 2..L-3.
Thus the first two units, final two units and END are NEVER scored.
All compared models score exactly the same target events.
Tokens shorter than 5 contribute no target event.

Frozen descriptors:
NONE.
LENGTH = exact total token length.
INITIAL1.
INITIAL2.
TERMINAL1.
TERMINAL2.
INITIAL1_TERMINAL1.
TERMINAL2_CLASS.
SLOT_PREFIX2_CLASS.
All descriptors are observational strata; INITIAL1/INITIAL2 are also prospectively available before any scored target event.

Model:
Same pooled suffix hierarchy + LOCUS/REGION additive factor + interaction as SG51, lambda=64, alpha=16, raw hard-zero legality. Training counts and testing loss include ONLY the common MIDDLE target events. Prefix context for each scored event contains the actually preceding units, as in the running grammar.

Primary question:
Can a non-length structural stratum reproduce the LENGTH-conditioned REGION_AFTER_LOCUS effect on leak-safe middle events?

Replacement gate:
In BOTH raw and atomic:
1. candidate REGION_AFTER_LOCUS >0.01 bits/scored-token-equivalent, exact six-quire p_upper<=.05, >=5/6 positive blocks;
2. candidate REGION_AFTER_LOCUS is at least 75% of LENGTH's positive effect OR candidate ADDITIVE loss is no worse than LENGTH by >0.02 bits/scored-event;
3. candidate+LENGTH does not yield a resolved >0.01 improvement on the SAME middle events.
INITIAL1/INITIAL2 passing is stronger evidence than terminal descriptors because it is prospective relative to the scored events.

If INITIAL1 or INITIAL2 passes, classify LENGTH_PROXY_FOR_EARLY_FAMILY_STATE.
If only terminal descriptors pass, classify LENGTH_PROXY_FOR_RETROSPECTIVE_FAMILY_STATE.
If none pass while LENGTH passes, classify LENGTH_REMAINS_DISTINCT_TEMPLATE_STATE.

Alternate transcription:
Repeat any leak-safe passing candidate and LENGTH on ZLZB/TTLI/JSLI/VDRB/TTIA before promotion.

Additional sanity:
Run LENGTH on the same common middle targets. If LENGTH itself no longer has positive cross-locus REGION_AFTER_LOCUS, the earlier length-conditioned result was driven by excluded edge/termination events and the proxy question is closed as EDGE/TERMINATION_LOCAL rather than a general internal-template state.

# SG52 LEAK-SAFE TEMPLATE MEDIATION v0.2 — ACCEPTED RESULT

The leak-safe common-middle test overturns the oracle interpretation.

Common target:
Only positions 3..L-2 (one-based), excluding first two units, final two units and END from BOTH training counts and held-out loss.

Exact LENGTH itself no longer carries ACZ REGION_AFTER_LOCUS information:
RAW: +0.00073 bits/middle-event, exact p=.50; only 1/6 blocks positive.
ATOMIC: -0.00565, p=.984375; only 1/6 positive.
Thus the prerequisite for a general internal-template proxy fails.

No structural descriptor restores a resolved regional effect on the common middle targets:
- INITIAL1 raw -0.0114 / atomic -0.0129;
- INITIAL2 -0.0112 / -0.0116;
- TERMINAL1 -0.0022 / -0.0023;
- TERMINAL2 +0.0033 p=.172 / +0.00094 p=.313;
- INITIAL1+TERMINAL1 +0.00031 / +0.00128;
- TERMINAL2_CLASS +0.00258 / +0.00085;
- SLOT_PREFIX2_CLASS negative in both.
THE METRIC DOES NOT RESOLVE any positive middle-token regional component.

Frozen leak-safe decision:
EDGE_TERMINATION_LOCAL.
The earlier SG51 exact-length-conditioned ACZ effect is not evidence that LENGTH proxies a latent whole-token internal template. It is generated by events among the first-two / final-two / END zones excluded from the common-middle test.
The earlier structural v0.1 results remain useful oracle localisation but are not mechanistic promotion evidence.
SG51 must be revised accordingly after zone localisation.

# SG52 EDGE/TERMINATION ZONE DECOMPOSITION v0.3 — PREREGISTERED BEFORE OUTCOMES

Purpose:
Locate exactly where the frozen SG51 LENGTH-conditioned ACZ REGION_AFTER_LOCUS gain comes from.

Use the already frozen SG51 length-conditioned factor model; do NOT retrain separate zone models. On each held-out token, attribute the LOCUS_ONLY minus ADDITIVE log-loss contribution event-by-event into mutually exclusive zones:
1. FIRST = first symbol/atom.
2. SECOND = second unit only when token length >=4.
3. MIDDLE = units indices 2..L-3 for length>=5.
4. PENULT = second-last unit for length>=2, except when already FIRST/SECOND in short tokens.
5. FINAL = last unit, except one-unit token where FIRST takes precedence.
6. END = explicit token termination event.
7. SHORT_INTERIOR = any unit of length 2-3 not assigned above by precedence.

Primary output:
Per zone, token-normalized contribution to the overall REGION_AFTER_LOCUS gain, share of total signed gain, exact physical-quire sign-flip p, and positive-block count, raw and atomic.

Decision:
- TERMINATION_DRIVEN if END alone contributes >50% of total positive gain.
- EDGE_MORPHOLOGY_DRIVEN if FIRST+SECOND+PENULT+FINAL contribute >50% and END <=50%.
- MIXED_EDGE_TERMINATION if END and edge morphology each contribute >=25%.
- If MIDDLE contributes >=25% and resolves, reopen internal-template interpretation.
No new model is fitted; this is an attribution of the frozen SG51 result.

# SG52 EDGE/TERMINATION ZONE DECOMPOSITION v0.3 — ZLZI RESULT

The SG51 exact-length-conditioned ACZ gain is overwhelmingly EDGE-local, not a whole-token internal-template effect.

RAW total REGION_AFTER_LOCUS = +0.109822 bits/token.
Attribution:
- FIRST -0.00143 (-1.3% signed share), unresolved.
- SECOND +0.07353 (67.0%), exact p=.015625, 6/6 blocks positive.
- MIDDLE +0.01327 (12.1%), p=.25, unresolved.
- PENULT +0.01329 (12.1%), p=.0625.
- FINAL +0.00265 (2.4%), unresolved.
- END +0.00851 (7.7%), p=.015625, 6/6.
FIRST+SECOND+PENULT+FINAL = 80.2% of total signed gain.

ATOMIC total = +0.085807 bits/token.
- FIRST -0.02932 (-34.2%), unresolved.
- SECOND +0.07534 (87.8%), p=.015625, 6/6.
- MIDDLE +0.00144 (1.7%), p=.484375, unresolved.
- PENULT +0.02174 (25.3%), p=.09375.
- FINAL +0.00811 (9.4%), p=.078125.
- END +0.00850 (9.9%), p=.015625, 6/6.
Edge sum = 88.4% of total signed gain.

Frozen classification: EDGE_MORPHOLOGY_DRIVEN.
The single dominant event is SECOND UNIT choice. END/termination contributes only ~8-10%; middle-token morphology contributes little and does not resolve.

Interpretation:
Exact token length is not acting as a proxy for a latent grammar governing the whole token. It stratifies an early token-family/edge decision, especially the distribution of the second unit after the first, plus a smaller penultimate/termination component.
This dovetails with earlier evidence that length is entangled with edge geometry/family and should not be promoted as an independent grammar state.

# SG52 SECOND-UNIT LOCALISATION — ALTERNATE TRANSCRIPTION PREREGISTRATION

Repeat the frozen LENGTH-conditioned event-zone attribution without any model change in ZLZB, TTLI, JSLI, VDRB and TTIA, raw and compound-atomic.

SECOND_UNIT_ROBUST if:
- SECOND contribution is positive in >=4/5 alternate layers in both raw and atomic;
- SECOND is the largest positive single zone in >=4/5 in both frames;
- no alternate gives a resolved negative SECOND contribution.

If passed, revise SG51 from generic LENGTH_CONDITIONAL_REGION to SECOND_UNIT_LENGTH_STRATIFIED_REGION and keep it COORDINATE, not CORE.

# SG52 SECOND-UNIT LOCALISATION — ALTERNATE TRANSCRIPTION RESULT

The ZLZI edge localisation replicates strongly.

Across ZLZB, TTLI, JSLI, VDRB and TTIA:
SECOND-unit contribution to the frozen exact-length-conditioned ACZ REGION_AFTER_LOCUS gain is positive in 5/5 raw and 5/5 atomic layers.
It is exact-sign-flip resolved in every layer/frame:
- raw SECOND contributions range +0.06286 to +0.07506 bits/token;
- atomic +0.06708 to +0.07577.
Positive physical blocks: 5/6 or 6/6 in every case.
SECOND is the largest positive single zone in 4/5 raw layers and 5/5 atomic layers.
No alternate layer has a negative SECOND contribution.

MIDDLE remains small and unresolved throughout.
END remains consistently positive but contributes only about 5-14% of total gain.
PENULT is a smaller secondary component.
JSLI raw is the only layer where FIRST exceeds SECOND, but SECOND remains positive/resolved.

Frozen SG52 decision:
SECOND_UNIT_ROBUST.

Scientific correction to SG51:
The earlier phrase "ACZ versus Pharmaceutical internal morphology contains a cross-locus regional/register component once token-length/template state is conditioned" was too broad.
The effect is NOT distributed across internal token morphology. It is overwhelmingly an early edge/prefix phenomenon: exact length stratifies the choice of the SECOND unit after the first, plus smaller penultimate/END effects.
On leak-safe middle events exact LENGTH itself has no regional signal.

Updated structural interpretation:
TOKEN-FAMILY / EDGE-GEOMETRY SELECTION is the appropriate parent. Exact final length is an informative coordinate for that early family choice, not an independently identified whole-token grammar state.

This is consistent with earlier length work:
- length is a real surface rhythm/geometry coordinate (SG35);
- length matching mediates near-copy effects (SG30);
- length participates in SG45 morphology state;
- prefix refinements are structurally strong.
It does NOT require a new independent LENGTH grammar.

# SG52 DESCRIPTIVE DRIVER AUDIT — FROZEN METHOD
Using the already fitted whole-quire length-conditioned models, attribute SECOND-unit LOCUS_ONLY-minus-ADDITIVE gain by (first unit, exact length) and by first unit alone. No refitting, no promotion test. Purpose only: identify whether the robust SECOND-unit effect is concentrated in a few established prefix families or diffuse.

# SG52 PROSPECTIVE PREFIX -> FINAL LENGTH TEST v0.4 — PREREGISTERED BEFORE OUTCOMES

Motivation:
The leak-safe and zone tests show that the SG51 length-conditioned regional signal is concentrated at the SECOND token unit. Because final length is a future observable, two causal readings remain possible:
A. a planned length/template state constrains the second unit;
B. early prefix/family choice constrains the continuation process and therefore predicts eventual length.
A prospective prediction test can determine whether the observed first two units carry held-out information about final length. It cannot alone prove causal direction, but strong early->length predictability would explain why conditioning on final length stratifies second-unit morphology.

Universe:
Ordinary RUN loci P0/P1/Pb/Pc/Pr/Pt across the canonical rulebook corpus. Whole physical quires are held out. Primary ZLZI raw+atomic; alternate layers ZLZB/TTLI/JSLI/VDRB/TTIA replicate the same test.

Target:
Exact final token length in the corresponding representation; lengths >=12 pooled as 12+ for support.

Hierarchical categorical predictors, training only:
M0 SECTION: P(length | section), shrunk to corpus-global length distribution.
M1 FIRST1: P(length | section, first unit), shrunk to M0.
M2 FIRST2: P(length | section, first two units), shrunk to M1; one-unit tokens back off to M1.
M2CLASS: P(length | section, first unit, coarse class of second unit Q/G/B/I/E/D/A/T/C/X), shrunk to M1.
Jeffreys global root; Dirichlet shrinkage lambda=32 at each child level, frozen before outcomes.

Primary contrast:
FIRST2 versus FIRST1 held-out codelength. Secondary: M2CLASS versus M1, and exact FIRST2 versus M2CLASS.
5,000 deterministic whole-quire sign flips (exact enumeration if <=12 blocks).

Decision:
PREFIX_PREDICTS_LENGTH if FIRST2 improves >0.05 bits/token with z>=2 in BOTH raw and atomic and positive in >=4/5 alternate transcription layers in both frames.
COARSE_PREFIX_FAMILY_SUFFICES if M2CLASS captures >=75% of FIRST2 incremental gain in raw+atomic and exact FIRST2 adds <=0.02 bits/token or unresolved.
This licenses a prospective prefix-length dependency, not a historical writing order.

# SG52 PREFIX -> LENGTH v0.4 — INITIAL RESULT, PHYSICAL-CONTROL REPLICATION PREREGISTERED BEFORE CONTROLLED OUTCOMES

Uncontrolled except for SECTION, the prospective dependency is strong:
ZLZI FIRST2 vs FIRST1 exact-length codelength gain:
- raw +0.08158 bits/token, null SD 0.02574, z=3.18, 15/16 physical quires positive.
- atomic +0.10480, SD 0.03370, z=3.11, 15/16.
All five alternate transcription/word-division layers are positive/resolved in raw and atomic; gains span raw +0.0758 to +0.1121 and atomic +0.0984 to +0.1590.

The coarse CLASS of the second unit captures essentially all of the incremental information:
ZLZI raw M2CLASS gain +0.07810 = 95.7% of exact-FIRST2 gain; exact second identity beyond class +0.00348, z=1.19: THE METRIC DOES NOT RESOLVE THIS.
Atomic M2CLASS +0.10715 = 102.2% of exact gain; exact identity is slightly worse, unresolved.
Across all alternate layers coarse class captures 94.3%-105.2% of exact second-unit gain; exact identity beyond class never resolves.

This strongly supports a prefix-family -> eventual-length dependency, but it must be checked against the already-known line-position/line-length effects before promotion.

# SG52 PREFIX -> LENGTH PHYSICAL-CONTROL v0.4b — PREREGISTERED

Same ordinary RUN corpus, whole-quire CV and length target.

Baseline physical context:
SECTION +
LINE_POSITION class {SINGLE,FIRST,LAST,INNER_EARLY,INNER_MIDDLE,INNER_LATE} +
LINE_TOKEN_COUNT capped at 15+.
LINE_POSITION uses the token's actual j/n within the IVTFF running-text line: first/last take precedence; remaining inner tokens are binned by normalized position thirds.

Hierarchy:
M0_PHYSICAL = P(length | section,line_position,line_token_count), shrunk hierarchically to section then global.
M1_FIRST1 adds first unit.
M2CLASS adds coarse second-unit class Q/G/B/I/E/D/A/T/C/X.
M2_FIRST2 adds exact second unit.
All child lambda=32, Jeffreys global root.

Primary:
M2CLASS vs M1_FIRST1.
Secondary:
M2_FIRST2 vs M2CLASS.
Same 5,000 whole-quire sign flips.

Promotion requires M2CLASS gain >0.05 bits/token with z>=2 in ZLZI raw+atomic and positive/resolved in >=4/5 alternate layers per frame. If it fails, the prospective prefix-length dependency is treated as mediated by physical line geometry.

# SUPER-GRAMMAR RESIDUAL SUPRA-LINE AUDIT v1.0 — PREREGISTERED BEFORE NEW OUTCOMES
Date: 2026-09-18

Target:
Revisit the 2026-08-22 calibrated +5.40-SD morphologically-distant complete-line order effect under the now-frozen Super-Grammar 1.0. This is a direct continuation of V20, not a new semantic fishing programme.

Legacy representation frozen:
- same 34,087-token ZLZI running-text event corpus from the Surface-Payload programme;
- same five physical-bifolium folds;
- same marginal-only 220-type eligible inventory construction;
- train-only K=8 degree-normalised spectral lexical representation;
- heldout line embedding = mean eligible-token embedding minus page eligible-token mean;
- same serial statistic: [(C1-Cfar)+.70*(C2-Cfar)]/1.70, Cfar separation >=6;
- complete physical line is the permutation unit.
First gate: exact reconstruction MUST reproduce the archived V20 target effect +0.000148617 above null with null SD ~0.000027547, z~5.40 and positive 5/5 folds before any modern result is interpreted.

Modern hostile sequence, frozen before outcomes:

R0 LEGACY_REPLICATION
Exact archived V20 nuisance exclusion and within-page coarse-line-length matched 999-permutation null.

R1 SG45_MORPH_FIREWALL
Rebuild the 220-type train-only association graph, but exclude from association learning any token pair that is:
(a) exact identity;
(b) raw ED<=1;
(c) same SG45 current-token morphology family = first raw character + final two raw characters + exact raw token length.
Everything else, including K=8 and statistic, remains unchanged.
Null = complete lines permuted within page and coarse line-length bin.
Question: does the effect survive the stronger modern visible-form firewall?

R2 GEOMETRY_PLUS_SURFACE_RESIDUAL
Using the R1 lexical embeddings, fit on training folds only a ridge model predicting each line's K=8 lexical embedding from NON-LEXICAL Super-Grammar/physical covariates. No exact token identity or 220-type identity enters the nuisance model.

Frozen line covariates:
- coarse line length bin and exact capped line token count;
- normalized vertical line index within page: y, y^2, quartile;
- mean/sd raw token length + proportions in length bins <=3,4,5,6,>=7;
- proportions of initial raw symbol classes for the ten globally common initial symbols plus OTHER;
- proportions of final raw symbol classes for the ten globally common final symbols plus OTHER;
- proportions of coarse second-unit classes Q/G/B/I/E/D/A/T/C/X;
- q incidence; maximal-i-run incidence and mean maximal i-run length;
- proportion of tokens whose SG45 morphology family appeared earlier in the preceding 64 tokens;
- exact-token R64 hit proportions in lag bins 2-5, 6-16, 17-64;
- within-line adjacent SG45-family repeat rate;
- first-token and final-token raw length, initial character and final character as coarse one-hot/hash-free categories;
- line-break m|p indicator from previous physical line;
- previous physical line's corresponding continuous surface summary, where defined.
Categorical levels are learned on training only; unseen heldout levels map OTHER.
Ridge penalty fixed at alpha=10 after train-standardisation; no target-order statistic is used to tune it.
Subtract predicted lexical embedding from each heldout line to obtain a cross-fitted residual embedding.
Primary R2 null = 9,999 complete-line permutations within page, coarse line-length bin AND physical vertical quartile. Lines in singleton strata remain fixed.
Report actual-minus-null effect, null SD, z, p, fold deltas, and effective movable-line fraction.

R3 ABLATION BOUNDS
Repeat R2 residual test after removing one nuisance family at a time:
- no geometry;
- no exact-R64/family-history features;
- no morphology distributions;
- no previous-line surface features.
This identifies which modern production layer absorbs any reduction. No R3 ablation can create a positive semantic claim; it is diagnostic only.

R4 PROSPECTIVE RESIDUAL-STATE TEST
If R2 remains positive:
On training folds only learn a ridge transition from residual line embedding at t and t-1 to residual embedding at t+1. Baseline predicts zero residual (page-centred) and a surface-only nuisance prediction is already removed in R2.
Score heldout pages prospectively by squared-error reduction and cosine/dot-product improvement on eligible t+1 lines.
Inference uses 9,999 within-page line-order permutations under the same length-bin + vertical-quartile strata, refitting no heldout parameter.
Positive call requires:
- pooled error reduction >0 with empirical p<=.01;
- positive fold delta >=4/5;
- standardized effect >=2 null SD.
This is the critical distinction between descriptive serial organisation and a reusable latent state.

R5 ALTERNATE-TRANSCRIPTION SENSITIVITY
Only if R2 or R4 passes, repeat the surviving statistic on ZLZB, TTLI, JSLI, VDRB and TTIA using the same physical lines/folds and representation construction where coverage is sufficient. A robust residual requires same positive direction in >=4/5 alternates and no resolved contradiction.

Interpretation gates:
COLLAPSES_TO_PRODUCTION if R2 |z|<2 or p>.05.
DESCRIPTIVE_RESIDUAL_ONLY if R2 resolves but R4 fails.
PREDICTIVE_SUPRALINE_STATE if R2 and R4 both pass.
No outcome licenses semantics, payload, natural language or cipher. A predictive residual is only structured state beyond the tested frozen surface covariates.

Audit rule:
Any implementation bug or mismatch with the archived R0 replication invalidates all downstream outcomes until corrected. Retractions are appended with equal prominence.

# SUPER-GRAMMAR SUPRA-LINE AUDIT — NEGATIVE-RESULT POWER BOUND ADDENDUM
Frozen after R0-R4 primary model definitions, before synthetic calibration outcomes are generated.

Because R2 may fail by either genuine absorption or an over-aggressive nuisance projection, calibrate the R2 residual statistic on planted supra-line states after primary scoring.

Synthetic calibration:
- retain the real R1 line embeddings, folds, page geometry and frozen nuisance covariates;
- add to every physical line an independent K=8 Gaussian AR(1) latent vector within each page, rho=.70, stationary per-dimension SD a;
- the latent vector is generated independently of every nuisance covariate;
- refit the identical training-only alpha=10 nuisance ridge and rerun the identical geometry+length-stratified conditional permutation test;
- amplitudes a in {.002,.003,.004}, 60 independent replicates each;
- B=499 conditional permutations per replicate for calibration only;
- complete detection call: empirical p<=.05, positive effect, >=4/5 positive folds.

Also run 60 a=0 exchangeable residual controls by permuting the observed R1 lexical line vectors within the frozen geometry+length strata before nuisance fitting, to estimate false-positive behaviour.

Interpretation:
If FPR<=.05 and power>=.80 at an amplitude whose pre-nuisance serial effect is no larger than the archived/geometry-controlled Voynich effect, an unresolved R2 result is a meaningful bound against that class of latent line state.
If power is <.80 at all comparable amplitudes, classify R2 negative as NON_IDENTIFIABLE rather than absorbed.
This calibration cannot convert the observed R2 into a positive result.

# SUPRA-LINE AUDIT v1.1 — PROSPECTIVE-NUISANCE CAUSAL SENSITIVITY, PREREGISTERED BEFORE STATE-PREDICTION OUTCOME

Reason:
The frozen R2 primary uses REALISED current-line morphology/length summaries as nuisance covariates. This is a valid conditional/mediation test, but it is not a pure prospective production-layer residual because a hidden line state could itself influence those realised surface summaries. Conditioning on them could block genuine shared-state signal.

Observed sensitivity already fixed:
Using only covariates available before the line's lexical/token realisation — physical line token-count/vertical geometry plus the previous physical line's continuous surface summary — the supra-line residual remains positive: effect +4.42823e-05 with null SD 1.62011e-05 (z=2.73), empirical p=.0036, 5/5 folds positive. Geometry-only gives +4.48494e-05 / SD 1.62246e-05 (z=2.76), p=.0026.
These are post-primary causal sensitivities, not replacements for the frozen R2 mediation result.

Frozen v1.1 critical test:
R4P PROSPECTIVE-NUISANCE LATENT STATE
- Use the already frozen GEOM_PREVLINE residual line embeddings; no nuisance retuning.
- Training folds only: ridge transition from residual embedding at lines t and t-1 to residual embedding at t+1, within page.
- Baseline predicts zero residual because page centering and prospective nuisance prediction are already removed.
- Ridge alpha=10 fixed.
- Score heldout pages by squared-error reduction per eligible target line.
- Null: 9,999 complete-line order permutations within page, coarse line-length bin and vertical quartile, preserving the realised line residual vectors; model parameters remain training-only and are not fit to heldout order.
- Positive call requires effect/null SD>=2, p_upper<=.01, and >=4/5 fold deltas positive.

Interpretation:
If R4P fails, classify the prospective residual as DESCRIPTIVE_ORDER_ONLY: line order remains nonexchangeable after prior-state production controls, but no reusable latent state has been demonstrated.
If R4P passes, classify PREDICTIVE_SUPRALINE_STATE and proceed to alternate-transcription replication before any content/payload interpretation.
The frozen R2 result remains MEDIATED_BY_REALISED_SURFACE: full current-line morphology makes the statistic unresolved. The two tests answer different causal questions and neither supersedes the other.

# SUPRA-LINE AUDIT v1.2 — LOW-COMPLEXITY PROSPECTIVE PERSISTENCE, PREREGISTERED BEFORE OUTCOME

Reason:
v1.1 R4P uses a 16x8 ridge transition (128 coefficients). The legacy V20 signal is a low-rank persistence statistic, not evidence for an arbitrary multivariate transition. A true persistent latent line state could therefore be underpowered under the high-dimensional transition even if its transferable rule is simply "the next line tends to remain in the same residual direction."

Frozen input:
The already-frozen GEOM_PREVLINE residual embeddings from v1.1. No nuisance refitting or representation change.

P1 SCALAR_AR1:
On training pages only fit one scalar rho minimizing ||R_{t+1} - rho R_t||^2 over all eligible adjacent residual-line pairs and all 8 dimensions jointly. Clip rho to [-1,1] only after unconstrained fit if necessary.
Predict heldout R_{t+1}=rho R_t.
Baseline predicts zero residual.

P2 SCALAR_AR2:
Secondary two-parameter model R_{t+1}=rho1 R_t + rho2 R_{t-1}, fitted jointly over all 8 dimensions on training pages only.

Heldout score:
Squared-error reduction relative to zero, per eligible target line.
Inference:
9,999 complete-line permutations within page, coarse line-length bin and vertical quartile, identical to v1.1. Training coefficients remain frozen for every heldout permutation.
Report effect, null SD, z, empirical p and fold signs.

Decision:
LOW_COMPLEXITY_PREDICTIVE_STATE if AR1 has effect/nullSD>=2, p<=.01 and >=4/5 positive folds.
AR2 is corroborative only and cannot rescue a failed AR1 unless it independently has >=2 SD, p<=.01, >=4/5 positive folds AND improves over AR1 in a separate heldout comparison.
If both fail, retain DESCRIPTIVE_ORDER_ONLY.

# SUPRA-LINE AUDIT v1.3 — DESCRIPTIVE ALTERNATE-TRANSCRIPTION ROBUSTNESS, PREREGISTERED BEFORE OUTCOMES

The prospective state gates v1.1/v1.2 have failed, so this test cannot promote a latent predictive state. Its only purpose is to decide whether the remaining descriptive line-order effect is an artefact of ZLZI word division/transcription.

For each alternate layer ZLZB, TTLI, JSLI, VDRB, TTIA:
- use exactly the same 4,097 physical running-text lines/pages and five bifolium folds as the reproduced August V20 corpus;
- require the alternate transcription for a physical line; missing lines are marked invalid, never imputed;
- rebuild the layer-specific eligible vocabulary from training-fold support using the identical min-training-count>=10 and top-220 rule;
- rebuild the train-only K=8 degree-normalised spectral lexical representation;
- exclude exact identity, raw ED<=1 and same layer-specific SG45 morphology family (first character + final two characters + exact token length) from association learning;
- page-centre line embeddings exactly as in V20;
- primary null: 4,999 complete-line permutations within page, alternate-layer coarse line-length bin and physical vertical quartile.

Report per layer: eligible-token vocabulary size before top-220 truncation, valid-line coverage, actual-minus-null effect, null SD, z, p and fold signs.

DESCRIPTIVE_TRANSCRIPTION_ROBUST if same positive direction occurs in >=4/5 alternate layers, >=3/5 have effect/nullSD>=2 and p<=.05, and no alternate has a resolved negative effect.
This result cannot override the failed prospective-state tests and cannot license semantics.

# SUPRA-LINE v1.1/v1.2 NEGATIVE PREDICTION — POWER CALIBRATION ADDENDUM
Frozen before calibration outcomes.

Observed prospective-state outcomes are already sealed. Calibrate whether the R4P detector could recover a genuinely reusable line state of plausible amplitude.

Synthetic source:
- use the fold-specific GEOM_PREVLINE residual embedding arrays before the transition model;
- first destroy their native line order by independently permuting complete residual line vectors within each page x coarse-length x vertical-quartile stratum;
- add an independent page-local K=8 Gaussian AR(1) state with rho=.70 and stationary per-component SD a;
- amplitudes a={0,.002,.004,.006};
- 40 independent replicates per amplitude;
- for every replicate refit the identical training-only 128-parameter R4P ridge transition and test heldout folds with 499 conditional permutations;
- a=0 is the false-positive control.

Detection call exactly mirrors R4P: p_upper<=.01, effect/nullSD>=2, >=4/5 positive folds.
Report false-positive rate and power.

If FPR<=.05 and power>=.80 at a signal whose induced adjacent/far descriptive effect is <= the observed prospective-nuisance residual effect, the failed R4P meaningfully excludes that persistent-state class at that scale.
Otherwise classify the predictive-state negative as POWER_BOUNDED_NON_IDENTIFIABLE below the first 80%-power amplitude.
This calibration cannot create a positive Voynich result.

# SUPRA-LINE v1.3 GEOMETRY GRANULARITY SENSITIVITY — POST-PRIMARY, NON-DECISION-DRIVING
The calibrated historical geometry control uses vertical quartiles (K_y=4). To test representation/decision fragility without changing the primary decision, rerun the R1 SG45-firewall descriptive statistic with K_y in {3,5,6,8} using the same page+coarse-line-length conditional shuffles and 4,999 permutations. Report effect/null SD, p and movable-valid-line fraction. K_y=4 remains the frozen primary geometry result. Fine-stratum collapse is interpreted only as sensitivity to unresolved continuous page geometry, not as retroactive invalidation.

# EXTERNAL CONTENT ANCHOR ON SUPRA-LINE RESIDUAL v1.0 — PREREGISTERED BEFORE VISUAL OUTCOMES
Date: 2026-09-18

Purpose:
Discriminate, as far as available external evidence permits, whether the transcription-robust supra-line residual is merely manuscript production/layout structure or covaries with independently measured illustrated-object content. This is an external-anchor test, not a decipherment test.

Frozen text inputs:
1. PROSPECTIVE residual = SENS_GEOM_PREVLINE residual line embeddings from the accepted supra-line audit. These remove physical geometry and previous-line surface summaries but do not condition on the current line's realised morphology.
2. FULL_SURFACE residual = R2_FULL residual line embeddings. These additionally condition on realised current-line morphology/history and made the supra-line serial statistic unresolved.
Both are held-fold/cross-fitted 8-D residuals on the exact August 4,097-line corpus; no visual data entered their construction.

Page fingerprint:
For every page independently, divide physical lines into four equal vertical quartiles by line index/n_lines. Within each quartile average all valid 8-D residual line embeddings. Concatenate the four quartile means in physical order into a 32-D vector and L2/cosine compare pages. A page is eligible only if it has >=8 valid residual lines and every quartile contributes >=1 valid line. No visual outcome is used to select pages or dimensions.

Primary visual cohort:
Exactly the previously frozen Herbal image-anchor eligibility:
- deterministic stored whole-plant target with embedding;
- >=30 primary_eligible RF token occurrences;
- non-null mapped quire.
Intersection with the predeclared residual-fingerprint eligibility only; no page is excluded by outcome.

Primary visual predictor:
Stored deterministic WHOLE-PLANT 3072-D embedding cosine similarity, selected by the same pre-existing rank rule as the 2026-08-22 image-anchor programme.

Primary text response:
Cosine similarity of PROSPECTIVE 32-D ordered residual fingerprints.

Nuisance controls:
Pairwise partial correlation controlling:
1. log(1+folio-sequence distance);
2. same Herbal section;
3. absolute log RF token-count ratio;
4. absolute log physical line-count ratio;
5. absolute difference in residual-valid-line fraction;
6. plant bbox centre distance;
7. absolute log plant bbox area ratio;
8. plant bbox aspect-ratio difference.
Same-quire structure is handled by permutation, not a regression dummy.

Primary inference:
9,999 deterministic within-quire QAP visual-label permutations. Plant embedding and plant bbox geometry move together exactly as in the old image-anchor programme. Text residual fingerprints and text-side covariates remain fixed.
Report raw r, nuisance-adjusted partial r, actual-minus-null, null SD, z, empirical one-sided p, and Herbal-A-only direction.

Primary gate:
EXTERNAL_CONTENT_COVARIATION if actual-minus-null >0, p<=.01 and z>=2.
SUGGESTIVE if p<=.05 but z<2.
UNRESOLVED otherwise.
No result alone licenses semantics.

Frozen hostile/sensitivity tests, opened only after primary is fixed:
A. FULL_SURFACE residual fingerprint under identical WHOLE-PLANT test. If primary survives but FULL_SURFACE collapses, classify CONTENT_OR_PLANNING_SIGNAL_MEDIATED_BY_REALISED_MORPHOLOGY; this is not semantic proof.
B. ROOT embedding replication using the stored deterministic root visual target, identical cohort intersection/nuisances/QAP. Root is corroborative, not independent in a strict statistical sense.
C. ORDER-SPECIFICITY: 999 deterministic within-page line-order permutations of the PROSPECTIVE residuals, preserving each page's residual-line multiset and visual assignment. Rebuild the 4-quartile fingerprint each time and recompute the exact same partial correlation without re-tuning. If the observed plant association exceeds >=95% of order-destroyed values, the external association depends on residual line order rather than only the unordered page residual distribution.
D. OLD TF-IDF BENCHMARK: quote/reproduce the frozen August whole-plant result only as a benchmark; no parameter from it can be tuned.

Power calibration:
After observed primary is sealed, inject an external visual-linked latent signal into copies of the real residual fingerprints at amplitudes chosen on a fixed grid, under within-quire QAP, to estimate the detectable partial-correlation frontier. A negative primary is called absence only above the >=80%-power frontier; otherwise NON_IDENTIFIABLE below it.

Interpretation hierarchy:
- External association + order-specificity + root corroboration = strongest available evidence that the residual tracks illustrated-object variation, but still not decipherment.
- External association without order-specificity = page-level residual/image co-structure, carrier unresolved.
- No external association with adequate power = evidence against illustrated-plant-linked residual content at the powered scale.
- No external association without adequate power = non-identifiable.

# EXTERNAL CONTENT ANCHOR v1.0 — PRE-OUTCOME REPRESENTATION CORRECTION

Before any plant/text association is calculated, the originally written 32-D concatenated quartile vector is withdrawn for cross-page comparison because the train-fold K=8 spectral lexical spaces are defined only up to orthogonal rotation/sign. Direct cosine between raw coordinates from pages held out in different folds is therefore not gauge-invariant.

Frozen corrected page fingerprint:
- retain the same four ordered vertical-quartile mean residual vectors q1..q4 in each page's own held-fold K=8 space;
- replace the raw 32-D concatenation by the 10 unique entries of the symmetric 4x4 Gram matrix G_ab = q_a dot q_b for 1<=a<=b<=4, in fixed order (11,12,13,14,22,23,24,33,34,44);
- cosine similarity of these 10-D Gram fingerprints is the primary text response.
This representation is invariant to any common orthogonal rotation/sign change of the fold-specific lexical basis, preserves coarse ordered trajectory geometry, and reduces transport dimensionality without looking at visual outcomes.

Eligibility remains >=8 valid lines and >=1 valid line in every quartile. All nuisance controls, QAP, hostile FULL_SURFACE test, ROOT replication and order-specificity test remain unchanged.
No 32-D cross-page outcome has been calculated or inspected.

# EXTERNAL CONTENT ANCHOR v1.0 — PRIMARY WHOLE-PLANT RESULT SEALED

Primary cohort after intersection with rotation-invariant prospective residual fingerprints:
N=94 Herbal pages, 4,371 page pairs; 88 Herbal-A pages.

PROSPECTIVE_GRAM whole-plant result:
- raw visual/text cosine correlation: +0.0719250.
- nuisance-adjusted partial r: +0.0608059.
- 9,999 within-quire QAP null mean: +0.0145046.
- actual-minus-null: +0.0463013.
- null SD: 0.0532828.
- z = +0.869.
- one-sided empirical p = .2000.
Primary decision under frozen gate: UNRESOLVED / DOES NOT PASS EXTERNAL_CONTENT_COVARIATION.

Herbal-A-only directional sensitivity:
raw r +0.0810961; partial r +0.0704316 after omitting the now-constant same-section nuisance. Direction remains positive.

Interpretation at primary seal:
The prospective supra-line residual page fingerprint does not show a statistically resolved association with whole-plant visual similarity under the frozen within-quire QAP. This is not yet evidence of absence because the v1.0 protocol requires power calibration before a negative can be interpreted as exclusion. Hostile/sensitivity analyses may now be opened without retuning the primary representation.

# CONTROL-SEQUENCE FORWARD TOURNAMENT v0.1 — PREREGISTERED BEFORE OUTCOMES
Date: 2026-09-18

Question:
Can a compact table/control-sequence layer, placed ABOVE the already-tested August Stage-2 surface realiser, materially improve forward Voynich surface adequacy without changing glyph/token realization machinery?

This is a constructive mechanism tournament, not a historical identification test. It explicitly follows the hypothesis that the visible grammar is a conditional renderer driven by an upstream family/control sequence.

Frozen corpus and evaluation:
- exact August Surface–Payload 34,087-token ZLZI running-text corpus;
- same 49 physical bifolia and same five outer folds;
- same real physical page/line skeleton in heldout folds;
- same Stage-2 token realiser, character novelty source, page token reinforcement, exact-reuse and mutation machinery;
- ONLY the family/control-sequence selector is replaced;
- all parameters/lookup tables for a heldout fold are learned from that fold's outer-training bifolia only;
- no heldout token/family multiset, page vocabulary, or future line is supplied;
- 8 deterministic forward replicates per fold/controller;
- the August 20-metric frozen battery is primary descriptive adequacy;
- real-vs-real covariance panel and joint Mahalanobis distance are recomputed exactly as in Stage 2;
- because the historical absolute 20/20 closure event is known to be low-specificity, ranking is by (a) joint distance relative to baseline, (b) mean absolute metric z, (c) number of |z|<=2 metrics, and (d) whether improvement is positive in >=4/5 folds. The old absolute closure count is reported only descriptively.

Shared realization:
After a controller proposes a canonical family, exact-form selection/reuse/mutation and the character novelty source are unchanged from the validated August Stage-2 code. Thus differences isolate the upstream controller as far as this implementation permits.

Frozen controllers (H1-H10):

H1 PAGE_CODEBOOK
At page start draw a restricted set of canonical families from the training section distribution. Codebook size is the training median distinct-family count for pages of the corresponding length bin. All non-char family draws are restricted to this set, with 15% baseline escape.

H2 LINE_CODEBOOK
At each physical line start draw a small allowed family table from the section/position distribution. Size is the training median distinct-family count for lines in the same line-length bin. Restriction persists only for that line; 15% baseline escape.

H3 MARKOV_LINE_REGIME
Cluster outer-training physical lines into K=8 family-frequency regimes using train-only TF-normalised family vectors and MiniBatchKMeans. Learn section-specific initial-state and adjacent-line transition tables. Each generated line has one latent regime; family probabilities are 85% regime distribution + 15% baseline context.

H4 DETERMINISTIC_REGIME_WALK
Uses the same K=8 line regimes as H3, but each state follows its modal learned successor with 85% probability and a transition-table draw otherwise. Tests quasi-tabular/cyclic line progression rather than stochastic Markov drift.

H5 PAGE_QUOTA_DECK
At page start sample a finite multiset of family cards from the section distribution with exactly one card per non-char token position. Draw without replacement, weighted by the ordinary context probability among remaining cards. This is an exchangeable preplanned page table, not a recency process.

H6 MOTIF_LIBRARY
Learn within-line canonical-family motifs of lengths 3-6 from outer training. At line start choose a same-section motif/template matched to line length and tile/truncate it; each family position follows the template with 80% probability and baseline context with 20%. No heldout line is used as an exemplar.

H7 TRAINING_PAGE_EXEMPLAR
At page start select one outer-training page from the same section and nearest physical line-count/token-count profile. Map its family trajectory onto the heldout page by normalised token position within corresponding line bands. Follow exemplar family with 75% probability, baseline with 25%. Tests source/exemplar copying as the missing control sequence.

H8 RANK_WALK_TABLE
For each training context (section, line-position class, previous family), freeze the top-8 family ranking. A hidden rank state resets at each line and performs a {-1,0,+1} random walk with probabilities learned from training by projecting observed families onto their context ranks. The current rank selects the table entry, with 20% baseline escape.

H9 DUAL_EDGE_TABLE
Replace direct family selection by two train-only tables: initial-symbol group and terminal/length-bin group, conditioned on section, line-position and previous family. Sample the two coordinates independently, then sample among training families satisfying both; back off to one-coordinate intersection then baseline. Tests a hand-usable two-axis paradigm table.

H10 LOW_ENTROPY_CONTROL_TAPE
At each token compute the baseline context's top-4 candidate families. Infer the observed candidate-rank tape on outer training and fit a four-state line-reset Markov chain over ranks. In forward generation the hidden rank chain chooses among the top four candidates; 15% baseline escape. This is the most literal compact control-tape model.

Baseline B0:
Unmodified August Stage-2 family selector, generated and scored in the same execution.

Decision:
A controller is a CONSTRUCTIVE_ADVANCE only if:
- joint distance beats B0 in >=4/5 folds;
- pooled mean joint distance is at least 10% lower than B0;
- pooled mean absolute metric z is lower than B0;
- it does not create a new systematic |z|>3 failure in >=3 folds.
A controller that wins only on one metric is not retained.
If several pass, carry only the Pareto-nondominated set into a second tournament with SG1.0-derived family definitions/edge state.
If none pass, reject the simple finite-table controller classes H1-H10 as sufficient at this resolution; do NOT reject an externally driven control sequence in general.

# CONTROL-SEQUENCE FORWARD TOURNAMENT v0.1 — RESULT

The ten-controller forward tournament was run exactly on the August 34,087-token Stage-2 corpus, five physical-bifolium folds, eight forward replicates per fold, with only the upstream family/control selector changed. The exact-form realiser, novelty source, reuse/mutation mechanics, physical layout skeleton and 20-metric evaluation were held fixed.

Only H1 PAGE_CODEBOOK passes the preregistered CONSTRUCTIVE_ADVANCE gate.

B0 unmodified August Stage-2 baseline:
mean joint Mahalanobis distance 2.10322;
mean absolute metric z 1.62640;
mean metrics within +/-2 SD = 13.2/20.

H1 PAGE_CODEBOOK:
mean joint distance 0.93970;
55.32% lower than B0;
beats B0 joint distance in 5/5 folds;
mean absolute z 1.10522;
mean metrics within +/-2 SD = 17.0/20;
no newly created >3-SD metric failure recurring in >=3 folds.
Fold joint distances B0 -> H1:
F0 3.215 -> 1.609
F1 1.491 -> 0.495
F2 1.908 -> 0.746
F3 2.230 -> 1.283
F4 1.671 -> 0.565.

H1 is not full closure. The main remaining recurring defects are:
- SD token length too narrow in 3/5 folds;
- exact lag-2-5 recurrence still too weak in 3/5;
- line-boundary ED1 reset still weak in 2/5;
- page composition excess remains high in 2/5.
But H1 substantially repairs type density, Simpson concentration, OOV-type deficit, raw family adjacency, page new-type slope and bifolium vocabulary overlap.

Important negative contrasts:
H2 LINE_CODEBOOK and H10 LOW_ENTROPY_CONTROL_TAPE achieve even lower joint distances numerically, but fail the frozen gate because they catastrophically overproduce family adjacency and distort length/diversity; they are imitation tradeoffs, not balanced improvements.
H8 RANK_WALK also lowers joint distance 55% in all folds but creates systematic length/family-adjacency failures.
H3/H4 line-regime states, H5 finite page quota deck, H6 motif library, H7 training-page exemplar and H9 dual-edge table do not improve the balanced forward phenotype.

Interpretation:
The winning object is specifically a PAGE-SCALE AVAILABILITY SET, not a line-local state, deterministic motif/exemplar sequence, quota schedule, or low-entropy rank tape.
At page start H1 draws a restricted family codebook from the training section distribution; subsequent local production uses the ordinary surface realiser with 15% escape. Typical observed page family-set sizes in the corpus are not tiny: overall median 56.5 distinct old-family classes, with section medians approximately Herbal 50.5, Pharma 62, Bio 86, Recipes 133, Astro 28.

This is a constructive refinement of, not a contradiction to, the August hierarchical working-set result. The old programme established that an active repertoire was a missing component but implemented it chiefly as a soft hierarchical reservoir/reinforcement process. The new tournament says a harder PRESELECTED AVAILABILITY SET is much more effective than the other nine controller classes.

Scientific status:
CONSTRUCTIVE_ADVANCE / NOT HISTORICAL IDENTIFICATION / NOT FULL CLOSURE.
The result supports testing:
PAGE CODEBOOK -> local SG1.0 family/edge realiser -> R64 recurrence / junction / line mechanics.
It does not by itself distinguish a content-selected codebook from a mechanically/randomly selected codebook.

# PAGE-CODEBOOK + SG1 REALISER v0.2 AND MONTE-CARLO — PREREGISTERED BEFORE OUTCOMES
Date: 2026-09-18

Purpose:
Take the sole v0.1 constructive winner, H1 PAGE_CODEBOOK, and replace the August exact-token realiser beneath it with the currently supported Super-Grammar 1.0 surface components. Then quantify forward robustness by Monte Carlo rather than a small seed panel.

Frozen outer structure:
- same 34,087-token ZLZI running-text corpus, five physical-bifolium folds, physical heldout page/line skeleton and August 20-metric forward battery;
- same H1 old-family page codebook selector: first raw character + final raw character + coarse raw-length bin, page codebook size learned from outer training, 15% baseline escape;
- no heldout token/family multiset or future line may be used.

SG1-inspired exact-token realiser under the H1 family:
1. SG edge/template coordinate for the current token = first raw character + final two raw characters + exact raw length.
2. Stable vocabulary reservoir: probability .60 of drawing an attested training token compatible with the selected old-family and SG edge/template; .40 compositional generation.
3. Compositional branch: section-conditioned order-3 raw character model with alpha=16 backoff, frozen 19 hard-zero bigrams, and START/END handled prospectively. The generated form is constrained to the already-selected first character, final-two suffix and exact length.
4. Local subword state: character continuation counts from the previous 64 generated tokens are shrunk to the training order-3 parent with alpha=16.
5. SG45 current-family selection: select the current SG edge/template within the H1 old family using outer-training counts conditioned on section, SPACE/LINE_BREAK and previous token morphology (previous first character + final two + exact length), shrunk with lambda=1024 to the old-family section/line-position parent.
6. Exact R64 recurrence: weight w=.05; eligible exact forms are generated tokens at distances 2-64 on the current page, constrained to the selected SG edge/template. If no eligible hit, mass returns to the base realiser.
7. Sequential ED1 mutation is OFF. Immediate exact previous-token identity is not a dedicated mechanism. These match the current SG1 closure conclusions.

Frozen variants:
A0 = v0.1 H1 PAGE_CODEBOOK unchanged.
A1 = SG_TEMPLATE: H1 + SG edge/template + .60 vocabulary reservoir + order3/hard-zero compositional branch; no SG45 child, no local subword cache, no R64.
A2 = SG_TEMPLATE_R64: A1 + exact R64 w=.05.
A3 = SG1_FULL: A2 + SG45 previous-token morphology and local subword64.
A4 = SG1_FULL_NO_CODEBOOK: identical A3 realiser but ordinary unrestricted family selector; isolates the page-codebook contribution.

Primary forward comparison:
Eight deterministic forward replicates per fold/variant. Recompute the same real-vs-real fold covariance panels used in v0.1.
A3 is an SG1 CONSTRUCTIVE ADVANCE over A0 only if:
- joint distance A3 < A0 in >=4/5 folds;
- pooled joint distance improves >=10%;
- mean absolute metric z improves;
- no new metric has |z|>3 in >=3 folds.
A4 must be materially worse than A3 on pooled joint distance or page codebook is not retained as an independent upstream layer.

Monte Carlo opens only after the small frozen comparison:
- select between A0 and A3 strictly by the primary gate; if A3 fails, Monte Carlo A0; if A3 passes, Monte Carlo A3 and A0 as comparator.
- 300 independent Monte Carlo draws per fold/model (1,500 generated heldout corpora per model), deterministic seeds fixed by model/fold/replicate.
- no parameter refitting across Monte Carlo draws.
Report distributions of joint distance, mean absolute z, within-2SD count, each metric z, and model-minus-comparator joint distance.
Primary MC robustness quantities:
(a) P(model beats comparator joint distance);
(b) P(within-2SD >=17/20);
(c) P(no |z|>3);
(d) P(joint distance <= fold real-vs-real 95% envelope);
(e) foldwise and pooled 95% Monte Carlo intervals.
A mechanism is MC_STABLE if P(beats comparator)>=.90 pooled and >=.75 in every fold, with no systematic metric failure appearing in >50% of draws.

Scope:
This is constructive forward adequacy only. It cannot identify whether the page codebook is content-selected, mechanically selected, copied from an exemplar, or arbitrary.

# PAGE-CODEBOOK MONTE-CARLO v0.2 — COMPUTE-COUNT AMENDMENT BEFORE OUTCOMES
The 300-draw-per-fold execution was launched in three 100-draw shards per fold. Before any shard completed or any Monte-Carlo outcome was opened, runtime profiling showed the full 1,500 paired fold simulations would be unnecessarily expensive for the requested robustness check.
Freeze the operational Monte-Carlo at the first deterministic 100-draw shard per fold/model (500 generated heldout corpora for H1 and 500 paired B0 corpora). The start=100 and start=200 shards are cancelled before producing result files.
The inference definitions and seeds are unchanged. At N=100 per fold the worst-case Monte-Carlo standard error for a foldwise probability is 0.05; pooled across 500 draws it is <=0.022. Additionally report a 10,000-resample bootstrap of the original eight-seed ensemble mean from those 100 draws, which directly targets the scale of the v0.1 forward comparison.
No Monte-Carlo outcome was inspected before this count amendment.

# PAGE-CODEBOOK + SG1 REALISER v0.2 / MONTE-CARLO — FINAL RESULT 2026-09-18

SG1 direct generative transplant:
The preregistered A3 SG1_FULL replacement FAILS as a constructive improvement over the old H1 page-codebook realiser.
A0 H1 old mean joint distance = 0.947067; A3 = 1.124817, i.e. A3 is 18.77% WORSE.
A3 is worse on joint distance in all 5/5 folds.
Mean absolute metric z worsens 1.14467 -> 2.00573.
Mean within-2SD metrics falls 16.2/20 -> 12.8/20.
Main A3 defects are excessive OOV event rate, excessive family adjacency, depressed exact-adjacent-same-family, and too-narrow length in several folds.
Interpretation: the frozen SG1 predictive mixture cannot be naively reinterpreted as a forward historical sampling algorithm. In particular, treating the v=.60 vocabulary mixture as a literal 60/40 sampling source over stable vocabulary vs compositional order-3 generation overproduces novelty. Predictive closure and generative mechanism remain distinct.

Per frozen rule, Monte Carlo therefore evaluates the original H1 PAGE_CODEBOOK against unchanged B0, not A3.

Monte Carlo operational count:
100 deterministic paired draws per fold/model after the pre-outcome count amendment = 500 H1 generated heldout corpora and 500 paired B0 corpora. 10,000 paired bootstrap resamples of eight-draw ensemble means per fold, plus 100,000 whole-five-fold combinations.

Monte Carlo result:
Single generated-corpus P(H1 joint distance < B0):
F0 1.00; F1 .94; F2 .99; F3 .99; F4 .98. Mean=.98.
No metric has |z|>3 in >50% of individual H1 draws in >=3 folds.

Eight-seed ensemble scale:
P(H1 beats B0)=1.000 in EACH of 5 folds.
Whole-five-fold Monte Carlo:
P(mean H1 joint < mean B0)=1.000.
P(H1 beats B0 in >=4/5 folds)=1.000.
Mean joint distance H1=.94503 vs B0=2.08802.
Median joint-distance improvement=54.86%; 95% MC interval 50.13%-58.61%.
P(joint distance lies inside the fold real-vs-real 95% envelope)=1.000 in every fold and P(>=4/5)=1.000.

The model is therefore MC_STABLE under the preregistered definition.

However it is NOT absolute forward closure:
At the eight-seed ensemble scale, P(within +/-2SD on >=17/20 metrics) is highly fold-dependent and P(>=4/5 folds meeting it)=.0048.
P(no individual |z|>3 in >=4/5 folds)=0.
Persistent defects are regime-specific:
F0 boundary ED1 reset;
F1 token-length SD / page composition / lag-1 exact excess;
F2 lag-2-5 exact recurrence deficit;
F3 lag-2-5 exact/ED1 deficit;
F4 length SD and lag-2-5 exact deficit.
No single >3SD metric fails in a majority of individual draws in >=3 folds.

Final scientific status:
SG53_PAGE_CODEBOOK_CONTROL_LAYER = CONSTRUCTIVE_ADVANCE_MC_STABLE_NOT_FULL_CLOSURE.
The hard preselected page-scale availability set is not a seed accident: its ~55% multivariate forward improvement is stable under Monte Carlo.
The naive SG1 predictive-to-generative transplant is rejected. The next realiser must preserve H1 and translate SG1 constraints into a calibrated generative transition engine rather than interpreting predictive mixture weights as source probabilities.

# PAGE-CODEBOOK FACTORISATION / MODULE RECONSTRUCTION v0.1 — PREREGISTERED BEFORE OUTCOMES
Date: 2026-09-18

Goal:
Determine whether the empirically successful H1 page-specific family availability sets can themselves be generated by a compact reusable module/table system rather than by unconstrained page-specific sampling.

Corpus/family definition:
Use the exact v0.1 control-tournament corpus and old-family coordinate (first raw character + final raw character + coarse raw-length bin) so the factorisation target is exactly the object that produced the Monte-Carlo-stable H1 forward improvement.
Primary page universe = ordinary running-text pages with >=20 non-char tokens.
A page codebook is the binary set of old-family types attested on that page.
Physical bifolia remain indivisible outer-validation blocks.

Models, all trained ONLY on outer-training bifolia:

M0 SECTION_INDEPENDENT:
For each section learn family inclusion probability p_f. Generate a page codebook by independent Bernoulli draws, then size-correct to the training median codebook size for the heldout page's section/length bin by weighted top-k sampling.

M1 SECTION_FIXED_SIZE:
Weighted sampling without replacement from section family frequencies, exactly matching the training median codebook size for section/page-length bin. This is the constructive H1 selector baseline.

M2 BOOLEAN_MODULES_K:
Learn K reusable binary family modules from the page x family incidence matrix using greedy Boolean matrix factorisation on training pages.
For each training page, represent its codebook as union of <=R modules plus <=A additions and <=D deletions.
Grid frozen before outcomes: K in {8,12,16,24,32}; R in {1,2,3,4}; choose by inner whole-bifolium validation minimizing heldout symmetric-difference loss + MDL penalty.
Heldout generation is NOT allowed to fit the heldout set. Instead learn P(module-set | section,page-length-bin) from training pages with Dirichlet smoothing and sample module combinations prospectively; additions/deletions are sampled from training residual distributions.

M3 NESTED_SECTION_MODULES:
Same as M2 but modules are hierarchical: global core modules + section-specific modules. Heldout selector samples global and section modules separately.

M4 BIFOLIUM_THEN_PAGE:
Learn bifolium-level module set first from the union of its two page codebooks, then page-specific submodules/add-drop residuals. On heldout bifolium, sample one bifolium module set from training same-section/material distribution, then derive each page subset prospectively.

M5 ROW_COLUMN_TABLE:
Learn Krow row groups and Kcol column groups over families by co-occurrence spectral biclustering. A page codebook is union of one sampled row-group and one sampled column-group plus sparse residual. Grid Krow,Kcol in {4,6,8}; inner validation chooses pair. This approximates a hand-usable two-axis table.

M6 INHERITANCE_CHAIN:
Within each training bifolium/page-order convention, model next page codebook as previous-page codebook plus learned section-conditioned add/drop probabilities. On heldout first page sample M1; subsequent page inherits prospectively. No heldout codebook is observed.

M7 LOW-RANK_NMF:
Nonnegative matrix factorisation of page-family incidence with K in {4,8,12,16}. Learn section-conditioned distribution over latent coefficients; generate heldout codebook by sampling coefficients then thresholding to target size. Included as soft-module control, not historical table claim.

Evaluation of codebook prediction/generation:
For each heldout page, generate 200 prospective codebooks/model.
Metrics:
- Jaccard similarity to observed codebook;
- precision/recall/F1 of family inclusion;
- absolute codebook-size error;
- pairwise page-overlap/Jaccard distribution within heldout bifolia and same-section remote pages;
- family-frequency calibration;
- MDL description length in bits per page (module IDs + residual add/drop + selector entropy).
Primary score = expected heldout Jaccard minus training-calibrated null, with whole-bifolium sign/permutation inference.
A module model is STRUCTURALLY_USEFUL if it improves expected Jaccard over M1 by >=0.05 absolute in >=4/5 folds, z>=2, AND reduces MDL/page by >=20% relative to listing page families independently.

Forward replay gate:
For each model that is STRUCTURALLY_USEFUL, plug 100 generated codebooks per fold into the already-frozen H1 local realiser (no local-realiser changes) and recompute the August 20-metric forward battery.
A module model is CONSTRUCTIVELY_VALID only if its generated codebooks preserve >=90% of H1's joint-distance improvement over B0 and do not create a new recurring >3SD failure.

Interpretation:
- BOOLEAN/TABLE_VALID if a small K/R system passes both structural and forward gates.
- INHERITANCE_VALID if M6 passes.
- RANDOM_WEIGHTED_SET if M1 remains unbeaten.
- SOFT_LATENT_ONLY if NMF helps but discrete modules/tables do not.
No outcome licenses semantics; any successful module set may later be tested against external content.

# PAGE-CODEBOOK FACTORISATION v0.1 — COMPUTE IMPLEMENTATION NOTE BEFORE OUTCOMES
No outer-fold model result has been opened. The frozen K/R grid, outer/inner bifolium splits, selectors and decision gates are unchanged.
For computational tractability the Boolean-module NMF seed solver is stopped at 250 iterations with tolerance 5e-3 rather than waiting for numerical convergence to machine precision; the resulting component weights are used only to seed/threshold binary modules and are not themselves scored. Inner selector Monte Carlo uses 20 prospective codebook draws/page instead of 40; outer evaluation remains 200/page. This changes only numerical work, not the preregistered model class or thresholds.

# PAGE-CODEBOOK FACTORISATION v0.1 — STRUCTURAL RESULT SEALED

Universe: 204 running-text pages, 49 physical bifolia, 555 old-family types. Five whole-bifolium outer folds. 200 prospective codebook samples/page/model.

Baseline M1 section-weighted fixed-size selector:
mean heldout page-codebook Jaccard = 0.25301.

Discrete reusable-module results:
M2 BOOLEAN_MODULES: Jaccard 0.23820, delta vs M1 -0.01499, z=-3.93; mean MDL reduction only 8.5%. FAIL.
M3 NESTED_SECTION_MODULES: Jaccard 0.22608, delta -0.02725, z=-6.20; MDL worsens by 8.1%. FAIL.
M6 INHERITANCE_CHAIN: Jaccard 0.19006, delta -0.06262, z=-6.56. FAIL.
M4 BIFOLIUM_THEN_PAGE: Jaccard 0.29287, delta +0.04038, z=6.43, positive 49/49 bifolia, but below the frozen +0.05 structural-usefulness threshold. NEAR-MISS / descriptive hierarchy signal.
M5 ROW_COLUMN_TABLE: Jaccard 0.30901, delta +0.05479, z=5.94, positive 43/49 bifolia. However explicit heldout MDL audit gives only 14.9-18.7% reduction across folds (mean ~17.1%), below the frozen >=20% requirement. FAIL STRUCTURAL_USEFUL gate despite predictive signal.

Soft latent control:
M7 LOW_RANK_NMF: Jaccard 0.38596, delta +0.13227, null SD .01939, z=6.82, p<5e-5, positive 49/49 physical bifolia.
This is by far the strongest prospective codebook predictor. It is not a discrete table claim and has no binary-module MDL promotion under the frozen protocol.

Frozen v0.1 interpretation:
SOFT_LATENT_ONLY at this stage.
The page availability sets are highly non-random and contain a large reusable low-rank component, but the tested small discrete Boolean/nested/table/inheritance systems do not compress/predict them well enough under the frozen historical-table gate.
A bifolium hierarchy is real but below the material-effect threshold.
The simplest finite row/column table captures some structure but misses the MDL threshold.

Follow-up opened after primary seal:
Because M7 is the dominant soft upper bound, run a constructive forward replay using M7-generated codebooks inside the unchanged H1 local realiser. Compare prospectively against fresh M1/H1 codebooks. Use 100 paired generations/fold. Also run M5 row/column and M4 bifolium-hierarchy at 50 generations/fold as diagnostics. No local-realiser parameter may change.
If M7 preserves >=90% of H1's joint-distance improvement over B0 in >=4/5 folds, classify the missing page-plan object as LOW_RANK_REPERTOIRE_STATE rather than merely a page-frequency phenomenon. This still does not imply semantic topic or historical matrix factorisation.

# PAGE-CODEBOOK FACTORISATION v0.1 — IMPLEMENTATION AUDIT CORRECTION BEFORE FINAL INTERPRETATION

Audit found that M5 ROW_COLUMN_TABLE was executed at fixed 6x6 rather than selecting Krow,Kcol from the frozen {4,6,8}x{4,6,8} grid by inner whole-bifolium validation, and M7 LOW_RANK_NMF inherited a component count from the Boolean-module run rather than selecting K independently from its frozen {4,8,12,16} grid.

Therefore the previously sealed numerical rows for M5 and M7 are QUARANTINED for promotion/decision purposes. They remain descriptive diagnostics and are not deleted.
M0-M4 and M6 are unaffected.

Corrective rerun, thresholds unchanged:
- same five outer whole-bifolium folds;
- three deterministic inner whole-bifolium folds using the already frozen mod-inner hash convention;
- M5 grid Krow,Kcol in {4,6,8}, chosen solely by mean inner heldout expected Jaccard;
- M7 grid K in {4,8,12,16}, chosen solely by mean inner heldout expected Jaccard;
- 30 prospective draws/page in inner selection, 200/page in outer evaluation;
- no outcome-dependent threshold or model change.
Forward replay conclusions involving M5/M7 are held provisional until corrected outer models are replayed.

# SG55 FOLLOW-UP A — WITHIN-SECTION LOW-RANK LOCALISATION, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18
Reason: global K=4 components are partly associated with manuscript section and page size. Determine whether low-rank repertoire structure survives after holding the dominant section fixed rather than mistaking section differences for a four-dimensional page plan.

Primary cohort: HERBAL pages only from the exact v0.1 page-codebook corpus (largest section, 114 pages). Same physical-bifolium outer fold assignment; pages on the same bifolium remain together.
Baseline H0: weighted fixed-size sampling from HERBAL training-family frequencies, matched to training median codebook size by page-length bin.
Low-rank candidates: NMFSoft K in {2,3,4,5,6,8}; K selected separately in each outer fold by 3 deterministic inner whole-bifolium folds, 30 prospective draws/page. Outer evaluation 200 draws/page.
Primary effect: heldout expected page-codebook Jaccard versus H0; whole-bifolium sign-flip inference.
Report selected K by fold, mean Jaccard, delta, z, and component stability.
Interpretation:
- WITHIN_HERBAL_LOW_RANK if delta>=.05, z>=2, positive >=80% bifolia.
- SECTION_DRIVEN if delta<.02 or unresolved.
- INTERMEDIATE otherwise.
This is a localisation test only and cannot by itself promote a historical mechanism.
Secondary descriptive cohorts PHARMA/BIO/RECIPES may be run with fixed K=4 if >=15 pages, but are not decision-driving.

# SG55 FOLLOW-UP B — DISCRETE APPROXIMATION OF THE K=4 REPERTOIRE SPACE, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18

Motivation:
Corrected outer validation selects K=4 for the soft low-rank page-repertoire model in all 5 folds. A soft NMF is not a historical mechanism. Test how much of its heldout advantage can be retained by a finite catalogue of discrete page-plan states that a human/table procedure could in principle implement.

Outer corpus/folds/family coordinate are unchanged.

Per outer-training fold:
1. Fit K=4 NMF to training page x family incidence only.
2. Convert each training page's four nonnegative coefficients to shares summing to one.
3. Fit candidate discrete state catalogues on TRAINING coefficient shares only:
   D4: KMeans 4 states.
   D8: KMeans 8 states.
   D12: KMeans 12 states.
   D16: KMeans 16 states.
   DOM4: four states determined only by the largest coefficient.
   TOP2: six unordered top-two component-pair states.
4. For each state freeze its centroid in the original four coefficient dimensions. For heldout page generation, sample a state from the training distribution conditional on section + page-length bin, backing off to section then global. Convert centroid through the training NMF component matrix, take the highest-scoring families up to the frozen H1 target codebook size.
No heldout page incidence or coefficients are observed.

Primary benchmark:
M1 weighted fixed-size selector and the corrected continuous/sampled K=4 NMF upper bound, both with 200 draws/page.
Report heldout expected Jaccard and whole-bifolium paired delta.

Finite catalogue success:
A discrete catalogue is FINITE_STATE_REPERTOIRE if:
- Jaccard improvement over M1 >=.05 and z>=2;
- it preserves >=80% of the corrected K=4 NMF Jaccard uplift over M1;
- positive delta on >=80% physical bifolia.
The smallest successful state count wins. Ties within .01 Jaccard favor the smaller catalogue.

Constructive follow-up:
If any finite catalogue succeeds, replay its generated codebooks through the unchanged H1 local realiser for 100 draws/fold. It must preserve >=90% of the H1 forward improvement in >=4/5 folds before historical/manual interpretation is advanced.

No semantic interpretation is licensed.

# SG55 FOLLOW-UP C — HAND/REGISTER MEDIATION OF LOW-RANK PAGE REPERTOIRE, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18

Trigger:
The descriptive full-HERBAL K=2 decomposition shows strong association with Davis hand: hand-only one-hot explains about 67.7% of the two component-share variance. This association was inspected after the within-HERBAL result and before the mediation test below.

Question:
Is the low-rank page-repertoire gain merely a compressed representation of known hand/register differences, or does reusable page-level structure remain within hand/register?

Same page/family corpus and physical-bifolium outer folds.

H_HAND baseline:
Weighted fixed-size page codebook sampling conditioned on section + Davis hand + page-length bin, backing off section+hand -> section -> global. Codebook size uses the matching training median.

LR_HAND:
Fit K=4 NMF on outer-training page incidence. Training page coefficient vectors are sampled for a heldout page conditional on section + Davis hand + length bin with identical backoff. Generate top-k family set from sampled coefficient vector and training NMF H.
No heldout incidence is observed.

Primary mediation:
Heldout expected Jaccard LR_HAND - H_HAND, 200 draws/page, whole-bifolium sign-flip inference.
LOW_RANK_BEYOND_HAND if delta>=.05, z>=2, positive >=80% bifolia.
PARTIAL if .02<=delta<.05 with z>=2.
HAND_MEDIATED if delta<.02 or unresolved.

Hardest localisation:
Repeat on Herbal Davis hand S1 pages only if >=50 eligible pages, using weighted fixed-size baseline versus K selected from {2,3,4,5,6} by inner whole-bifolium validation.
If within-S1 delta>=.05 and z>=2, the page repertoire state cannot be reduced to the gross Herbal hand/Currier partition.
No historical or semantic interpretation follows automatically.

# PAGE-REPERTOIRE RECONSTRUCTION — CORRECTED STRUCTURAL RESULTS 2026-09-18

Implementation audit correction complete:
M5 and M7 were rerun with their independently preregistered inner whole-bifolium grid selections.

Corrected M7 LOW_RANK_NMF:
- selected K=4 in all five outer folds;
- outer mean heldout page-codebook Jaccard=.403187;
- M1 section-weighted baseline=.253012;
- paired whole-bifolium delta=.150180;
- null SD=.0216355, z=6.941, p<5e-5;
- positive 49/49 physical bifolia.

Corrected M5 ROW_COLUMN_TABLE:
- selected 4x8 in folds 0-2, 4x6 in folds 3-4;
- mean Jaccard=.349150;
- paired delta vs M1=.095654;
- z=6.758, p<5e-5;
- positive 49/49 bifolia;
- corrected heldout MDL reduction=20.342%, now ABOVE the frozen >=20% gate.
Therefore M5 passes the v0.1 STRUCTURALLY_USEFUL historical-table approximation gate, subject to forward replay.

Within-section localisation:
HERBAL-only (114 pages) independently selects K=2 in all five folds.
Mean low-rank Jaccard=.362449 vs weighted fixed-size=.215704;
bifolium delta=.146791, z=5.350, p<5e-5, positive 29/29 bifolia.
Thus the global low-rank effect is not merely between-section differentiation.

Hand mediation:
Conditioning the weighted baseline itself on section + Davis hand + page-length bin gives Jaccard=.264439.
A hand-conditioned K=4 low-rank selector gives .408915;
delta=.145440, z=6.970, positive 49/49 bifolia.
Hardest control, HERBAL Davis hand S1 only (84 pages), independently selects K=2 in all 5 folds and gives .360398 vs .220888 baseline; delta=.139095, z=4.555, positive 21/21 bifolia.
Therefore low-rank page-repertoire structure is not reducible to the gross section/hand/Currier register partition.

Finite-state approximation:
Using the corrected K=4 training decomposition, finite catalogue models were trained without heldout incidence.
Continuous/sampled K=4 upper bound Jaccard=.403014.
A FOUR-STATE KMeans catalogue D4 gives .405911; delta vs M1=.152830, z=6.950, positive 49/49, preserving 101.9% of the K=4 uplift.
An even simpler DOM4 rule (state = dominant one of the four repertoire components) gives .406268, preserving 102.2%.
TOP2 six-state catalogue gives .407860, preserving 103.2%.
D8/D12/D16 do not materially improve over four states.
Thus the apparent continuous four-dimensional repertoire can be discretised to only FOUR page-plan states without losing heldout predictive power.

Scientific interpretation at this stage:
The missing page-plan object is much smaller than a page-specific arbitrary family list. A reusable four-state repertoire catalogue predicts page family availability as well as the soft K=4 model; a separately learned 4x6/8 co-occurrence table also passes the structural+MDL gate.
Forward replay is still required before either is classified as a constructive manual/table mechanism.

# SG55 FOLLOW-UP D — HERBAL WHOLE-QUIRE STRESS TEST, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18

Question:
Does the within-Herbal low-rank repertoire effect survive when no page from the heldout physical quire is available during training, eliminating same-quire family/component leakage?

Primary universe:
HERBAL pages belonging to physical quires q01-q08. Quire mapping follows the manuscript's physical quire ranges. Leave one whole quire out at a time (8 outer folds).

Models:
Q0 weighted fixed-size HERBAL training-family selector, page-length-bin matched.
Q2 continuous K=2 NMFSoft, fixed from the already accepted within-Herbal bifolium result; page coefficient donor sampled only from training pages of the same length bin, otherwise all training Herbal pages.
D2 dominant-component two-state approximation: fit K=2 NMF on training quires, assign each training page to its dominant component; heldout state sampled by training length bin, then use state centroid to rank families up to Q0 target size.

No heldout family, state, page or quire information is used. Training NMF contains only families observed outside the heldout quire.
200 prospective codebooks/page.

Inference:
Per-quire mean Jaccard Q2-Q0 and D2-Q0. With only eight independent quires use all 2^8 exact sign flips.
QUIRE_ROBUST if delta>=.05, positive >=7/8 quires and exact one-sided p<=.05.
D2 is a finite two-state quire-robust approximation if it also preserves >=80% of Q2 uplift.
This is a stress/localisation test, not a new tuning opportunity.

# SG56 FOLLOW-UP — FOUR-STATE PLAN GRANULARITY, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18
Purpose: locate the physical persistence scale of the inferred four-state page plan.

Descriptive state map:
Fit K=4 NMF to the full 204-page incidence matrix only for localisation; normalise page coefficients to shares; define D4 state by 4-cluster KMeans and DOM4 by dominant component.

Primary persistence contrast:
For physical bifolia with >=2 eligible pages, calculate same-state fraction. Null permutes state labels among pages within section + Davis-hand strata while preserving global state counts and bifolium membership; 20,000 permutations.
Secondary: adjacent manuscript-page same-state rate under the same matched null.
Report D4 and DOM4.

This is localisation, not heldout evidence. A positive bifolium excess suggests a material/packet-level state component; no excess supports page-by-page selection. It cannot replace the heldout factorisation/forward tests.

# LOCAL REALISER REPAIR v0.1 — REFRACTORY R64 / MUTATION TOURNAMENT, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18

Trigger:
The corrected M5 row/column page-plan model has passed the structural+MDL gate and in forward replay improves mean joint distance over fresh H1 codebooks in all five folds while preserving >100% of the original H1 gain. Its recurrent remaining defect is exact lag-2-5 recurrence being too weak, often alongside excess lag-1 recurrence; the inherited August sequential-mutation channel is also no longer supported by the modern SG programme.

Freeze the corrected M5 page-plan selector and August family/exact-token base realiser. Change ONLY short-memory source mechanics.

Variants:
L0 CURRENT: inherited August exponential same-family exact reuse + inherited sequential ED1 mutation.
L1 NO_MUTATION: identical L0 with w_mut=0 and its mass returned to base exact-form source.
L2 R64_UNIFORM_NO_MUT: exact-reuse candidates are prior generated tokens on the same page with the selected family and overall token lag 2-64; choose uniformly; mutation off.
L3 R64_BANDED_NO_MUT: same candidates, with frozen band weights lag2-5:1.0, lag6-16:0.35, lag17-64:0.15, normalized over available candidates; mutation off.
L4 R64_BANDED_WITH_MUT: L3 exact reuse but retain inherited mutation channel.
L5 R64_TRAINED_BANDS_NO_MUT: estimate only the three relative reuse-band weights on outer training from the empirical distribution of nearest prior exact-token recurrences at lags 2-5,6-16,17-64, add-one smoothed; use those weights for candidate sampling; mutation off.

All variants retain the outer-training model's frozen total w_reuse; no heldout metric is used to tune it. R64 candidates never include lag1.

Evaluation:
100 forward generations/fold/variant, same physical skeleton, corrected M5 page-plan model, same August 20-metric real-vs-real panel.

Primary objective:
Improve exact_lag2_5_ratio residual magnitude in >=4/5 folds AND reduce pooled joint distance versus L0 by >=10%, with no new >3SD metric failure in >=3 folds.
Secondary: boundary_ed1_delta, exact_lag1_ratio, ed1_lag1_ratio, ed1_lag2_5_ratio.

If a mutation-off variant wins, classify sequential ED1 mutation as unnecessary for the best constructive realiser.
If an R64 variant wins, the modern predictive R64 discovery has a successful calibrated generative analogue; this is not a claim that predictive w=.05 is itself historical.

# PAGE-PLAN PHYSICAL LOCALISATION / WHOLE-QUIRE STRESS 2026-09-18

Within-Herbal whole-quire stress:
Primary Herbal q01-q08, leave one entire physical quire out at a time. K=2 low-rank model is fixed from prior bifolium validation; no family from heldout quire is in the training NMF unless also attested elsewhere.
Q2 continuous two-component selector beats the weighted Herbal baseline in all 8/8 heldout quires:
mean quire-level Jaccard uplift +.144038; exact 2^8 sign-flip one-sided p=.00778.
A dominant-component TWO-STATE approximation D2 also beats baseline in all 8/8 quires:
uplift +.145695; exact p=.00778; preserves 101.15% of Q2 uplift.
Therefore the within-Herbal repertoire mode is not an artefact of same-quire family/component leakage and does not require continuous coefficients.

Physical localisation:
In full 204-page K=4 state mapping, same-state membership is enriched within physical bifolia after permuting states within section+Davis-hand strata.
DOM4 same-state fraction .800 vs matched null .771, z=2.20,p=.029.
D4 same-state fraction .779 vs .733, z=3.04,p=.006.
Ordinary adjacent manuscript-page state similarity is not resolved (D4 z=.46; DOM4 z=.58).

Hardest clean subset:
Herbal Davis hand S1 (84 pages; gross hand/section fixed) independently has a K=2 repertoire decomposition. Dominant K2 state is the same across pages of a physical bifolium at rate .810 versus unconstrained matched state-count null .504; z=7.06,p<5e-5.
This state is not a monotone quire/folio sequence: both states recur across quires, while many physical bifolia are state-pure.
The two S1 states differ primarily in family edge composition (initial/final family combinations); their coarse length-bin mass is nearly identical.

Interpretation:
The repertoire-plan object has a material/bifolium-scale component rather than behaving like a simple sequential page drift. In the clean Herbal-S1 subset it can be represented as a two-state repertoire mode that generalises across completely heldout quires.
This strengthens a packet/bifolium planning interpretation but does not yet establish whether the state is selected by content, exemplar, scribal task, or another production variable.

# CORRECTED PAGE-PLAN FORWARD REPLAY / FINITE-STATE RESULT 2026-09-18

Corrected M5/M7 forward replay through the unchanged H1 local realiser:
Fresh M1/H1 page-codebook comparator mean joint distance 1.2260.
Corrected M5 row/column table: 1.0088; lower in all 5 fold means; mean absolute z 1.222 vs 1.323; mean within-2SD 16.00 vs 15.30; 96.6% of single generated corpora inside the fold real-vs-real multivariate 95% envelope.
Relative to fresh unrestricted B0, M5 preserves/exceeds the entire H1 page-codebook improvement in EVERY fold: fractions 1.328,1.053,1.383,1.219,1.250 (mean 1.246).
Thus corrected M5 passes the frozen forward-preservation gate and is CONSTRUCTIVELY_VALID as a table approximation.

Corrected continuous K4 M7: mean joint 1.0720, also lower than M1/H1 in all five fold means and preserves/exceeds the H1 gain in every fold (mean fraction 1.170).

Finite-state forward replay:
Four-state D4 catalogue mean joint distance 1.01956 vs fresh M1/H1 1.19630.
D4 is better in all 5 fold means and preserves 116.5%-125.6% of the H1 gain foldwise (mean 118.5%).
Mean absolute z improves 1.353 -> 1.276; within-2SD 15.14 -> 15.85.
No metric creates a new >3SD failure in a majority of draws in >=3 folds.
Dominant-component-only DOM4 behaves similarly: mean joint 1.03392, mean H1-gain fraction 1.167.
TOP2 six-state: 1.02965, fraction 1.174.
Additional state complexity is therefore unnecessary: FOUR states are sufficient and constructively valid.

Important residuals remain:
page-plan improvements do not close the local exact-recurrence / line-boundary / length mechanics. The corrected M5 table largely removes the previous bifolium vocabulary-overlap defect; dominant residuals are exact lag2-5 recurrence, line-boundary ED1 reset, and regime-specific token-length dispersion/page concentration.

Status:
SG56_FOUR_STATE_PAGE_PLAN = CONSTRUCTIVELY_VALID.
SG55 row/column/table approximation = STRUCTURALLY_AND_FORWARD_VALID.
Neither identifies what chooses the state. The state catalogue is a forward-working control layer, not yet a historical attribution.

# SG56 FOLLOW-UP — FOUR-COLUMN TABLE STABILITY, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-18

Question:
Can the constructively valid four-state repertoire controller be written as a stable finite ranked table rather than an opaque latent model?

For each of the five existing outer-training folds:
- fit the already frozen K=4 NMF;
- assign training pages by dominant component (DOM4);
- for each state compute the mean original coefficient vector of its assigned training pages and multiply by H to obtain one ranked score over old-family types;
- align the four state score vectors across folds by maximum cosine similarity on the common training vocabulary, using exhaustive 4! matching.

Report:
- aligned state-vector cosine stability;
- top-30, top-50 and top-75 family-list Jaccard across folds;
- fraction of top-K families common to all four state columns versus state-specific extensions;
- a consensus four-column ranked family table.

FOUR_COLUMN_STABLE if every aligned state has mean cross-fold cosine >=.80 and mean top-50 Jaccard >=.50.
This is descriptive mechanism compression; forward validity is already supplied by DOM4. Stable columns would make the constructive mechanism explicitly executable as: choose one of four columns, choose page codebook size, take top-K families, then run the local realiser.

# FOUR-COLUMN TABLE STABILITY 2026-09-18

The constructively valid DOM4 page-plan controller can be rendered as an explicit ranked four-column family table.

Across five outer-training folds, after maximum-cosine state alignment:
state mean pairwise score-vector cosines = .967, .972, .990, .987.
Mean top-50 family-list Jaccards = .760, .814, .833, .722.
All four states therefore pass the frozen FOUR_COLUMN_STABLE criterion (cosine >=.80 and top50 Jaccard >=.50).

The columns share a large master core:
top30: 10 families common to all four;
top50: 31/50 common to all four (62%);
top75: 45/75 common to all four (60%).
State-specific differences therefore behave mainly as stable priority/extensions around a common family inventory rather than four unrelated lexica.

Concrete executable compression:
choose one of four state columns -> choose page codebook size K -> take the highest-ranked K family options from that column -> run local realiser.
This four-column procedure already has independent forward validity from SG56. The table rankings themselves are highly stable under outer-fold removal.

Status:
FOUR_COLUMN_STABLE. Historical interpretation remains bounded: this is a compact executable representation of the statistical controller, not evidence that a literal four-column physical table existed.

# LOCAL REALISER REPAIR v0.1 — EXECUTION QA / PAIRED-RNG AMENDMENT, FROZEN BEFORE TARGET OUTCOMES
Date: 2026-09-19

Exact August inputs were recovered from the archived Stage-1/Stage-2 bundles before executing the already-preregistered L0-L5 tournament:
- events_raw.pkl = 34,087 events;
- 49 eligible physical bifolia;
- exact five fold loads = 6794 / 6535 / 6560 / 8040 / 6158;
- original Stage-2 gate0 passes probability sanity and target-token oracle test.
The recovered Stage-2 fold-0 training parameters reproduce alpha_family=665.3158538, alpha_token=40.5527248, w_char=.20382864, w_reuse=.12403024, w_mut=.00718979.

The corrected M5 row/column selector was reconstructed from the frozen definition and selected dimensions (4x8 folds 0-2; 4x6 folds 3-4). A 200-draw/page reconstruction QA gives pooled heldout Jaccard .34883 versus the persisted corrected .34915; difference .00032, consistent with finite sampling / set-order RNG and materially negligible. No local-repair target metric was opened in this QA.

Execution amendment for control fairness:
For each outer fold and replicate, all L0-L5 variants receive the identical prospectively sampled M5 page-codebook realization. Variant differences therefore isolate the short-memory source mechanics rather than page-plan Monte-Carlo noise. The page-plan seed is independent of the local-realiser seed. All variants retain the same outer-training model and frozen total w_reuse. No heldout target metric is used for fitting or selecting any variant.

L5 implementation detail, fixed before outcomes:
Training counts the nearest prior exact recurrence for each token event on the same page when the nearest previous identical token lies at lag 2-64, assigning it to bands 2-5, 6-16, 17-64. Add one to each band count; normalize to three relative weights. During generation each eligible same-family R64 candidate receives its band's learned weight, then candidate weights are normalized over the actually available candidates.

No other protocol element changes.

# LOCAL REALISER REPAIR v0.1 — COMPLETED 2026-09-19

Frozen context:
Corrected M5 row/column page-plan selector, August local family/exact-token base realiser, 100 forward generations per fold/variant, five physical-bifolium outer folds. Only short-memory exact-reuse / mutation mechanics changed.

Primary results (joint distance is computed from each variant's 100-generation mean metric vector against the same fold real-vs-real covariance):
L0 CURRENT mean joint=.782572.
L1 NO_MUTATION=.649189, 17.04% lower than L0, better 5/5 folds; paired-label null SD=.015515 on the absolute joint reduction .133384 (z=8.60).
L2 R64_UNIFORM_NO_MUT=.641146, 18.07% lower than L0, better 5/5.
L3 R64_BANDED_NO_MUT=.623178, 20.37% lower than L0, better 5/5 and best balanced primary variant.
L4 R64_BANDED_WITH_MUT=.766328, only 2.08% lower than L0, better 3/5; FAIL.
L5 R64_TRAINED_BANDS_NO_MUT=.676300, 13.58% lower than L0, better 5/5.

Frozen primary gate:
L1, L2, L3 and L5 pass; L4 fails. L3 is the strongest balanced winner.

Target recurrence repair:
L0 pooled exact_lag2_5 z=-2.6966 and exact_lag1 z=+2.1322.
L3 gives exact_lag2_5 z=-1.4172 and exact_lag1 z=+.7749.
Relative to L0, L3 reduces mean absolute exact-lag2-5 residual by 1.2793 z-units with permutation-null SD=.09063 (z=14.13), and reduces lag1 absolute residual by 1.2560 z-units with null SD=.08398 (z=14.96). It improves lag2-5 magnitude in 5/5 folds.

Mechanism decomposition:
Removing sequential ED1 mutation alone (L1 vs L0) produces joint reduction .133384 with null SD=.015515 (z=8.60), but does NOT resolve lag2-5 specifically: absolute lag2-5 residual reduction only .0787 with null SD=.08266 (z=.95).
Adding banded R64 on top of mutation-off (L3 vs L1) gives only .02601 additional joint reduction with null SD=.01642 (z=1.59; unresolved on total joint fit), BUT specifically repairs the recurrence geometry: lag2-5 absolute residual improves by 1.2006 z-units with null SD=.09223 (z=13.02), and lag1 by 1.1730 with null SD=.08147 (z=14.40).
Thus two distinct corrections are identified:
1. inherited sequential ED1 mutation is globally harmful / unnecessary;
2. refractory exact reuse at lags 2-64 is specifically required to reproduce the observed recurrence geometry.

Banded versus uniform R64:
L3 versus L2 total joint reduction .01797 with null SD=.01809 (z=1.00; total metric does not resolve this).
However banding improves exact-lag2-5 absolute residual by .39263 z-units with null SD=.08345 (z=4.70) and lag1 by .18919 with null SD=.06512 (z=2.90). The metric therefore resolves the band shape on the targeted recurrence coordinates even though not on total joint distance.

Outer-training empirical nearest-recurrence band estimates strongly favour longer lags (typical fold proportions about .16/.29/.55 for 2-5/6-16/17-64). L5 using those occurrence frequencies underperforms the deliberately refractory L3 weights 1/.35/.15. This is consistent with the generative selector needing per-opportunity/refractory weighting rather than copying the raw marginal recurrence histogram.

Remaining L3 residuals:
- boundary_ed1_delta remains severe in F0 (-4.32 SD) and F1 (-2.91), but not universally;
- sd_len remains too narrow in F1/F3/F4 (-2.71/-2.42/-2.56);
- exact_lag2_5 remains low in F2 (-2.93) despite major overall repair;
- ed1_lag2_5 remains low in F3 (-3.44);
- exact_adj_samefam remains low in F0 (-3.07);
- page_comp_excess remains high in F1/F3 (~+2.4/+2.2).

Status:
LOCAL_REALISER_REPAIR_R64 = PASS.
Sequential ED1 mutation is DEMOTED/REMOVED from the best constructive realiser.
Best current local-memory mechanism = mutation-off + refractory/banded exact R64 reuse (lag1 excluded).
This is a constructive generative analogue of the modern R64 finding, not evidence that the exact band weights are historically literal.

# LOCAL REALISER REPAIR v0.2 — LINE-BREAK / TOKEN-DISPERSION TOURNAMENT, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-19

Trigger:
v0.1 establishes mutation-off + refractory/banded R64 exact reuse as the best local-memory mechanism. After that repair, the largest recurring residuals are boundary_ed1_delta (especially folds 0/1) and token-length SD too narrow (folds 1/3/4). This follow-up targets those mechanisms only; page plan, R64 bands, total reuse weight, character source and all other August machinery remain frozen.

Base B0:
Corrected M5 row/column page plan + L3 R64_BANDED_NO_MUT (lag1 excluded, R64 bands 2-5:1.0, 6-16:.35, 17-64:.15).

Line-boundary variants:
B1 LB_EDGE_1X1:
At the first token of each physical line, use an outer-training line-break table P(next token first character | previous line final character, section), shrunk with existing BETA=5 to the section line-start first-character parent. Sample an old family compatible with that first-character target and the frozen page codebook. No ED1 statistic or heldout line is used.

B2 LB_FAMILY:
At line start use outer-training P(next old-family | previous-line final old-family, section), BETA=5 shrinkage to the section FIRST-family parent, again restricted to the frozen page codebook. This tests whether the full boundary-family table is needed beyond the 1x1 edge scaffold.

Token-dispersion variants:
T1 TOKEN_ALPHA4:
Identical B0 but multiply the page-local exact-token reinforcement alpha_token by 4, reducing same-page token Polya reinforcement while leaving family/page-plan and R64 unchanged.

T2 TOKEN_NOPOLYA:
Identical B0 but disable the page-local exact-token reinforcement branch entirely; exact base forms come only from the frozen outer-training token context. R64 remains active.

Combined:
C1 LB_EDGE_1X1 + TOKEN_ALPHA4.

Evaluation:
100 forward generations/fold/variant, same five outer physical-bifolium folds and same real-vs-real 20-metric covariance panels.
Report fold-ensemble joint distance, mean absolute z, within-2SD count, boundary_ed1_delta, sd_len and all secondary metrics.

Boundary success gate:
Preidentified B0 problematic folds are F0/F1 (|boundary z|>2).
A boundary variant is BOUNDARY_REPAIR if:
- absolute boundary residual is reduced by >=1.0 z-unit in BOTH F0 and F1;
- no other fold worsens boundary absolute residual by >1.0 z-unit;
- pooled ensemble joint distance is not >5% worse than B0;
- no new >3SD metric failure appears in >=3 folds.

Length success gate:
Preidentified B0 problematic folds are F1/F3/F4 (|sd_len z|>2).
A token-dispersion variant is LENGTH_REPAIR if:
- absolute sd_len residual improves by >=0.5 z-unit in at least 2/3 of F1/F3/F4;
- no fold worsens sd_len by >1.0 z-unit;
- pooled joint distance is not >5% worse than B0;
- no new recurring >3SD failure.

Combined promotion:
C1 is promoted only if it passes BOTH targeted gates and improves pooled joint distance by >=5% versus B0.

Interpretation:
B1 success with B2 unnecessary supports a minimal 1x1 line-junction scaffold rather than explicit global ED1 avoidance.
B2-only success implies a richer family-level line transition.
T1/T2 success identifies page-local exact-token reinforcement as the source of the length underdispersion.
Failure leaves the residual as unresolved and no ad hoc ED1/length correction is added.

# LOCAL REALISER REPAIR v0.1 — SEALED RESULT

No L1-L5 variant passes the frozen primary gate. The gate required lag-2-5 residual improvement in >=4/5 folds, pooled joint-distance improvement >=10% versus L0, and no new recurring >3SD failure.

L0 CURRENT under corrected M5 page plans:
mean joint distance .99417; mean absolute metric z 1.21991; mean within-2SD metrics 16.062/20.
Pooled residuals include exact_lag1 z +2.143 and exact_lag2_5 z -2.686.

L1 NO_MUTATION:
joint .91791, 7.67% better than L0; lag-2-5 magnitude improves only 3/5 folds. FAIL primary gate.
Computational matched-generation sign-flip bound: joint-distance improvement effect +.07626 with null SD .02081 (z=3.66), while lag-2-5 |z| improvement does not resolve: effect +.01663 with null SD .07622 (z=.22).

L2 R64_UNIFORM_NO_MUT:
joint .90788, 8.68% better; lag-2-5 improves 5/5 folds. FAIL only the frozen >=10% joint threshold.
Matched-generation bound: joint improvement +.08629, null SD .02173 (z=3.97); lag-2-5 |z| improvement +.69872, null SD .08320 (z=8.40).

L3 R64_BANDED_NO_MUT:
joint .91697, 7.76% better; lag-2-5 improves 5/5. FAIL >=10% joint threshold.
It produces the strongest no-mutation lag repair: pooled exact_lag2_5 z -1.453 versus -2.686 at L0 and exact_lag1 z +.738 versus +2.143.
Matched-generation bound: joint improvement +.07719, null SD .02141 (z=3.61); lag-2-5 |z| improvement +1.00556, null SD .08753 (z=11.49).

L4 R64_BANDED_WITH_MUT:
joint .98496, only .93% better; lag-2-5 improves 5/5. FAIL joint gate.
The metric does not resolve a joint improvement: effect +.00921 with null SD .01977 (z=.47). Lag-2-5 |z| improvement is +1.03316 with null SD .08433 (z=12.25).

L5 R64_TRAINED_BANDS_NO_MUT:
joint .89812, 9.66% better; lag-2-5 improves 5/5. This is the closest primary near-miss but remains below the frozen 10% gate.
Matched-generation bound: joint improvement +.09605, null SD .01992 (z=4.82); lag-2-5 |z| improvement +.53691, null SD .07704 (z=6.97).

Interpretation:
1. The directional R64 correction is real in the constructive simulation: every R64 variant improves the exact lag-2-5 residual in all 5 folds, and the no-mutation R64 variants materially reduce joint distance.
2. Full replacement of the inherited short-reuse mass by an R64 source is not admitted: none reaches the frozen 10% joint-improvement threshold.
3. Sequential ED1 mutation is not rescued as a necessary generative mechanism. Removing it alone improves joint distance in 4/5 folds, and under banded R64 the mutation-on arm loses joint distance to mutation-off in 4/5 folds. This remains a demotion/non-necessity result, not proof that no one-edit process ever occurs.
4. The persistent boundary_ed1 residual (~-2 SD pooled) is essentially untouched by R64 and remains a separate line-boundary problem.
5. The likely defect is allocation: L0 puts too much recurrence into immediate/short reuse and too little at lag 2-64; L2-L5 move all inherited w_reuse to R64 and thereby underproduce exact adjacent-same-family recurrence.

The matched-generation nulls above quantify Monte-Carlo robustness across 500 paired simulated heldout corpora; they are not independent manuscript-level inferential p-values.

# LOCAL REALISER REPAIR v0.2 — SPLIT-REUSE TEST, FROZEN BEFORE OUTCOMES

Rationale:
v0.1 shows that replacing ALL inherited exact-reuse mass by R64 moves lag1 and lag2-5 in the correct direction but does not cross the joint gate. Test the minimal two-channel allocation implied by that result.

Frozen page plan/local machinery:
corrected M5 page plan, same Stage-2 base exact-form source, mutation OFF, identical heldout skeleton and 20-metric panel.

No new heldout tuning:
Reuse the independently established outer-training Regional Persistence lambda selected on 2026-08-22: lambda_R64=.03 for folds 0,1,2,4 and .02 for fold 3.
For each fold, preserve the inherited total exact-reuse mass w_reuse:
w_R64=lambda_R64;
w_local=max(w_reuse-lambda_R64,0).
Thus this test redistributes, rather than adds, recurrence probability.

Variants:
S0 = L1 NO_MUTATION old local exponential reuse only (already sealed).
S1 = SPLIT_UNIFORM: local old exponential source with w_local + uniform same-family R64 lag2-64 with w_R64.
S2 = SPLIT_BANDED: same split, R64 candidate weights 1/.35/.15 for lag bands 2-5/6-16/17-64.
S3 = SPLIT_TRAINED: same split, outer-training three-band weights from v0.1.

All variants use identical prospectively sampled M5 page plans per replicate. 100 generations/fold.

Primary gate remains the ORIGINAL v0.1 criterion against L0, not a relaxed threshold:
- exact_lag2_5 absolute residual improves in >=4/5 folds versus L0;
- pooled joint distance >=10% below L0;
- no new >3SD metric failure in >=3 folds.
Secondary adjudication: exact_lag1_ratio and exact_adj_samefam must not become worse than L0 in >=3 folds.

If one split arm passes, admit a TWO-SCALE_EXACT_REUSE generative mechanism. If none passes, do not continue tuning recurrence weights against the heldout manuscript; move to the independent LINE_BREAK residual instead.

# LOCAL REALISER REPAIR v0.2 — LINE-BREAK / LENGTH TOURNAMENT COMPLETED 2026-09-19

Frozen base:
M5 page plan + v0.1 winner L3 mutation-off/banded R64. B0 mean fold-ensemble joint=.623178, mean abs z=.9863, 17.2/20 within +/-2SD.

Boundary table interventions:
B1 LB_EDGE_1X1 mean joint=.636876, 2.20% WORSE than B0. It fails the frozen boundary gate: in the preidentified bad folds F0/F1 the boundary residual changes -4.32 -> -4.40 and -2.91 -> -3.05 respectively, i.e. neither improves by the required 1 SD.
B2 LB_FAMILY mean joint=.666003, 6.87% WORSE. F0/F1 boundary becomes -4.84/-3.31. FAIL.
Thus neither a trained section-conditioned line-break last-char -> first-char table nor a richer previous-final-family -> next-first-family table repairs the residual. Do not add an ad hoc boundary transition layer.

Token-reinforcement interventions:
T1 TOKEN_ALPHA4 mean joint=.722278, 15.90% WORSE; sd_len does not materially improve in the target F1/F3/F4.
T2 TOKEN_NOPOLYA mean joint=.750322, 20.40% WORSE; sd_len likewise does not improve.
C1 line-edge + alpha4 mean joint=.743210, 19.26% WORSE and passes neither boundary nor length gate.
Therefore page-local exact-token Polya reinforcement is NOT the source of the remaining length underdispersion.

Decision:
All v0.2 interventions FAIL their preregistered targeted gates.
The remaining boundary_ed1_delta and sd_len mismatches should not be patched by direct line-transition or token-reinforcement knobs. The next test must determine whether these old August aggregate metrics are shadows of modern parent coordinates already identified by Super-Grammar (1x1 junction scaffold / token edge-template geometry) rather than independent production mechanisms.

This is particularly motivated by prior accepted compiler results:
- global junction dissimilation was already demoted to DERIVED_FROM_1X1_SCAFFOLD;
- local ED1 adjacency collapsed after matching line repertoire + edge state + token length;
- exact length was classified as template/edge proxy rather than an independent whole-token grammar.
No metric is to be removed from the historical battery without an explicit parent-conditioned reconciliation test on real and generated corpora.

# BOUNDARY-ED1 PARENT MEDIATION v0.1 — PREREGISTERED BEFORE OUTCOME
Date: 2026-09-19

Trigger:
The v0.2 direct line-break transition interventions fail to repair the old August boundary_ed1_delta metric. Prior Super-Grammar work separately showed global ED1/dissimilation aggregates can collapse after conditioning on edge state, token length and local repertoire. Before inventing another boundary mechanism, test whether the REAL manuscript's boundary ED1 contrast contains independent information beyond those parents.

Corpus:
Exact 34,087-token August running-text corpus. Outcome ED1(a,b) for adjacent token pairs. Boundary pairs are previous-line-final -> next-line-first within the same physical page; within-line adjacent pairs are controls.

Conditional randomisation statistic:
Within each stratum, preserve the number of boundary and within-line pairs and the pooled number of ED1 outcomes; randomise which pairs are labelled boundary. Statistic = expected boundary ED1 count under the pooled stratum rate minus actual boundary ED1 count, divided by matched boundary-pair count. Positive means extra boundary avoidance beyond the parent coordinate.
20,000 hypergeometric/random-label null draws; report effect, null SD, z, one-sided p and fraction of physical boundary pairs retained in strata containing both boundary and within pairs.

Frozen parent sets:
P0 SECTION only.
P1 PAGE + exact 1x1 junction edge (left final character, right initial character).
P2 PAGE + 1x1 edge + coarse left/right length bins.
P3 PAGE + 1x1 edge + exact left/right token lengths.
P4 PAGE + old-family pair (left family,right family), where old family = initial + final + coarse length bin.

Decision:
BOUNDARY_INDEPENDENT only if a positive residual remains >=2 null SD with >=60% matched boundary coverage under P3 or P4.
DERIVED_BOUNDARY_METRIC if P3/P4 does not resolve a residual at adequate coverage.
NON_IDENTIFIABLE if coverage <60%.
No generator rule is added from this test; it adjudicates whether boundary_ed1_delta is a legitimate independent forward target.

# LOCAL REALISER REPAIR v0.2 — SPLIT-REUSE RESULT SEALED

No split-reuse arm passes the unchanged primary gate. Recurrence-weight tuning against the heldout manuscript therefore stops here, as preregistered.

S1 SPLIT_UNIFORM:
mean joint .90713 = 8.75% below L0; lag-2-5 improves 4/5 folds, but the >=10% joint gate is missed and secondary edge/adjacency gate fails.
Matched-generation joint effect +.08704 with null SD .02049 (z=4.25). Lag-2-5 |z| improvement +.16225 with null SD .07807 (z=2.08).

S2 SPLIT_BANDED:
mean joint .91572 = 7.89% below L0; lag-2-5 improves 4/5, but joint and secondary gates fail.
Matched-generation joint effect +.07845 with null SD .01976 (z=3.97). Lag-2-5 |z| improvement +.19643 with null SD .07889 (z=2.49).

S3 SPLIT_TRAINED:
mean joint .91204 = 8.26% below L0; lag-2-5 improves 4/5, but joint and secondary gates fail.
Matched-generation joint effect +.08212 with null SD .02047 (z=4.01). The metric does not resolve the lag-2-5 robustness comparison: |z| improvement +.12697 with null SD .07776 (z=1.63).

The split models preserve more immediate recurrence but consequently repair medium-range recurrence less than the full-R64 arms. Together v0.1/v0.2 bound the issue:
- R64 is directionally useful and repeatedly repairs lag-2-5.
- The exact allocation/weighting tested here is not sufficient for frozen forward closure.
- No further heldout tuning of recurrence weights is permitted in this programme.
- The independent LINE_BREAK residual is next.

# LINE_BREAK GENERATIVE RESIDUAL v0.1 — DIAGNOSTIC FREEZE BEFORE OUTCOMES

Question:
Why does the M5 + inherited local realiser underproduce the observed boundary_ed1_delta by ~2 SD even though ED1 has already been demoted as an independent copy/mutate rule after matching line repertoire, edge state and length?

No mechanism is changed in the diagnostic stage.

For real heldout text and 100 L0 simulations/fold, decompose boundary_ed1_delta into:
A = within-line adjacent ED1 rate;
B = physical cross-line last->first ED1 rate;
delta=A-B.
Estimate natural SD/covariance of A and B from the same real-vs-real bifolium panels used by the forward battery.
Report heldout z for A, B, delta separately in every fold.

Matched boundary control:
Within each page, hold every realised line-end token fixed and permute complete line-start tokens only among lines in the same exact start-token length and start fine-morphology bucket (first raw char + final two raw chars + exact length), where bucket support permits; singleton buckets fixed.
Run 9,999 permutations on real text.
This tests whether physical pairing contributes ED1 beyond the already-supported edge/length composition.
Report effect, null SD and z.

Decision:
- If delta error is driven chiefly by B and the matched real boundary pairing itself resolves, open a direct boundary carryover/avoidance model.
- If matched real pairing does not resolve after edge/length matching, DO NOT add an ED1 boundary rule. Instead test only fine morphology/length realization at FIRST/LAST as the mediator.
- If A is the dominant error, return to within-line surface realization rather than LINE_BREAK.

This stage cannot resurrect generic ED1 copy/mutation.

# BOUNDARY-ED1 PARENT MEDIATION v0.1 — STRICT EXACT-MATCH RESULT 2026-09-19

Raw August metric:
29,990 within-line adjacent pairs, 3,893 physical line-boundary pairs.
Within-line ED1 rate=.036412; boundary ED1 rate=.018238; raw boundary_ed1_delta=.018174.

Conditional exact-stratum results:
P0 SECTION: full coverage; avoidance residual=.016522, null SD=.002734, z=6.04, p<5e-5.
P1 PAGE + exact 1x1 edge (left-final,right-initial): boundary coverage 63.58%; residual=.005637, SD=.002705, z=2.084, p=.0140.
P2 PAGE + 1x1 edge + coarse left/right length bins: coverage falls to 26.84%; residual=.003953, SD=.004244, z=.931, p=.212.
P3 PAGE + 1x1 edge + exact left/right lengths: coverage 12.48%; residual=.007127, SD=.006649, z=1.072, p=.175.
P4 PAGE + old-family pair: coverage 3.67%; residual=.028213, SD=.011379, z=2.479, p=.00345.

Frozen decision:
NON_IDENTIFIABLE under v0.1 because neither decisive parent set P3/P4 reaches the preregistered >=60% matched-boundary coverage. P4's apparent residual is explicitly not promotable at 3.7% coverage.
The useful directional fact is that adding coarse length to page+edge collapses the point estimate and z (.00564/z2.08 -> .00395/z.93), but the exact-matching design becomes too sparse to decide causality.

No boundary generator rule is added or removed from this test alone.

# BOUNDARY-ED1 PARENT MEDIATION v0.2 — SPARSITY-RESCUE CROSS-PAGE MATCHING, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-19

Trigger:
v0.1 exact matching becomes non-identifiable once PAGE is crossed with edge+length/family parents: P3 coverage 12.5%, P4 3.7%. The point estimate already collapses after adding coarse length, but this cannot be promoted at 26.8% coverage.

Goal:
Retain exact conditional randomisation while replacing exact PAGE identity with gross production-register controls, so the decisive parent coordinates have adequate overlap.

Pairs/outcome/null:
Identical to v0.1: adjacent within-line controls versus physical previous-line-final -> next-line-first boundary pairs; ED1 outcome; within-stratum hypergeometric/random-label null; 20,000 draws; positive residual means extra boundary avoidance.

Frozen rescue strata:
R0 SECTION + DAVIS_HAND.
R1 SECTION + HAND + exact 1x1 edge (left final, right initial).
R2 SECTION + HAND + 1x1 edge + coarse left/right length bins.
R3 SECTION + HAND + 1x1 edge + exact left/right token lengths.
R4 SECTION + HAND + old-family pair.
R5 SECTION + HAND + left-family + right initial character + exact right length.
R6 SECTION + HAND + left final character + exact left length + right-family.

R5/R6 are asymmetric lower-cardinality controls motivated by the physical direction of a boundary pair; they are frozen before any rescue outcome and are not decision-driving unless R3/R4 remain sparse.

Decision:
PRIMARY decision uses R3 and R4.
- DERIVED_BOUNDARY_METRIC if either R3 or R4 has >=60% boundary coverage and |z|<2.
- BOUNDARY_INDEPENDENT if either has >=60% coverage and positive z>=2, with the other adequate parent not contradicting direction.
- NON_IDENTIFIABLE if both remain <60%.
If primary remains non-identifiable, R5/R6 are reported only as directional diagnostics, not promotion evidence.

Cluster robustness:
For every adequate parent set also aggregate stratum-centred boundary residuals by physical bifolium and run all/Monte-Carlo sign flips across bifolia (20,000 if >20 bifolia). A residual is considered robust only if both pair-level conditional randomisation and bifolium sign-flip give the same positive/negative decision at p<=.05.

# BOUNDARY-ED1 PARENT MEDIATION v0.2 — CROSS-PAGE SPARSITY RESCUE RESULT 2026-09-19

Replacing exact PAGE identity with SECTION + DAVIS HAND substantially restores overlap.

R0 SECTION+HAND: 99.97% boundary coverage; residual avoidance=.016840, z=6.08; bifolium sign-flip z=4.41.
R1 + exact 1x1 edge: 91.60% coverage; residual=.006922, pair-null z=2.98,p=.00105; bifolium z=2.48,p=.00565.
R2 + 1x1 edge + coarse left/right length bins: 71.13% coverage; residual=.002252, z=.917,p=.163; bifolium z=.855,p=.215.
R3 + 1x1 edge + exact lengths: 52.38% coverage; residual=.001474, z=.526,p=.281; bifolium z=.560,p=.291.
R4 + old-family pair: 24.94% coverage; residual=-.001051, z=-.248; bifolium z=-.296.
Asymmetric lower-cardinality diagnostics R5/R6 are also unresolved (z=.80/.16; bifolium z=.93/.15).

Frozen primary decision remains formally NON_IDENTIFIABLE because R3/R4 do not reach the preregistered 60% coverage threshold.

However the mediation pattern is now very strong directionally:
raw section/hand avoidance .01684 -> edge .00692 -> edge+coarse-length .00225 (adequate 71% coverage, unresolved) -> edge+exact-length .00147 (unresolved) -> family-pair approximately zero.
This is concordant with SG30, where generic ED1 adjacency collapses after exact line+edge+length matching (raw z=.83; atomic z=.31).

Because exact matching still loses too much support, a final cross-fitted hierarchical-backoff reconciliation is required before formally demoting boundary_ed1_delta as an independent forward target.

# BOUNDARY-ED1 PARENT MEDIATION v0.3 — CROSS-FITTED HIERARCHICAL RECONCILIATION, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-19

Trigger:
v0.1/v0.2 exact matching shows the raw boundary ED1 effect collapses toward zero after edge+length/family parents, but exact high-order strata lose support. This final reconciliation removes the overlap problem by using cross-fitted hierarchical backoff rather than discarding unmatched boundary pairs.

Corpus/pairs:
Same 34,087-token August running-text corpus and same within-line adjacent / physical line-boundary pair definitions.
Outer split = frozen five physical-bifolium folds from fold_assignment.json.

Training restriction:
Parent ED1 models are fit ONLY on within-line adjacent pairs from outer-training bifolia. Boundary outcomes are never used to fit parent probabilities.

Models:
H3_EDGE_LENGTH hierarchy:
root -> section+hand -> section+hand+1x1 edge -> +coarse length pair -> +exact length pair.

H4_FAMILY hierarchy:
root -> section+hand -> section+hand+left old-family -> section+hand+left/right old-family pair.

Each child Bernoulli rate is beta-binomial shrunk to its parent:
p_child=(successes_child + alpha*p_parent)/(n_child+alpha).
Alpha is selected separately for each outer fold/model from {4,16,64,256} by 3 inner physical-bifolium folds using ONLY within-line log loss. No boundary pair participates in alpha selection.

Heldout statistic:
For every heldout within-line and boundary pair obtain the outer-training parent-only ED1 probability.
Residual avoidance = predicted_ED1 - observed_ED1.
Within each heldout physical bifolium compute:
delta_bif = mean residual_avoidance(boundary pairs) - mean residual_avoidance(within-line pairs).
Positive means boundary pairs are more ED1-avoiding than their parent coordinates predict.
All heldout pairs receive predictions via hierarchical backoff, so boundary coverage is 100%.

Inference:
Primary statistic = unweighted mean delta_bif across the 49 physical bifolia.
20,000 exact/Monte-Carlo sign flips of the 49 bifolium deltas.
Also report token-pair-weighted global delta and outer-fold deltas.

Decision:
BOUNDARY_INDEPENDENT if BOTH H3 and H4 have positive mean delta >=.005, z>=2 and one-sided p<=.05.
DERIVED_BOUNDARY_METRIC if BOTH H3 and H4 have |mean delta|<.005 OR are unresolved (|z|<2), with neither showing a contradictory resolved positive effect.
MIXED if one resolves positive and the other does not.
This test decides whether boundary_ed1_delta remains an independent forward target; it does not alter the already accepted general 1x1 junction scaffold.

# BOUNDARY-ED1 v0.4 — MARGINAL-PRESERVING LINE-PAIRING NULL, PREREGISTERED BEFORE OUTCOME
Date: 2026-09-19

Reason:
v0.3 cross-fitted within-line parent models can overpredict sparse boundary combinations because boundary pairs occupy different line-position marginals. The scientifically direct question is whether the ACTUAL previous-line-final -> next-line-first pairing is more ED1-avoiding than would arise from the independently observed LAST-token and FIRST-token marginals themselves.

Primary null:
Within each physical page, freeze:
- every line-final token used on the left of a boundary;
- the multiset of next-line FIRST tokens used on the right;
- page, section, hand, number of lines, token identities, lengths, edge morphology and page repertoire.
Randomly permute the right-side FIRST tokens among the page's boundary slots. This destroys only which line end is paired to which next-line start.
20,000 deterministic permutations.

Statistic:
expected shuffled boundary ED1 rate - actual boundary ED1 rate.
Positive means an independent cross-boundary avoidance relation beyond LAST/FIRST marginal morphology.
Report actual rate, null mean/SD, z and one-sided empirical p.

Hostile geometry null:
Repeat the permutation only within page x vertical-quartile boundary groups, where quartile is based on boundary's normalized line index. This preserves coarse vertical trajectory as well as all page/token marginals. Groups of size <2 are fixed.
Report movable-boundary fraction.

Decision:
INDEPENDENT_BOUNDARY_INTERACTION only if primary and vertical-quartile null both give positive z>=2, p<=.05, with >=60% movable boundaries in the hostile null.
DERIVED_FROM_LINE_POSITION_MARGINALS if primary |z|<2, or if primary resolves but hostile geometry null is unresolved at adequate movable coverage.
NON_IDENTIFIABLE if hostile movable coverage <60% and primary resolves.
This test supersedes v0.3 for the question of a dedicated cross-boundary relation because it preserves the exact observed line-final and line-first token distributions rather than predicting them from sparse parent classes.

# BOUNDARY-ED1 v0.3/v0.4 — FINAL RECONCILIATION 2026-09-19

v0.3 cross-fitted hierarchical backoff (recorded for audit, NOT final):
Training only on within-line adjacency, hierarchical edge+length and family parent models predicted a strong positive boundary residual:
H3 EDGE_LENGTH mean bifolium delta +.01420, z=4.04;
H4 FAMILY +.01499, z=4.44.
However a support audit showed the effect is dominated by the exact parent combinations least represented among within-line training pairs: for H3, 2,072/3,893 boundary pairs had zero exact-leaf training support; for H4, 3,080/3,893 had zero exact family-pair support. The model therefore extrapolates across a severe LAST/FIRST covariate shift and systematically overpredicts boundary ED1. v0.3 is QUARANTINED as a sparse-support predictive extrapolation, not accepted as causal evidence.

v0.4 marginal-preserving pairing null directly tests the cross-boundary relation without predicting token marginals:
Within each physical page, freeze every observed line-final token and the complete multiset of next-line first tokens, and shuffle ONLY which final is paired with which start.

Observed actual boundary ED1 rate = .0182379 (71/3893).
Primary within-page shuffle null:
mean ED1 rate=.0138701, SD=.00175985.
Expected-minus-observed=-.0043678, z=-2.482.
One-sided p for extra avoidance=.99275.
Thus the actual line-to-line pairing is NOT unusually ED1-avoiding. If anything it contains MORE ED1 pairs than random pairing of the same page's line-final and line-start token marginals.

Hostile page x vertical-quartile shuffle:
98.46% of boundary slots movable.
null mean=.0163985, SD=.00170724;
expected-minus-observed=-.0018393, z=-1.077, avoidance p=.87596.
After preserving coarse vertical trajectory, the actual pairing is completely unresolved relative to the marginal null.

Frozen v0.4 decision:
DERIVED_FROM_LINE_POSITION_MARGINALS.

Interpretation:
The old August boundary_ed1_delta compares within-line adjacency to LAST->FIRST cross-line adjacency and therefore conflates a relation with the very different marginal morphology of line-final and line-initial tokens. Once the exact observed LAST/FIRST marginals are preserved, there is no independent ED1-avoidance relation across the line break.
The earlier direct line-break generator patches failed for the same reason: there is no missing boundary-avoidance transition to add.

Production consequence:
boundary_ed1_delta remains reportable as a surface phenotype but is retired as an independent closure/generator target. Future forward closure should judge line-position morphology through its parent edge/family/line-position coordinates, not require an explicit ED1 boundary repulsion.

This reconciles with SG30: generic ED1 adjacency is a reproducible phenotype but parent-mediated by edge/length geometry.

# MECHANISM-CLASS BLINDED FORWARD TOURNAMENT v1.0 — PREREGISTERED BEFORE OUTCOMES
Date: 2026-09-19

Scientific question:
Has the constructive evidence become specific enough that a generated/table-driven geheimschrift should be the primary working production class, with conventional/nomenclator-like cipher mechanisms retained as controls rather than the default explanation?

Scope and epistemic bound:
This tournament compares surface-production mechanisms. It CANNOT prove absence of plaintext or semantics. A stateful code system could still be driven by meaningful source material. "Generated geheimschrift" here means a deliberately produced opaque writing stream in which page repertoire and local surface rules are causal parts of production, rather than a narrow stateless substitution/transposition cipher.

Frozen corpus/evaluation:
- exact August 34,087-token running-text corpus;
- five outer physical-bifolium folds;
- same physical heldout page/line skeleton;
- same old family coordinate (first, last, coarse length);
- sequential ED1 mutation is OFF except in its explicit control hypothesis;
- direct boundary_ed1_delta is RETIRED from the PRIMARY closure vector following SG61; it remains reported as a secondary descriptive phenotype.
- Primary vector = the other 19 August metrics, with real-vs-real covariance recomputed in 19D separately in each fold.
- Secondary full 20-metric output is retained for audit.

Monte Carlo:
100 independent forward generations per fold/hypothesis = 500 per hypothesis, 5,000 generated heldout corpora total.
No retuning after outcomes.
Primary score for each hypothesis:
1. joint Mahalanobis distance of its 100-generation fold mean in the 19D parent-aware battery;
2. single-run joint-distance distribution;
3. mean absolute z and count within +/-2 SD;
4. 5-fold consistency;
5. pairwise probability of beating the current constructive baseline using matched replicate index.

Outcome blinding:
The ten mechanisms are executed and stored only as labels A-J. The mapping below is sealed now. Intermediate per-hypothesis outcomes are not inspected until all five folds have completed.

A CURRENT_M5_R64:
Corrected M5 row/column page plan, contextual family realiser, page-local exact-token reinforcement, mutation OFF, refractory banded R64 exact reuse (lags 2-5/6-16/17-64 weights 1/.35/.15). This is the current best constructive model.

B FOUR_COLUMN_DOM4_R64:
Four-state DOM4 page plan learned only on outer training: K=4 NMF, state=dominant component, state sampled by section+page-length bin, deterministic top-K families from state centroid. Same local realiser/R64 as A.

C BIFOLIUM_SHARED_DOM4_R64:
As B, but one repertoire state is sampled per physical bifolium and shared by all eligible pages on that bifolium; page-specific K still sets codebook size. Tests literal material/packet planning.

D CORE_EXTENSION_DOM4_R64:
As B but codebook is explicitly rendered as a shared master core (intersection of the four state top-50 training rankings) plus state-ranked extensions to K. Tests a compact human table with common core + four extension modes.

E FOUR_COLUMN_NO_R64:
As B but no dedicated exact-reuse source; mutation remains OFF. Tests whether the page-plan table alone is sufficient.

F LOCAL_R64_NO_PAGEPLAN:
No page repertoire restriction; ordinary contextual family selection + page token reinforcement + mutation-off banded R64. Tests whether local generation alone can substitute for the page plan.

G MARKOV_ONLY:
No page plan, no page-local family/token Polya reinforcement, no exact R64 and no mutation. Families are drawn from the frozen section/hand/line-position/previous-family transition model; exact forms from frozen outer-training token contexts. This is a sequential source/cipher-like surface control.

H NOMENCLATOR_PAGE_IID:
Four-column page codebook, but within the page families are sampled independently from outer-training section frequencies restricted to the codebook; exact forms sampled independently from outer-training family vocabularies. No page Polya, R64 or mutation. This approximates a page-specific nomenclator/codebook without a generated local grammar.

I FINITE_DECK_DOM4_R64:
Four-column codebook; at page start pre-draw one family card for each non-character token slot from the state-ranked codebook distribution and consume cards without replacement, while retaining mutation-off banded R64 exact-form reuse. Tests a literal finite preplanned deck/quota procedure.

J COPY_MUTATE_DOM4:
Four-column page plan with the inherited August short-memory exact-reuse + sequential ED1 mutation mechanism rather than refractory R64. Historical pseudo-text/copy-mutate control.

Decision framework:
GENERATED_GEHEIMSCHRIFT_PRIMARY_WORKING_CLASS if:
- at least one of A-D/I beats both G and H by >=20% pooled 19D joint distance and in >=4/5 folds;
- removing BOTH the page-plan layer (F/G) and local generated-memory layer (E/H) materially worsens forward fit;
- a compact table model B/C/D remains within 10% of or improves A;
- results are Monte-Carlo stable: P(best generated mechanism beats G)>=.90 and P(beats H)>=.90 pooled, >=.75 in every fold.

This does NOT set P(cipher)=0. It changes the default mechanism label from "cipher" to "generated/table-driven geheimschrift candidate". A cipher remains live only in a broader stateful sense if plaintext/source choices drive the repertoire/control states.

If generated mechanisms do not satisfy the above, retain mechanism-class agnosticism.

# MECHANISM-CLASS TOURNAMENT v1.0 — POST-UNBLIND ROBUSTNESS ADDENDUM, FROZEN BEFORE BOOTSTRAP OUTCOMES
Date: 2026-09-19

The ten A-J 100-draw/fold outcomes have now been unblinded. The preregistered individual-run dominance criterion is retained exactly and will not be redefined: A CURRENT_M5_R64 has pooled P(single draw beats H NOMENCLATOR_PAGE_IID)=.752, below the frozen .90 criterion, although its mean 19D joint distance is substantially lower.

For comparability with the earlier generator programmes, run one secondary robustness calculation that does NOT rescue or replace the primary gate:
- within each fold/model, bootstrap 10,000 independent ensembles of 8 generated corpora from the already frozen 100 draws;
- recompute the 19D metric-vector mean and its Mahalanobis joint distance, rather than averaging single-run distances;
- report P(A ensemble beats G), P(A beats H), and the same for compact table models B/C/D;
- report 95% bootstrap intervals for A-H and A-G joint-distance differences.
No model or threshold changes.

# MECHANISM-CLASS BLINDED FORWARD TOURNAMENT v1.0 — UNBLINDED RESULT 2026-09-19

Design:
Ten preregistered A-J production mechanisms; 100 forward generations per fold/hypothesis, five physical-bifolium folds = 5,000 heldout generated corpora. Primary covariance battery contains 19 independent/parent-aware August metrics, with boundary_ed1_delta retained only descriptively after SG61.

Blind ranking by mean 19D fold-ensemble joint distance:
1 A CURRENT_M5_R64 = .667239
2 C BIFOLIUM_SHARED_DOM4_R64 = .717418
3 D CORE_EXTENSION_DOM4_R64 = .723366
4 B FOUR_COLUMN_DOM4_R64 = .727152
5 J COPY_MUTATE_DOM4 = .856824
6 E FOUR_COLUMN_NO_R64 = .884496
7 H NOMENCLATOR_PAGE_IID = 1.004582
8 I FINITE_DECK_DOM4_R64 = 1.308185
9 F LOCAL_R64_NO_PAGEPLAN = 1.963540
10 G MARKOV_ONLY = 2.282608

All four top positions were occupied, while still blinded, by page-plan + generated-local-mechanics models.
A also has the best mean absolute metric z (.927) and 17.4/19 primary metrics within +/-2SD on average.

Mechanism contrasts:
A vs MARKOV_ONLY G: 70.77% lower pooled joint distance; A lower in all 5 fold means. Single-run P(A beats G)=.998, fold probabilities .99-1.00.
A vs page-specific NOMENCLATOR/IID H: 33.58% lower pooled joint distance; A lower in all 5 fold means. Single-run P(A beats H)=.752, so the preregistered strict .90 individual-realisation dominance criterion is NOT met.
The failure is stochastic-overlap, not equality of expected fit: fold-mean A improvements over H are 28.9%,24.3%,48.6%,19.7%,57.1%.

Compact manual/table representations remain near-optimal:
C bifolium-shared four-state = 7.52% above A.
D explicit common-core + four extensions = 8.41% above A.
B plain four-column DOM4 = 8.98% above A.
All are within the preregistered 10% compact-table bound.

Layer necessity:
E FOUR_COLUMN_NO_R64=.8845: table/page plan alone is materially worse than A; local recurrence machinery adds value.
F LOCAL_R64_NO_PAGEPLAN=1.9635: local generated mechanics without a page plan are dramatically worse.
G MARKOV_ONLY=2.2826: sequential transition structure alone is worst.
H NOMENCLATOR_PAGE_IID=1.0046: page repertoire alone does surprisingly large work, but IID code-group selection has systematic failures in page composition, family adjacency and local recurrence.
I FINITE_DECK=1.3082: literal preplanned quota/deck is too rigid.
J COPY_MUTATE=.8568: old copy-mutate mechanism is clearly inferior to refractory R64.

Post-unblind fixed ensemble-8 bootstrap (secondary, not a rescue of the primary single-run criterion):
A vs G: P(A beats G)=1.000 in every fold.
A vs H: mean fold P=.937; minimum fold=.783. Thus at the established multi-generation model-evaluation scale the generated model meets the >=.90 pooled / >=.75 every-fold stability criterion.
Compact B/C/D all beat G with probability 1.00 in all folds; against H their ensemble-8 probabilities are lower and more variable (.684-.9998), consistent with A's additional fit from the M5 page-plan flexibility.

Decision:
GENERATED_GEHEIMSCHRIFT_PRIMARY_WORKING_CLASS__NOT_HISTORICAL_IDENTIFICATION.

Rationale:
The data now support using a generated/table-driven geheimschrift as the primary WORKING production class for ordinary running text:
material/page repertoire selection is causally necessary for forward adequacy; a compact 4-state/four-column representation remains near-optimal; refractory exact reuse is independently necessary; Markov-only and IID nomenclator-like controls are substantially worse; copy-mutate and finite-deck alternatives lose.
This is a change in model default, not proof of semantic nullity.

Critical bound:
The result does NOT exclude plaintext, meaning or a cipher in a broad stateful sense. A meaningful source could choose bifolium/page repertoire state, codebook entries, or control decisions. What is no longer a good default is a narrow conventional cipher in which the visible Voynich surface is merely a passive substitution/transposition of source text. Any surviving cipher hypothesis now needs to explain why the encryption procedure itself contains the empirically required page/bifolium repertoire controller and generated local recurrence mechanics; at that point it is operationally a generated code/geheimschrift system.

Remaining independent mismatch in the best A model:
sd_len remains the largest recurrent residual (mean z -1.96; F1/F3/F4 below -2). Remaining recurrence residuals are smaller: exact_lag2_5 mean -1.52 and ed1_lag2_5 -1.38. Boundary ED1 is no longer an independent target.


SUPERSEDED 2026-09-19: the explicit PAGE_CODEBOOK causal layer is removed by the preregistered emergent-occupancy deletion test. Canonical successor: voynich_supergrammar_v11_running_text_20260919. Local SG1.0 grammar components otherwise inherited.
