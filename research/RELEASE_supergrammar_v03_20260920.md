# Voynich Super-Grammar v03 — Certified Kernel Release

**Date:** 2026-09-20  
**Release ID:** `supergrammar_v03_release_20260920`  
**Status:** `FROZEN_CERTIFIED_KERNEL`  
**Certification standard:** `FORTUNE_STYLE_FAIL_CLOSED_V1`  
**Release-manifest SHA256:** `c4bbace849cf1cc79ff3999f02a2031ddc888bc3aa222a78a2174c3b6253027e`

## Authority and scope

This release supersedes legacy `CORE` / `SUPPORTED` labels as scientific authority. No claim is exported merely because it appeared in the v01/v02 candidate graph. The authoritative chain is the canonical SG1.1 surface architecture plus Theory Compiler Stages B–E, followed by the v03 claim/obligation/certificate layer and the independent clean-room replays recorded below.

The release contains **9 certified theorem wrappers** backed by **32 certified or certified-bounded claim certificates**. Two atomic claims failed clean-room replay. Five older kernel candidates remain explicitly **excluded as uncertified** rather than being grandfathered into the grammar.

This is a certified structural grammar kernel for Voynich running text under the tested representations and folds. It is not a decipherment, semantic model, historical production proof, cipher identification, or coverage of labels/circular/radial loci.

## Certified theorem kernel

### SGT01 — Local form grammar — CERTIFIED_BOUNDED

Running-text token forms require at least second-order within-token context and obey representation-robust graphotactic zero constraints.

Independent replay: the weakest order1→order2 condition across six transcription layers, both directions and four frozen smoothing values has effect **0.231214 bits/event**, null SD **0.040103**, ratio **5.77**. All 19 frozen hard-zero bigrams remain zero across all six layers; the weakest matched-null ratio is **3.55**.

Third-order context is **not** included by this theorem.

### SGT04 — Repertoire bound — CERTIFIED_BOUNDED

The registered explicit preselected page-family codebook is not part of the certified architecture.

Preregistered SG1.1 deletion: no-codebook A4 is adequate in **4/5** frozen folds and lies inside empirical d95 in **5/5**; explicit-codebook A3 is adequate in **0/5**. Mean joint distance is **1.140162** without the codebook versus **3.060952** with it.

This rejects that registered explicit codebook model class; it does not reject every possible page-scale controller.

### SGT05 — Boundary ED1 bound — CERTIFIED_BOUNDED

Physical-line ED1 attenuation is not certified as an independent avoidance rule after marginal and coarse vertical-geometry preservation.

The hostile page+vertical-quartile shuffle gives expected-minus-observed effect **-0.001822**, null SD **0.001704**, ratio magnitude **1.07**, at **98.46%** movable coverage. The metric does not establish independent avoidance.

### SGT06 — Currier identifiability bound — CERTIFIED_BOUNDED

Currier A/B may mark descriptive distributional differences, but the manuscript does not identify Currier as independent of both section and Davis hand.

Currier-versus-hand effect is **0.005258 bits/token**, null SD **0.004419**, ratio **1.18**: the metric does not resolve this. No observed section+Davis-hand stratum contains both Currier A and B.

### SGT07 — SPACE versus LINE_BREAK attenuation — CERTIFIED

Across the frozen transcription family and smoothing grid, the local edge dependency is stronger across within-line SPACE than across physical LINE_BREAK.

The weakest clean-room SPACE-minus-LINE_BREAK condition has effect **0.101343 bits/event**, null SD **0.047498**, ratio **2.13**.

### SGT08 — Order-3 is not core — CERTIFIED_BOUNDED

Third-order within-token context is excluded from the certified core because its incremental gain is smoothing-dependent.

Example: ZLZI at alpha=16 gives LTR effect **-0.000987**, null SD **0.002475**, ratio **0.40**; RTL gives **+0.003715**, null SD **0.002560**, ratio **1.45**. Every transcription layer contains at least one frozen smoothing/direction condition where positive order-3 gain fails the release gate.

### SGT09 — Current-parent R64 recurrence — CERTIFIED_BOUNDED

Under the current SG1.1 **no-codebook** parent, a banded same-page exact-reuse channel over lags 2–64 improves the targeted lag-2–5 recurrence geometry.

Banded R64 versus no reuse: effect **0.005672**, null SD **0.002538**, ratio **2.23**, positive **5/5** folds and better in **100%** of paired generated runs.

Banded versus uniform R64: effect **0.002084**, null SD **0.000938**, ratio **2.22**, positive **5/5** folds.

A requirement to exclude lag 1 is **not** established: lag2–64 versus lag1–64 joint contrast is **-0.000328**, null SD **0.000211**, ratio **1.55**. The metric does not resolve lag-1 exclusion.

### SGT10 — Conditional graphotactics — CERTIFIED_BOUNDED

Two local conditional effects survive independent folio-heldout replay.

For q-conditioned k/t selection, the exact-continuation-tail matched ZLZI effect is **0.023703 bits/event**, null SD **0.004965**, ratio **4.77**. Four of five alternate layers resolve positive; JSLI is positive but unresolved at **0.000917 / 0.002157 = 0.43**. No alternate layer gives a resolved negative contradiction.

For i-run-length-dependent termination, both preceding-1 and preceding-2 context bounds resolve in the primary layer and in all five alternates. The weakest reported replay ratio is **3.52**.

### SGT11 — Junction morphology — CERTIFIED_BOUNDED

The running-text junction architecture contains a small, representation-robust previous-token morphology/length residual together with the SPACE>LINE_BREAK edge scaffold.

Primary ZLZI morphology effect is **0.007355 bits/token**, null SD **0.001914**, ratio **3.84**. Four of five alternate layers resolve positive and none resolves negative; JSLI remains positive but unresolved.

The earlier claim that exact previous-token identity is completely closed after the morphology parent is retracted. TTLI retains a small positive exact residual: **0.001036 bits/token**, null SD **0.000382**, ratio **2.72**. The frozen exact-closure criterion passes only **2/6** transcription layers.

The correct conclusion is asymmetric: universal exact closure is false, but exact-token identity is not promoted as a universal transition layer.

## Superseded theorem wrappers

- `SGT02_JUNCTION_GRAMMAR` is superseded by `SGT11_JUNCTION_MORPHOLOGY` because universal exact-identity closure failed clean-room replay.
- `SGT03_SHORT_MEMORY` is superseded by `SGT09_R64_CURRENT`, which retests exact recurrence under the current SG1.1 no-codebook parent.

## Explicitly excluded uncertified kernel candidates

These remain stored for future work but are **not exported as established grammar**:

1. `SGK10_SHORT_STATE_LAG1` — local paragraph subword-state lag-1 similarity.
2. `SGK11_SHORT_STATE_LAG2` — local paragraph subword-state lag-2 similarity.
3. `SGK12_REMOTE_STATE_REUSE` — split-heldout remote subword-state recurrence.
4. `SGK13_R64_RECURRENCE_GEOMETRY` — legacy pre-SG1.1 R64 formulation; replaced for current architecture by SGT09.
5. `SGK14_SEQUENTIAL_MUTATION_REJECTED` — legacy dedicated sequential-ED1 mutation rejection; not independently replayed to v03 standard.

Exclusion is deliberate fail-closed behaviour, not evidence that these claims are false.

## Prominent retractions and corrections

1. The old candidate ledger is not current authority.
2. Explicit preselected PAGE_CODEBOOK causality is **DATA_KILLED**.
3. Third-order within-token context is not smoothing-robust core grammar.
4. The 53-super-rule v02 graph is not a complete current authority.
5. Legacy `CORE` / `SUPPORTED` labels are not certification.
6. “132 evidenced nodes” is retracted.
7. Legacy HOLD frequency does not demonstrate strength-based promotion.
8. Lag 1 is not proven to require exclusion from the exact-reuse window.
9. Complete exact-previous-token closure after SG45 morphology is retracted; a small representation-sensitive TTLI residual survives.

## Independent verifier ledger

Completed scientific verifier outputs:

- Atomic surface/graphotactic replay: Git `82caeb96df395035528eb43c387ffec7895d10af`; script SHA256 `d0dcd9a24ad3157ee2ea9cdc4cfce7844499afc02e91612244b4b9c84a799b15`; result SHA256 `6b235bd6002d8f51823181b6766061056063384332a6cdc3262bb5d0f9d4b41a`.
- Current-parent R64 tournament: Git `23ff233e315bd932dd0777fd90ed41b7f0036cf9`; script SHA256 `f55b309d5801fefa98991d79ec991d20e61aaef06d2ca7cd6ec38f14735d2306`; result SHA256 `a282e222a8f83c20091e39d5eca85215f4ec07b4dfc79902d5b8543b95a7d0e8`.
- Conditional graphotactics replay: Git `c42a80faac061bab84e773cf282f1ad3b25085cb`; script SHA256 `5615eb45b72aa5aba763d9df38d2516f8b3dd361f2b477e6e8070f831354ccda`; result SHA256 `15ae35e720ace4dff1475140cdd1eeb737b43f841cd648d573191c3ac6fa8e5a`.
- Closure/junction replay: Git `a959a95a3c3dfae5d5dd6a1c7c019052250049ee`; script SHA256 `da33220eff309db753877bbe0618a787344257f74fa2398016ab4e10604180a7`; result SHA256 `774819b6b2f8cfb4d8957011ad0a418ed9b2545ce5f10b2bf57f4f2fb09d0fc4`.

Pre-outcome implementation failures are retained in the database verifier ledger and are not deleted or counted as evidence.

## Canonical source hashes

- Corpus SHA256: `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`
- Canonical event rows SHA256: `74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f`
- Canonical fold assignment SHA256: `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`

## Validation and export surfaces

All frozen release invariants pass:

- canonical authority chain: PASS
- all required obligations on certified claims closed: PASS
- certified theorem dependency closure: PASS
- no failed claim appears as a required positive premise: PASS
- active verifier jobs: 0 / PASS
- v03-specific Supabase security adviser findings: 0
- v03-specific Supabase performance adviser findings: 0

All v03 physical tables have RLS enabled. No client-facing RLS policy is created; the release remains service-role/private rather than being accidentally exposed.

Authoritative database surfaces:

- `public.vms_supergrammar_release_v03` — frozen cryptographic release manifest.
- `public.vms_supergrammar_certified_theorems_v03` — export only certified theorem wrappers.
- `public.vms_supergrammar_theorem_dependencies_v03` — theorem-to-atomic-lemma proof graph.
- `public.vms_supergrammar_open_obligations_v03` — outstanding/failed proof obligations.
- `public.vms_supergrammar_retractions_v03` — prominent correction/retraction ledger.
- `public.vms_supergrammar_verifier_runs_v03` — immutable execution/provenance ledger.
- `public.vms_supergrammar_validation_v03` — fail-closed release checks.

## Release rule

Future work may add new candidate claims, experiments, or stronger theorem wrappers, but it must not mutate this frozen release into a stronger claim by reinterpretation. A future promotion requires a new release ID, explicit dependencies, closed obligations, and a new manifest hash.
