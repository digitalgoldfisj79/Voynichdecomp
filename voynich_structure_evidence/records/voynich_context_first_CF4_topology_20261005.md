# Voynich context-first CF4 topology-matched audit — 2026-10-06

- key: `voynich_context_first_CF4_topology_20261005`
- status: completed
- created: 2026-10-06T04:37:32.984895+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# VOYNICH CONTEXT-FIRST CF4 — TOPOLOGY-MATCHED AUDIT
Date: 2026-10-06
Status: COMPLETED / MIXED REPLICATION / BRANCH CLOSEOUT
Code commit: 4d4d9ba742b674065f76b99cc894de26f3e09fea
Script: research/context_first_CF4_topology_audit_20261005.py
Original interrupted HF job: 6ac41921404719ba376596f4
Resume HF job (ZLZB+TTLI only): 6ac420e1404719ba37659c6d

Purpose:
Audit CF context-first selected-pair results against a topology-matched null that preserves the exact selected graph degree/repeated-vertex structure while matching frequency-bin + dominant-section nuisance buckets. Also enumerate maximum disjoint matchings to test whether hub/repeated-vertex structure is necessary.

ZLZI:
Full graph cross obs .0365142, null mean -.00276287, null SD .0202068 => 1.94376 SD. The metric does not resolve direct heldout fingerprint similarity.
Full graph pool obs .01069045, null mean -.00741325, null SD .00534226 => 3.38877 SD.
Pool positive in both final folds: fold0 .01250653, fold1 .00933821.
Maximum disjoint matching size 7, two alternatives.
Disjoint cross z range 1.06965–2.39732.
Disjoint pool z range 2.54453–3.63280.
Interpretation: ZLZI pooling survives topology correction and survives removal of shared vertices; direct fingerprint-similarity claim is weaker.

ZLZB:
Full graph cross obs .03644440, null mean -.00344725, null SD .01864892 => 2.13909 SD.
Full graph pool obs .01073746, null mean -.00727779, null SD .00521881 => 3.45198 SD.
Pool positive in both final folds: fold0 .01266466, fold1 .00930250.
Maximum disjoint matching size 7.
Disjoint cross z range 1.02536–2.39821.
Disjoint pool z range 2.68413–3.29903.
Interpretation: essentially replicates ZLZI, but ZLZI/ZLZB are closely related transcription representations and should not be counted as independent corpora.

TTLI:
Full graph cross obs .07004850, null mean .00048433, null SD .01884522 => 3.69134 SD.
Full graph pool obs .00762932, null mean -.00213412, null SD .00461024 => 2.11777 SD.
Pool positive in both final folds: fold0 .00545037, fold1 .00988751.
Maximum disjoint matching size 5 across 12 alternatives.
Disjoint cross z range 1.19666–4.19860.
Disjoint pool z range -0.16080–1.49965.
Interpretation: TTLI confirms a strong context-similarity graph and a modest full-graph pooling effect, but pooling does NOT survive removal of shared vertices. The pooling signal in TTLI is therefore at least partly network/hub-dependent.

Final adjudication:
SUPPORTED:
1. Sparse context-derived relationships among exact Voynich token types survive a topology-matched audit.
2. Direct heldout context similarity is strong in TTLI and modest/fragile in ZLZI/ZLZB.
3. Full selected-graph pooling is positive in all representations and positive in both final folds.
4. ZLZI/ZLZB disjoint-pair pooling survives strongly after removing shared vertices.

NOT SUPPORTED / RETRACTED AS TOO STRONG:
1. Do not claim a general smaller hidden “functional vocabulary”.
2. Do not claim context-selected pairs are independently interchangeable lexical equivalents.
3. TTLI disjoint-pair pooling does not resolve: all 12 maximum-match pooling z scores are below 2, range -0.16 to 1.50.
4. The robust object may be a sparse context graph with hub/network structure rather than a set of independent synonym/homophone pairs.

Relation to prior post-NK work:
This substantially overlaps the prior CF3N/post-NK pooling result and should be treated as a stricter audit/confirmation, not a new discovery. Earlier broad morphology/FORM/ED family hypotheses remain rejected. Earlier masked-novel CF4 also failed to establish a productive latent-family system for unseen forms.

Programme consequence:
Close further “find context-similar pairs and pool them” variants. Next discriminating test should bridge to the SELECT/state programme: freeze these context relations and predict which surface member/neighbor is realized from renderer/state, morphology, hand/physical metadata, and upstream SELECT variables on held-out bifolia.
