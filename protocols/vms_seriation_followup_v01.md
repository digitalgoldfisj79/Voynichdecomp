# VMS blind-seriation follow-up v0.1 — FROZEN BEFORE OUTCOMES

Date: 2026-09-07
Protocol ID: `vms_seriation_followup_20260907_v01`

Purpose: distinguish a genuine local production-order signal from latent production-state clustering after the blind bifolium neighbourhood graph qualified.

No new total-order optimisation is permitted.

## F1 — q01_b3_6 ↔ q03_b17_24 EVA-m ablation

Threat: prior Voynich work noticed a Currier-A/high-EVA-m subtype involving these regions. The fully promoted edge could therefore be driven primarily by that morphological subtype rather than finer production proximity.

Frozen ablations, evaluated independently on all four ZL/RF/IT/GC transcriptions:

1. `DROP_M_TOKEN`: remove every token containing `m` before constructing LEX/C3/SHAPE.
2. `MASK_M`: replace every `m` with a neutral placeholder `x` inside tokens, preserving token counts/positions but destroying m-specific identity.
3. `DROP_M_C3`: for C3 only, exclude every trigram containing `m`; LEX/SHAPE unchanged.

The graph is recomputed blind under each ablation with the original consensus rule (>=6/12 channels, >=2/3 families, >=3/4 transcriptions). For the target edge report support and mutual-kNN status. No threshold is tuned after outcomes.

Ablation conclusion:
- if the edge disappears under both `DROP_M_TOKEN` and `MASK_M`, classify `M_SUBTYPE_DEPENDENT`;
- if it remains a consensus candidate in both and support >=8/12 in at least one, classify `SURVIVES_M_ABLATION`;
- otherwise `PARTIALLY_M_DEPENDENT`.

Held-out rare-word sensitivity is recomputed after removing m-containing rare types; >=2 matched-null SD is required for an m-independent held-out confirmation.

## F2 — Q9–Q10–Q11 bridge/betweenness

Frozen endpoints are the blind stable nodes q09_b67_68 and q11_b71_72. Candidate middle is q10_b69_70, selected by the already-frozen discovery graph, not by this test.

Held-out validators only:
- `RARE`: mean rare-word Jaccard to the two endpoints (higher better).
- `LAYOUT`: mean line-length-distribution JS distance to the two endpoints (lower better).

Compare q10 against every other complete bifolium with the same revealed `(L,H,I)` metadata signature as q10 where possible; if this yields <8 alternatives, use all complete bifolia and report that relaxation explicitly.

For each held-out family report effect and null SD. Bridge support is `HELDOUT_BRIDGE_SUPPORT` only if both effects point in the favorable direction and at least one is >=2 null SD. Otherwise `the metric does not resolve q10 as a held-out bridge`.

The known historical quire labels 9–10–11 remain an independent **binding-stage** validation only; they cannot by themselves license production order.

## F3 — Q20 blind path held-out path test

Frozen 5-node blind stable path (reversal equivalent):
`q20_b104_115 — q20_b106_113 — q20_b107_112 — q20_b108_111 — q20_b103_116`.

The path was selected solely from LEX/C3/SHAPE; path validation uses held-out families only.

Enumerate all 5!/2 = 60 reversal-equivalent paths over these same five nodes.
- RARE score: mean adjacent rare-word Jaccard, higher better.
- LAYOUT score: mean adjacent line-length-distribution JS distance, lower better.

Report exact rank/p for each held-out family. Convert each score to favorable z relative to the 60-path exact null and compute equal-weight mean z. Component-level held-out support requires:
1. RARE and LAYOUT both favorable versus null mean;
2. at least one exact one-sided p <= .05;
3. equal-weight mean favorable z >=2.

If not, report `the held-out metrics do not resolve the blind Q20 path`.

## F4 — q20_b105_114 insertion test

Only if F3 passes. Hold the five-node path fixed up to reversal. Insert q20_b105_114 into each of the six positions. Use held-out RARE and LAYOUT only.

An insertion location is licensed only if the same position ranks first under both held-out families and its combined favorable z versus the six-position null is >=2. Otherwise insertion is unresolved.

## Reporting boundary

- A neighbourhood edge or path is not a chronological direction.
- Current folio order and quire numbers are reveal-only validators.
- No text-derived result may be called `original order` without independent physical orientation evidence.
