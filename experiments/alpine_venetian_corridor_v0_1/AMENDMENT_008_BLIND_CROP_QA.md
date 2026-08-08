# Amendment 008 — Second-pass blind crop QA

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Reason

Stage 4 blind page triage is intentionally recall-oriented. Inspection of its output schema, before any VMS similarity is computed, confirms that it can propose oversized regions, generic initials, text blocks, or a wrong member of the frozen class vocabulary. The protocol already requires crop QA before scoring. This amendment freezes the acceptance rule.

## Reviewer input firewall

The reviewer receives only:

- the proposed crop pixels;
- its opaque object ID;
- the proposed frozen object class;
- the neutral morphology description produced during page triage.

It receives **no** manuscript title/shelfmark, date, production place, corridor/control label, archive status, VMS image, VMS reference description, or similarity score.

## Acceptance criteria

A proposed crop is `usable` only if all are true:

1. it contains a substantive non-text visual object relevant to the proposed frozen class;
2. the proposed class is materially consistent with the visible crop;
3. the crop is sufficiently tight that the target object, not surrounding page/text, dominates the visual signal;
4. the crop is not merely an initial, isolated letter, text block, border, generic ornament, scanning artefact, colour target, binding, or blank region;
5. for `other_relevant`, the crop must still contain a substantive technical/scientific/practical visual structure that does not fit another frozen class; generic decoration/text never qualifies.

Otherwise classify as `spurious` or `bad_crop`. No failed crop may be manually rescued after similarity inspection.

## Model

Second-pass reviewer: `Qwen/Qwen2.5-VL-7B-Instruct`, deterministic (`do_sample=false`). It is larger than the 3B page triager and receives only the isolated crop.

## Persistence

All Stage 4 v2 proposals are persisted first with `crop_qa='unreviewed'`. The second-pass decision then updates `crop_qa`; only `usable` objects can enter image/text/geometry scoring.

## Outcome firewall

At amendment time `vms_similarity_computed=false`. This QA can remove bad model inputs only; it cannot change cohort membership, matching, metadata, dates, geography, thresholds, VMS reference set, or positive-result criteria.
