# Amendment 003 — Blind VMS reference review gate

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Reason

The sealed corridor/control census reached the numerical coverage threshold before any Voynich similarity was computed, but the existing VMS reference catalogue was inadequately reviewed outside the plant/root arms. The primary convergence rule requires at least three independent visual families. This amendment defines a pre-outcome reference-set QA step; it does not alter the corridor or control cohort and is completed without exposing the reviewer to corridor/control images or scores.

## Frozen review set

The twelve rows already present in `public.cat_zodiac_worklist`, in their existing sequence and with their existing Yale IIIF identifiers, panel IDs, expected sign/variant, and iconographic descriptions. No new zodiac panel may be added because it looks useful.

## Blind review procedure

1. Retrieve the eight Yale source images referenced by the twelve pre-existing worklist rows.
2. Review the complete Yale source image rather than blindly trusting stored `cx/cy/radius`, because the worklist contains mixed display-scale coordinates.
3. Reviewer sees only the Yale source image and the pre-existing expected panel identity. Reviewer receives no corridor/control image, manuscript identity, region, similarity score, or retrieval result.
4. For each worklist row record three booleans: expected panel present; iconographic identity consistent; spatial assignment unambiguous.
5. Promotion rule: `verified` only if all three are true. Otherwise `ambiguous` or `rejected`; no manual rescue after seeing corridor results.
6. Raw reviewer output is preserved in `public.corridor_reference_reviews` before updating `cat_zodiac_worklist.reviewed`.

## First reviewer

`Qwen/Qwen2.5-VL-7B-Instruct`, deterministic generation (`do_sample=false`), run as bounded HF job `6a773d383e1f34a7e32bedb2` on one L4, hard timeout 30 minutes.

## Outcome firewall

At amendment time:

- VMS similarity computed: **false**.
- Corridor similarity scores seen: **none**.
- Corridor/control cohort seal v2 remains `0b499356c5f901d9b1ac825c0657e494`.
- This review may promote or reject only the pre-existing VMS zodiac references. It cannot change corridor/control membership, matching, dates, geography, or content tags.

## Gate

The DINOv3 similarity stage remains blocked unless the reference set supplies at least three independently usable visual families under the programme's frozen convergence rule. Failure of the zodiac review leaves the programme non-resolving rather than lowering the rule.
