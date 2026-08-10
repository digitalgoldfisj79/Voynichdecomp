# Q9/Q10 Scale Comparanda Protocol v0.2

Frozen 2026-08-10 after v0.1 was declared NO-GO and before any v0.2 target scores are computed.

## Why v0.2 exists
v0.1 exposed three implementation/method failures: (1) visually identical database rows carrying multiple labels were ranked as separate candidates; (2) multiple folios from one codex could monopolise the top ranks; (3) the target-specific zodiac 'calibration' was not a valid known-answer control because the Voynich zodiac pages are intentionally iconographically atypical. The blind VLM parser also failed to coerce numeric-string JSON fields. No v0.1 target rank or candidate identity is used to tune v0.2.

## Frozen question
After removing duplicate-image and codex-style effects and validating the representation on independent labelled medieval images, do any Q9/Q10 panels retrieve structurally related medieval comparanda at scale?

## Candidate universe and acquisition
Unchanged from v0.1: `public.comparanda_illuminations`, `date_start >= 1250`, `date_end <= 1500`, classes astro_diagram, sun_moon, sun, moon, star, sphere_heavens, computus_table, astrology_diagram, zodiac_wheel, zodiac_aries, zodiac_taurus, zodiac_pisces. Exact-folio IIIF resolution only; no guessed pagination. Acquisition failure is retained as missing coverage.

Rows resolving to the same image URL are collapsed into one visual entity with a union of all database labels. A manuscript key is derived from the manifest URL where available, otherwise from the normalized shelfmark/cote. Primary discovery output is manuscript-diverse: at most one image per manuscript in the displayed top list. The unrestricted image-level ranking is retained for audit.

## Representation
Model: Meta DINOv3 ViT-S/16 LVD1689M. No colour stream is used in v0.2.

Two independently ranked visual streams:
1. grayscale manuscript image;
2. edge/line-art image derived deterministically by grayscale autocontrast plus PIL FIND_EDGES, inverted and autocontrasted.

Candidate image views: full page, central 60% crop, and a 3×3 grid of overlapping 55%-width × 55%-height local windows (duplicates removed by box identity). Target views: full frozen panel plus central 60% crop. Candidate-to-target score in each stream is the maximum cosine over all candidate-view × target-view pairs. Streams are fused by reciprocal-rank fusion. This is retrieval only, not historical evidence.

## Independent known-answer calibration
Calibration does not use Voynich targets.

For each of three labelled classes—`zodiac_aries`, `zodiac_pisces`, `astro_diagram`—select eight query images deterministically by SHA-256 of the unique image URL. For each query, rank all other acquired unique images after excluding every image from the query manuscript. Collapse the ranked output to one image per manuscript. Measure same-class share in top 20 against the class prevalence in the eligible leave-manuscript-out pool.

Primary calibration statistic: median enrichment factor across the eight queries for each class. Representation passes a class if median enrichment >= 2.0 and at least 6/8 queries have enrichment > 1.0. Overall representation calibration passes only if all three classes pass. If calibration fails, target rankings remain exploratory and no machine-screen candidate can be promoted.

## Target retrieval
The sixteen frozen Q9/Q10 panel crops and frozen morphology descriptions are unchanged. Retrieval is metadata-blind. Existing named comparators do not seed or weight the rankings.

## Blind morphology adjudication
Run only if independent representation calibration passes. Compare the top three manuscript-diverse candidates per target with Qwen2.5-VL-3B using only the two images and frozen target morphology. Candidate title, date, shelfmark, class and subject are withheld.

Scores: layout 0–3; centre 0–2; partition/count 0–2; line morphology 0–2; object/figure class 0–2; text placement 0–2; fatal mismatch boolean. Numeric strings are coerced to integers before scoring. Screen pass >=8/13 with no fatal mismatch; strong screen >=10/13 with no fatal mismatch. These thresholds create a review queue only.

## Unblinding and evidence
Metadata is restored only after blind scoring. A surviving machine-screen candidate is not called a historical comparator until exact-folio repository inspection confirms the image and a feature-by-feature human/vision review survives obvious mismatches. Shared circularity or generic celestial content is insufficient.

## Compute discipline
Every GPU job has a hard timeout <=35 minutes and is non-detached. Failed/stalled stages are cancelled. A failed preregistered gate terminates downstream GPU stages rather than consuming compute.
