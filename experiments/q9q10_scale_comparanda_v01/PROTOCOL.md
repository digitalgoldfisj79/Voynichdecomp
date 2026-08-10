# Q9/Q10 Scale Comparanda Protocol v0.1

Frozen 2026-08-10 before target-to-candidate scores were computed.

## Question
Can a broad, date-bounded medieval image corpus retrieve source-secure comparanda for the sixteen Q9/Q10 panels on more than generic circularity, and which retrieved candidates survive a blind feature-by-feature visual screen?

## Targets
The sixteen Q9/Q10 Yale-derived panel crops. Their visible morphology is inherited unchanged from the frozen Q9/Q10 inventories and `FROZEN_PANEL_FEATURES.csv`. Existing named external comparators are not used to seed retrieval.

## Candidate universe
`public.comparanda_illuminations`, date window 1250–1500, classes: astronomical diagram, Sun/Moon/Sun/Moon/star, celestial sphere/computus/astrology, zodiac wheel and Aries/Taurus/Pisces. The universe is frozen by the SQL predicate, not by similarity. Records lacking a resolvable image are reported as acquisition failures rather than replaced manually.

## Stage A: image acquisition
Use an explicitly usable stored thumbnail where available. Otherwise resolve the row's IIIF manifest and match the declared folio against canvas labels. No guessed page-number offsets. Multi-folio/range rows may contribute only explicitly matched endpoints/tokens; unresolved records remain unresolved.

## Stage B: blind visual retrieval
Model: Meta DINOv3 ViT-S/16 LVD1689M. Candidate page representation: full image plus four overlapping quadrant crops. Target: frozen panel crop. Two image views are ranked independently: RGB and grayscale. Candidate score is maximum target cosine across its five local views. RGB and grayscale ranks are fused by reciprocal-rank fusion. Metadata fields such as title, subject, shelfmark and class are not inputs to the visual representation.

DINO similarity is candidate generation only. It is not iconographic evidence.

## Calibration and failure checks
Before interpretive use, report class enrichment for the explicit zodiac targets f70v1 (Aries) and f70v2 (Pisces), plus astronomical-diagram enrichment for f68v3. A failure to retrieve conventional zodiac peers would invalidate broad use of the ranking.

## Stage C: blind VLM morphology screen
Top four fused candidates per target are compared pairwise to the target by Qwen2.5-VL-3B. The model is shown only the two images and the frozen target morphology, not candidate metadata. It assigns bounded scores to layout, centre, partition/count, line morphology, object/figure class and text placement, plus a fatal-mismatch flag.

Screen pass: >=8/13 and no fatal mismatch. Strong machine-screen hit: >=10/13 and no fatal mismatch. These are triage thresholds, not historical identifications.

## Unblinding and evidential rule
Candidate title/date/shelfmark/subject metadata are reattached only after blind visual scoring. A candidate can become a serious comparator only after repository-level source verification and human/vision inspection of the exact folio. Shared circularity, shared broad subject class, or a high embedding score alone is insufficient.

## Hard compute discipline
One bounded GPU job; hard timeout <=35 minutes. DINO is released before loading Qwen. No detached job is permitted to remain running after the turn; stalled/error jobs are cancelled.
