# AMENDMENT 003 — MATCHER QA AND PRE-OUTCOME REPAIR

Date: 2026-08-08
Run: `corridor_v01_20260808_run01`
Run ID: `523d7ffe-dafd-46cb-ad5b-165f6fcea367`

## Trigger

After the first neutral census was sealed, but **before any Voynich similarity score was computed or inspected**, the preregistered control matcher was audited against its stated priority order.

The protocol states that control matching priority is:

1. same time bin;
2. overlapping broad genre/content tags;
3. same substrate if known;
4. comparable image coverage;
5. different holding institution where possible.

The implementation in `src/corridor_programme.py` instead combined Jaccard genre distance, a 0.5 substrate penalty, image-coverage penalty and holder penalty into one scalar. Consequently, a zero-content-overlap parchment manuscript could outrank a content-matched paper manuscript. This violated the preregistered ordering.

The first exact audit produced 36 corridor-control match rows for 12 corridor manuscripts, of which 27 had zero shared content tags. Six corridor manuscripts had no same-bin control sharing any tag.

## Repair

No VMS similarity had yet been seen. The repair therefore concerns design/implementation QA rather than outcome-dependent analysis.

Two actions were taken.

### 1. Semantic tag normalisation

Only already-frozen semantic equivalents were normalised. Records explicitly described as maps/charts/diagrams received the predeclared `diagram_geometry` family where missing. No manuscript was selected or removed on the basis of visual similarity.

### 2. Missing neutral controls

Two controls from the already frozen Bavaria/Swabia ecology were added because the audit demonstrated missing predeclared content families:

- Andreas Walsperger world map, Konstanz, 1448 (`Pal.lat.1362 B`) — cartographic control.
- Mendel Hausbuch, Nürnberg, primary-window stratum only (individually dated images <=1450) — practical/technical control.

Both were identified from institutional catalogues without VMS-similarity querying. Both are already present in the internal Voynich archive and are therefore replication/control material, not novel discoveries.

## Corrected matcher

Matcher v2 implements the preregistered priority lexicographically:

1. same time bin is mandatory;
2. candidates with >=1 shared content tag rank before zero-overlap candidates;
3. lower content Jaccard distance;
4. substrate match;
5. image-coverage proximity;
6. different holding institution;
7. deterministic SHA-256 tie break using seed 20260808.

The corrected v2 match table contains 36 rows (three controls per corridor manuscript). All 12/12 corridor manuscripts have at least one content-overlap control. Fourteen of 36 secondary/fallback matches have zero shared tags and are retained transparently as lower-ranked fallbacks.

## Seals

- Seal v1 MD5: `98969e6132d5b83129e394cfb26f29e3` — invalidated for primary analysis due to the pre-outcome matcher QA failure.
- Seal v2 MD5: `0b499356c5f901d9b1ac825c0657e494` — current cohort/matcher seal.
- v2 cohort size: 22 manuscripts = 12 corridor-core + 10 controls.

The v1 records are retained for auditability.

## Inference rule

Primary analysis uses matcher v2. Sensitivity analysis must additionally report:

- only control matches with >=1 shared content tag;
- exclusion of the Venetian cartographic cluster;
- exclusion of the two matcher-repair controls;
- original v1 matcher as a diagnostic only, never as the primary result.

No threshold, feature weight, VMS reference set or significance criterion was changed.
