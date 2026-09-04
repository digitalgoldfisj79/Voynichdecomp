# e-codices authority backfill — audit

Date: 2026-09-04
Programme: Rhine–Bodensee–Upper Rhine Corridor v0.2
Status: **COMPLETE — 0 untracked registry manifest identities**

## Why this repair was required

A check prompted by the e-codices shelfmark interface showed that the ManuComp e-codices ingestion was materially lossy for provenance analysis. Before repair, 125 registry rows mapped to 102 manifest strings; 87/125 rows lacked production origin and 120/125 lacked language. Several rows also used holding institutions as apparent origins, and some stored IIIF URLs were stale aliases, case variants, or viewer-derived pseudo-manifests.

This defect could bias the neutral Bodensee/Upper-Rhine manuscript-ecology denominator downward while simultaneously creating false Swiss/St-Gall localisation through modern custody.

## Repair design

New Supabase structures:

- `public.cmp_ecodices_authority_backfill_v01` — authority metadata staged from e-codices IIIF manifests or, where a manifest endpoint failed, the e-codices scholarly object description.
- `public.cmp_ecodices_backfill_audit_v01` — field-level before/after audit.
- `public.cmp_ecodices_canonical_v01` — one canonical analytical row per current manifest identity, consolidating duplicate registry records and exposing authority origin/date/language separately from registry fallbacks.

The public registry is a UNION view, so updates were applied safely to its source tables (`public.manuscripts` and `public.cmp_acq_candidates`) rather than by adding an unsafe write-through trigger.

### Guardrails

1. Production origin was filled only where missing or demonstrably a holding/generic placeholder (`St Gallen, Stiftsbibliothek`, etc.). Good manual curation was not overwritten.
2. Language was filled only where blank.
3. Date display text was normalized only where blank or an unparsed authority string.
4. Holding location was never converted into production origin.
5. Qualified authority wording (`?`, broad regional attributions) was retained.
6. Stale manifest URLs were normalized only after object identity was verified.
7. Duplicate registry rows were not deleted; a canonical analytical view was created instead.

## Completion metrics

Registry rows: **125**.

Current unique manifest identities after URL normalization: **100**.

Authority staging records include parsed manifests plus retained alias records:

- `parsed`: 97
- `parsed_description`: 5
- `alias_resolved`: 9

There are **0 untracked current manifest identities**.

Registry missing fields after repair:

- production-origin blank rows: **20/125**, down from **87/125**;
- language blank rows: **2/125**, down from **120/125**.

Blank production origin after repair is not equivalent to unfetched metadata: in many cases e-codices itself does not provide a production place. The canonical authority view currently has 25/100 manifests without an explicit authority production-origin value and only 1/100 without a recoverable language.

Field-level audit rows: **252**:

- `place_of_origin`: 88 changes
- `language`: 118 changes
- `date_display`: 35 changes
- `iiif_manifest_url`: 11 changes

No source rows were deleted.

## Important corrections

### St. Gallen, Cod. Sang. 827

Old state included duplicate rows, one with `St Gallen, Stiftsbibliothek` in the origin field and another duplicate with null numeric dates.

Authority state:

- **production: Lake Constance region**
- **date: 1425/28**
- **languages: Latin, German**
- illustrated computistic/astronomical/cosmographical composite containing zodiac material, winds, ecliptics, planets, constellations, a bloodletting diagram, world maps and German month verses.

This is a secure primary-window Bodensee manuscript-ecology anchor. It is not scored as a Voynich resemblance.

### Basel A II 12 and A II 13

These are not Basel productions. e-codices assigns both to **Freiburg im Breisgau** and the Rüdiger Schopf multi-volume Nicolaus de Lyra Postilla project:

- A II 12: 1405–1407
- A II 13: 1413–1415

The project was sold/transferred to the Basel Charterhouse in **1430**. This is a direct documented Freiburg → Basel manuscript-transfer mechanism. The sibling volumes are dependency-linked and count as one production project, not independent votes.

### Zürich ZB Ms. C 5

Authority confirms:

- **Hagenau**
- **around 1431–1437**
- German + Alemannic
- richly illustrated Diebold Lauber workshop Historienbibel.

The pre-existing hand-curated ManuComp row was already more specific (`Hagenau, Werkstatt Diebold Laubers`; Low Alemannic/Alsatian), so it was preserved rather than overwritten.

### St. Gallen, Cod. Sang. 942

Authority gives production at the **Monastery of St Gall**, probably c.1423/1436, and explicitly describes a composite written/compiled by **several hands**. This is a direct primary-window collaborative-production ecology anchor.

## Duplicate-manifest bug caught during the repair

A first `DISTINCT ON` census accidentally omitted Cod. Sang. 827 because one of four duplicate registry rows had null numeric dates and happened to be selected before the date-window filter.

This is now prevented by `public.cmp_ecodices_canonical_v01`, which consolidates rows by manifest identity and retains aggregate numeric ranges plus a conflict flag. Two current identities have numerical date variants across duplicate rows: Cod. Sang. 827 and Cod. Sang. 931. Their authority date text remains exposed separately.

Analyses must use the canonical view or explicit authority records rather than naïve first-row deduplication.

## Repaired secure primary-window corridor core

Using only explicit authority production origin and requiring the registry numerical range to fall entirely within 1404–1438, the e-codices subset contains five secure corridor exemplars:

1. Basel UB A II 12 — Freiburg im Breisgau, 1405–1407 (Schopf project; later Basel transfer).
2. Basel UB A II 13 — Freiburg im Breisgau, 1413–1415 (same dependency project; later Basel transfer).
3. St Gall Cod. Sang. 942 — St Gall, 1423–1436, several-hand composite.
4. St Gall Cod. Sang. 827 — Lake Constance region, 1425–1428, illustrated Latin/German astro-cosmographical/computistic composite.
5. Zürich ZB Ms. C 5 — Hagenau, c.1431–1437, illustrated Lauber workshop manuscript.

A II 12 and A II 13 count as **one production-project dependency unit**. Therefore this list is five manuscripts but four independent ecology units.

Broad/uncertain overlaps (e.g. `Southwestern Germany`, `first half of 15th century`, `St Gall (?) after 1430`) remain in a separate possible-context stratum and do not enter the secure numerator.

## Graph integration

The secure repaired exemplars were added as manuscript-ecology nodes to `rhine_bodensee_v02`, together with explicit Lake Constance, St Gall, Hagenau, Freiburg, Schopf-project and Charterhouse transfer nodes/edges.

Current graph after this integration: **63 nodes / 53 edges / 8 environment bridges**.

The graph additions are production-ecology evidence only. They do not alter the frozen decision rule or count as source-family transmission unless independently justified.

## Evidential consequence

The pre-backfill neutral manuscript census was not safe for geographic inference. The repair materially increases correctly localized primary-window Bodensee/Upper-Rhine production evidence and removes several holding-location artefacts.

This does **not** establish Rhine/Bodensee provenance. It does establish that the neutral ecology test must be rerun on the repaired canonical dataset before any corridor-vs-control prevalence statement is made.
