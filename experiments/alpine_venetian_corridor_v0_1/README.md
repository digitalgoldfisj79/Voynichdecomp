# Alpine–Venetian Corridor Programme v0.1

Status: **BUILT / NOT YET RUN**  
Date frozen: 2026-08-08  
Supabase project: `ymaqlcfjmdwncdbjprmw`  
Primary registry: `public.manuscripts`  
Existing image/object spine: `cat_herbal_folios`, `herbal_objects`, `comparanda_illuminations`, `zodiac_sign_ratings`, `cmp_*` tables.

## Question

Does the manuscript-and-illustration ecology along the Alpine–Venetian route

**Brixen/Bressanone → Bolzano/Bozen → Trento → Rovereto → Verona → Padua/Padova → Venice/Venezia**

show greater multi-feature affinity to the Voynich Manuscript than date-, genre-, and digitisation-matched control regions?

This is a provenance/ecology test, **not** a search for a single "source manuscript".

## Core design

1. Build a bounded census using geography/date/illustration status only.
2. Verify production place and dating from institutional catalogue authorities.
3. Acquire available page images, preferably IIIF.
4. Blind-triage illustrations into predeclared classes without access to manuscript identity or region.
5. Extract and normalise crops; never compare raw full-page colour embeddings across institutions.
6. Score independent visual/structural feature families against VMS objects.
7. Compare corridor manuscripts against matched controls at the **manuscript level**, not the crop level.
8. Run the scan/institution-confound gate before accepting any image-arm inference.
9. Report nulls, missingness and underpowered strata explicitly.

## Chronology

- **A — primary ecology window:** 1390–1450.
- **B — antecedent window:** 1350–1389, with a causal antecedent subtest requiring `date_end <= 1438`.
- **C — reception window:** 1451–1475, descriptive/supporting only.
- **D — late reception supplement:** 1476–1500, never used as evidence for VMS production date or antecedence.

The Trento late-15th/early-16th-century material therefore belongs to reception/network reconstruction, not the primary causal test.

## Predeclared visual families

`plant`, `root`, `flower`, `zodiac`, `star_astronomy`, `bath_human`, `architecture_cartography`, `diagram_geometry`.

Script/palaeography and codicology are independent corroboration arms and must not be used to select the image cohort.

## Primary endpoint

A stratified manuscript-level permutation test comparing the corridor and matched controls on a cross-family composite similarity score. The programme additionally requires convergent evidence across independent families; a large effect in one family alone is insufficient.

See `PROTOCOL.md` and `PREREGISTRATION.md` for frozen rules.

## Existing database baseline

A pre-build query found a small seed set already registered in `public.manuscripts`, including Egerton MS 2020 (Padua), the Roccabonella Herbal (Padua), the De Virga map (Venice), Tyrolean manuscripts, and later Padua comparanda. None currently has downstream `cat_herbal_folios`/`herbal_objects` coverage under those registry IDs. This programme therefore begins with **coverage and binding**, not similarity ranking.

## Run shape

```text
stage 0  audit existing registry and freeze snapshot
stage 1  discover + verify corridor/control candidates
stage 2  resolve IIIF/facsimiles and page inventory
stage 3  blind illustration triage
stage 4  crop + normalise + describe + embed
stage 5  feature-family scoring and null calibration
stage 6  confound gates + manuscript-level statistics
stage 7  sensitivity analyses and falsification report
```

The executable entry point is `src/corridor_programme.py`.
