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

## Mandatory archive-first novelty gate

Before external catalogue work is interpreted, the programme searches Ed's internal Voynich archive (`public.sources` + `public.source_passages`) for prior discussion of the hypothesis, each candidate manuscript, people/workshops, route links and claimed visual/documentary features.

The build-time archive scan already establishes that the broad German/Alpine + North-Italian/Tyrol/Trento/Padua/Venice proposition is **prior art**. The programme therefore prioritises genuinely new manuscripts, previously unseen images, documentary/transmission links, new negative evidence and controlled measurements. It never reports the corridor idea itself as a new discovery.

See `AMENDMENT_001_ARCHIVE_NOVELTY_GATE.md`. The executable archive scanner is `src/archive_prior_art.py`.

## Core design

1. Search the internal Voynich archive and classify prior art before external interpretation.
2. Build a bounded census using geography/date/illustration status only.
3. Verify production place and dating from institutional catalogue authorities.
4. Acquire available page images, preferably IIIF.
5. Blind-triage illustrations into predeclared classes without access to manuscript identity or region.
6. Extract and normalise crops; never compare raw full-page colour embeddings across institutions.
7. Score independent visual/structural feature families against VMS objects.
8. Compare corridor manuscripts against matched controls at the **manuscript level**, not the crop level.
9. Run the scan/institution-confound gate before accepting any image-arm inference.
10. Report nulls, missingness, prior-art status and underpowered strata explicitly.

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

See `PROTOCOL.md` and `PREREGISTRATION.md` for frozen inferential rules. Amendment 001 adds the archive novelty gate without changing the endpoint or thresholds.

## Existing database baseline

A pre-build query found a small seed set already registered in `public.manuscripts`, including Egerton MS 2020 (Padua), the Roccabonella Herbal (Padua), the De Virga map (Venice), Tyrolean manuscripts, and later Padua comparanda. None currently has downstream `cat_herbal_folios`/`herbal_objects` coverage under those registry IDs. This programme therefore begins with **prior-art classification, coverage and binding**, not similarity ranking.

## Run shape

```text
stage -1 archive prior-art / novelty scan (mandatory)
stage 0  audit existing registry and freeze snapshot
stage 1  discover + verify corridor/control candidates; archive-scan each candidate
stage 2  resolve IIIF/facsimiles and page inventory
stage 3  blind illustration triage
stage 4  crop + normalise + describe + embed
stage 5  feature-family scoring and null calibration
stage 6  confound gates + manuscript-level statistics
stage 7  sensitivity analyses, novelty classification and falsification report
```

Executable entry points:

- `src/archive_prior_art.py` — internal archive prior-art and novelty gate.
- `src/corridor_programme.py` — census, facsimile, matching and analysis pipeline.
