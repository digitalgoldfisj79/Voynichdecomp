# Source acquisition status for recoverability recalibration v0.6

**Date:** 2026-07-25  
**Branch:** `experiment/voynich-recoverability-open-set-v0.5-20260725`

## Immediately usable public corpus

### CoReMA — Cooking Recipes of the Middle Ages

Status: **machine-readable source confirmed**.

The University of Graz GAMS collection exposes manuscript-level and recipe-level TEI/XML, RDF/XML and plaintext. Relevant period and region examples include:

- Vienna, ÖNB Cod. 2897 (`o:corema.w1`), dated 1350–1450, Bavarian and Medieval Latin, CC BY 4.0;
- Graz UB Ms. 1609 (`o:corema.gr1`), dated 1450–1500, Early New High German and Medieval Latin, CC BY 4.0;
- recipe-level objects dated 1400–1500 with semantic annotations.

This is the first appropriate real medieval procedural corpus for v0.6. Its TEI contains diplomatic surface transcription and recipe structure suitable for deriving action, ingredient, quantity, preparation and sequence fields without target tuning.

## Relevant but not yet released as an open bulk corpus

### E-LAUTE

Status: **period-correct project identified; bulk machine-readable data not publicly released yet**.

E-LAUTE covers German-speaking lute tablature, 1450–1550, approximately 1,700–2,000 pages. The project states that it is converting Fronimo and MuseScore data to MEI 5.1 and that files will be made available. The current public site documents the workflow and edition platform but does not expose a downloadable bulk MEI release.

Action: prepare a direct data-access request to the project team for a research snapshot containing surface tablature and aligned MEI/event representations.

## Small usable proof-of-concept, not a corpus

### TScore German lute tablature examples

Status: **semantically aligned examples available, insufficient scale**.

The 2024 TScore paper includes explicit German lute tablature source examples and an XML intermediate model containing duration, string, fret, graphical position, beam and prolongation fields. These examples can validate the ingestion and recoverability code but cannot support manuscript-grouped calibration.

## Facsimile available; aligned transcription not located

### Buxheim Organ Book

Status: **period-correct facsimile/source identified; no open machine-readable surface/event corpus located**.

The manuscript is dated approximately 1450–1470 and contains more than 250 pieces. Public descriptions and facsimile access exist, but no complete open MEI or aligned diplomatic transcription was located in the initial search.

Action: either obtain an existing scholarly transcription under licence or curate a preregistered representative sample from the facsimile with independent double entry.

## Existing calibration sources already acquired

- AmmerbachReal: 2,400 paired duration/special and pitch/rest annotations, 1575/1583; pipeline calibration only.
- ECHOES GABCtoMEI: Aquitanian and square-neume surface encodings.

## Acquisition order

1. Ingest CoReMA TEI/XML and freeze its operational-field extraction.
2. Contact E-LAUTE for a machine-readable research snapshot.
3. Use TScore examples as an ingestion/unit-test set.
4. Decide whether Buxheim warrants manual double transcription after the CoReMA recalibration result.

The v0.5 `CALIBRATION_FAILURE` result is not revised by this source audit. v0.6 begins only after the CoReMA parser, field ontology and manuscript-grouped holdout rules are frozen.
