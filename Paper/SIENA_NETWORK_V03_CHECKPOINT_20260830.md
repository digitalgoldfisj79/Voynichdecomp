# Siena network checkpoint — v0.3 / v0.3.1

Date: 2026-08-30

This checkpoint records the append-only Siena extension built after `SIENA-NETWORK-0.2-20260830`. The authoritative structured graph remains in Supabase project `ymaqlcfjmdwncdbjprmw`; this file is a reproducibility/audit summary and does not replace the database.

## Frozen runs

### SIENA-NETWORK-0.3-20260830

Run ID: `6fd4a4d9-dd60-4825-adf1-771b5c197c96`

Status: `v03_frozen`

Verified people/institution core:

- nodes: 47
- edges: 73
- components: 1
- all 47 nodes reachable from `studium_senese`

Hashes:

- nodes MD5: `3f3ca2caf97e9881bdbed3d7db7a1644`
- edges MD5: `ba7b1ce363e32dd9cf819091dd4c3317`
- graph MD5: `0e6d9f02c52e979f38da5e82815a571e`

### SIENA-NETWORK-0.3.1-20260830

Correction-only run. No topology change.

Purpose: correct Paper 3 D1 to the same regional-access standard used for paired control environments. Siena is therefore descriptively 6/6 under the frozen Paper 3 domain definitions, but this is **not a localisation score** because the known-answer calibration failed.

## Main hard result: a person-level Siena → Pavia → Alfonso chain

Antonio Beccadelli (Panormita) is now a major production/transmission bridge:

`Studium Senese / Siena humanist circle → Antonio Beccadelli → Pavia / Visconti court-university → Alfonso V court`

Documentary components:

- Beccadelli studied law in Siena under Niccolò de Tudeschi in the early 1420s.
- He belongs to the Sienese humanist network linked to Mariano Sozzini.
- Gasparino Barzizza was an important teacher/intellectual father figure.
- Beccadelli was in Pavia in the 1429–1434 period, combining court, university and manuscript activity.
- He entered Alfonso V's service in 1434 and became a central royal humanist/councillor.
- Lorenzo Valla moved to Pavia in 1431 encouraged by Beccadelli and later also entered Alfonso's service.

This is a genuine named human carrier joining Siena to two previously important Paper 3 environments and to the court context of Yates Thompson MS 36.

## Yates Thompson MS 36: collaborative manuscript-production analogue

Hard evidence retained in the graph:

- BL Yates Thompson MS 36 is a mid-1440s illuminated Dante manuscript.
- One principal scribe copied the text; rubrication is probably by the same hand.
- The page design was planned to receive extensive illumination.
- At least two distinct illumination hands participated.
- Giovanni di Paolo is securely responsible for the Paradiso campaign.
- The first two cantiche are represented conservatively as the anonymous `Maestro della Commedia Yates Thompson` rather than identified with Priamo della Quercia or another named painter.
- Alfonso V is the early royal owner/destination.

Specialists independently infer a highly learned iconographic `concepteur`/`designator` because the images depend on commentary and learned/classical interpretation rather than simple literal illustration.

Candidates proposed in scholarship include:

- Guiniforte Barzizza
- Antonio Beccadelli (Panormita)
- Lorenzo Valla

**Guardrail:** none has a surviving commission/payment/document proving authorship of the iconographic programme. Candidate links remain hypothesis-level.

The relevant analogue is therefore not “Priamo made Voynich”; it is the demonstrable production structure:

`learned programme → principal scribe + multiple artists → complex illustrated manuscript`

## Yates Thompson Master → Padua/Pavia

Turin BNU I.I.13, Francesco Zabarella, *Lectura super Clementinis*:

- copied by Johannes de Polonia
- colophon dated Padua, 27 March 1417
- several specialists have attributed or closely associated its illumination with the later anonymous Yates Thompson first master

Pavia BUP Aldini 343, Bartolo da Sassoferrato, *Lectura Digesti Veteris*:

- Bollati identifies stylistic affinities with the Yates Thompson first master

**Guardrails:**

- the Padua copying colophon does not prove that the illumination campaign was executed in Padua in 1417
- current Pavia holding does not prove Pavia production
- same-hand identity remains stylistic attribution, not a documentary itinerary

This route is therefore classified as `stylistically_supported_not_documentary`.

## Mariano Sozzini → De sortibus → Bessarion

This is a new hard primary-window result.

Maura Mordini (2024), *Scripsit de sortibus: Mariano Sozzini il Vecchio e la magia, tra teologia e diritto nell’epoca del cambiamento (sec. XV)*, DOI `10.32064/26.2024.18`, establishes that:

- Sozzini composed the *Tractatus de sortibus* in the late 1430s / early 1440s.
- By 19 September 1443 it was finished and had already been sent to Cardinal Bessarion.
- The dedication manuscript is BAV Reg. lat. 1272.
- Fabio Troncarelli has regarded the dedication copy as substantially autograph.
- Sozzini probably knew Bessarion from the Ferrara-Florence council environment.

This supplies a person-level Siena-core bridge to a major Greek/Byzantine scholar and manuscript collector.

**Guardrail:** the work is a learned canon-law/theological treatment of sortes, divination and magic. It is not evidence that Sozzini personally practised geomancy or magic.

## Jacopo della Quercia → Priamo succession

Hard evidence:

- Jacopo died in October 1438.
- Priamo was his brother and executor.
- Priamo petitioned Siena on 5 April 1440 concerning the estate.
- a 1440 cathedral review of the estate survives.
- Priamo is securely working as a painter at Santa Maria della Scala in 1442.
- Jacopo's own use of detailed design drawings and model/design practice is independently documentary, including the Fonte Gaia parchment drawings.

**Unresolved:** no item-level evidence recovered in this pass proves that Jacopo's drawings, books, pattern sheets or model material passed specifically to Priamo. Estate/personnel continuity is verified; model-sheet continuity is not.

## Paper 3 D1 symmetry correction

The earlier Siena database entry had retained a stricter city-wall rule for illustrated-herbal access than was used for paired Paper 3 environments. That was corrected in `SIENA-NETWORK-0.3.1-20260830`.

Hard mobility:

`Giovanni Sermoneta → Florence medical chair 1432–1437 → Siena medical teaching 1437–1438`

Contemporary regional illustrated-herbal context:

- Laurenziana Redi 165 / the Ghino herbal is dated 1430–1449, generally Tuscan / presumably Florentine.
- It contains roughly ninety illustrated medicinal plants, including anthropomorphic/zoomorphic root imagery and harvesting/astral information.

Under the same regional-access ecology standard used for Padua/Venice and Pavia/Milan, Siena D1 is covered.

**Guardrail:** this does not prove Sermoneta personally saw Redi 165, nor that Redi 165 was physically present in Siena.

## Paper 3 descriptive coverage

Under the existing frozen domain definitions Siena is now descriptively 6/6:

- D1 illustrated medicinal-plant access — covered under symmetric regional-access rule
- D2 bathing/water-system/process imagery — covered directly through Taccola
- D3 medical-pharmaceutical institutional culture — covered through Studium, Giovanni Sermoneta, Ugo Benzi and Santa Maria della Scala
- D4 astronomy/cosmology — covered through primary-window astronomy/astrology teaching at the Studium
- D5 cross-regional scholar integration — covered, now reinforced by Panormita/Pavia/Alfonso and Sozzini/Bessarion
- D6 manuscript-copying infrastructure — covered through Domus/Studium, private copying and manuscript-procurement networks

**Critical calibration guardrail:** `paper3_known_answer_v01` is not a validated provenance classifier. It weakly recovered Canon Misc. 554 and failed the Clm 671 known-answer test. Therefore `6/6` is evidence of source-environment breadth, not evidence that Siena is the Voynich production location.

## Closed negative visual result remains closed

Nothing in the Siena historical graph changes the earlier Taccola/Q13 visual result:

- frozen v0.1 visual calibration failed
- v0.2 motif development failed transfer
- Q13 remained sealed during calibration
- the visual programme was closed

Historical convergence evidence must not be used to resurrect failed visual significance.

## Current claim limit

The strongest defensible statement after v0.3 is:

> Siena c.1420–1450 is a highly connected, historically documented convergence and manuscript-production ecology joining all six pre-specified Paper 3 knowledge/production domains, with named people linking Siena to Pavia/Visconti, Alfonso's court, German/imperial networks and Bessarion's Greek/curial world. This does not identify the Voynich author or illustrator and does not establish that the Voynich manuscript was produced in Siena.

## Remaining discriminating questions

1. Can the Yates Thompson learned programme author be identified documentary rather than stylistically?
2. Can the Yates Thompson first-master itinerary be established beyond stylistic attribution to Turin I.I.13 / Pavia Aldini 343?
3. Can Jacopo's estate be shown item-by-item to have transferred drawings/model sheets/books to Priamo or another named workshop successor?
4. Can the Yates Thompson scribe be localized more securely than the current southern-Italian linguistic signal?
5. Can practical geomancy, as distinct from Sozzini's learned *De sortibus*, be securely placed in Siena before 1450?

These are now evidence-upgrade questions, not missing-domain searches.
