# Paduan / Northern-Italian Technical-Notation Comparanda — First Pass

Date: 2026-08-12
Workstream: B
Status: first-stage source audit; no Voynich-adjacent literature used.

## Question

What representational technologies were available to students, physicians, mathematicians, astronomers and practitioners in northern Italy around 1350–1450 for compressing multidimensional technical information into recombinable signs, modifiers, tables, diagrams or formal notation?

This review compares **abstract mechanisms**, not glyph shapes.

## Current result

One mechanism currently merits grade **B**: the Paduan mensural-music tradition represented by Prosdocimo de Beldomandi. Four manuscript anchors merit grade **C** as direct pedagogical/curricular evidence. No grade-A precedent has yet been established.

### B — Prosdocimo / mensural notation

Prosdocimo de Beldomandi was a Paduan mathematician, astronomer and music theorist whose early-fifteenth-century musical writings include `Tractatus practice cantus mensurabilis` (Padua 1408), `Brevis summula proportionum` (1409), and `Tractatus practice de musica mensurabili ad modum Ytalicorum` (1412). Source: Treccani, `https://www.treccani.it/enciclopedia/prosdocimo-de-beldemandis_%28Dizionario-Biografico%29/`.

The relevant point is representational architecture, not resemblance. Mensural notation is explicitly context-dependent and compositional: note/sign value depends on a hierarchy of mensuration levels; dots, slashes and orientation modify a base mensuration sign; numeric proportions modify durations; ligatures combine multiple notes into a single sign and their rhythmic interpretation is contextual. Modern formal documentation of these historical mechanics is explicit in the Music Encoding Initiative mensural specification: `https://music-encoding.org/guidelines/v5/content/mensural.html`.

This is therefore a real **B-grade mechanism precedent** for:

`base sign + modifier + proportion/context + relational grouping -> interpreted value`.

It is not evidence that Voynichese is musical notation or was invented by Prosdocimo.

### C — Vat. lat. 4082

Ptolemaeus Arabus et Latinus identifies ff. 47–246 as a University of Padua unit copied by Petrus de Fita, with multiple subscriptions in 1401–1402. The codex combines optics, astronomy, astrology, natural philosophy and medicine. Particularly relevant are:

- Blasius of Parma, astronomical/natural-philosophical material;
- Nicole Oresme, `Algorismus proportionum`;
- Andalo di Negro on medical astrology;
- a 7×12 table on f. 212v mapping parts of the body to each planet's position in the twelve zodiacal signs.

Catalogue: `https://ptolemaeus.badw.de/ms/596`.

The 7×12 table is strong evidence of multidimensional technical compression in the exact Paduan place/date window, but a table is not yet an extended compositional sign system. Grade **C** pending direct folio-level notation inspection.

### C — Berlin, Staatsbibliothek, lat. fol. 246

This is the most relevant **student-notebook object class** found so far. PAL identifies it as the notebook of Ludolph Bochtorp, written principally in his own hand across Erfurt, Padua and Brunswick; he was a medicine student at Padua. It contains algorithms, astronomical notes and diagrams, astrolabe and instrument construction, solar/star/shadow tables, arithmetic, rithmimachia, astrology, uroscopy, lunar-election tables and zodiacal dignity/degree tables.

Catalogue: `https://ptolemaeus.badw.de/ms/726`.

This directly demonstrates a technically trained student carrying a personal, multi-disciplinary working volume through Padua. It remains grade **C** because the catalogue does not establish a locally devised symbol language or an extended private compositional shorthand. Direct folio inspection is a priority.

### C — Oxford, Bodleian Library, Canon. Misc. 554

PAL identifies Hand I as copied at Padua in 1435 by Candus, Prosdocimo's nephew and an arts/medicine doctor. Contents include Euclid I–VI, Prosdocimo's astronomical canons, Boethian arithmetic, constellation and planet drawings, star material and Ptolemaic material.

Catalogue: `https://ptolemaeus.badw.de/ms/558`.

This is direct evidence for the coexistence of mathematical, astronomical and diagrammatic representation in a Paduan technical codex. Grade **C** until actual notation mechanics are inspected.

### C — Venice, Museo Civico Correr, Cic. 3747 (2712)

PAL dates the manuscript between 1429 and 1460 and considers a northern-Italian, perhaps Paduan origin plausible because it contains two works by Prosdocimo. It combines astronomy, astrology, astrolabe/instrument construction and tables with Prosdocimo's `Algorismus de integris` and `Ars calculatoria`.

Catalogue: `https://ptolemaeus.badw.de/ms/433`.

This is good evidence for a Paduan technical curriculum in which procedural mathematical notation and astronomical apparatus material were copied together. Grade **C**.

### C — latitudines formarum / calculatores tradition

The broader `latitudines formarum` tradition is structurally important because qualitative states such as intensity/change are encoded geometrically rather than merely described in prose. The tradition was widespread in late-medieval science; Blasius of Parma is a central northern-Italian/Paduan receiver of calculator traditions, and Vat. lat. 4082 independently proves that Blasius and Oresmean mathematical material were being copied in a University-of-Padua technical codex in 1401–1402.

For this programme, however, the current evidence is **C**, not B: the representational mechanism itself is strong, but the first pass has not yet produced a securely c.1350–1450 Paduan student manuscript showing an extended `latitudines` graphical notation in working use.

## Manucomp check

An early exact-name/shelfmark query of `public.manuscripts` found **none of the five principal anchors above** (`Vat. lat. 4082`, `lat. fol. 246`, `Canon. Misc. 554`, `Cic. 3747`, Prosdocimo/Beldomandi). They are therefore not duplications of existing exact `manucomp` records under those names.

The broader database does already contain other Padua/northern-Italian comparanda (including herbal and scientific items), so subsequent searches should continue to query it before adding candidates.

## Important negative result

The first pass has **not** found a Paduan 1350–1450 analogue that reaches grade A: an extended, ordinary working notation in which recombinable written components systematically replace prose across many semantic dimensions.

Nor has it yet found a student notebook with a demonstrably private invented symbol vocabulary. Berlin lat. fol. 246 is the best target for that test, but the claim cannot be made from catalogue metadata alone.

## Direct-image inspection status

Direct folio inspection remains incomplete. PAL catalogue descriptions were verified, and the Vatican/Bodleian digital ecosystems were searched, but the relevant image viewers did not expose the exact target folios reliably through the current web interface. A Prosdocimo edition sample PDF was also located from the American Institute of Musicology, but screenshot retrieval failed with a cache error. No candidate has been promoted on the basis of an unseen folio.

## Workstream-A firewall

The historical B/C findings above were obtained independently of the visual discovery result. They did not determine which Voynich component was tested or selected.

If Workstream A eventually establishes a reusable invariant modifier across token families, the Prosdocimo/mensural case is already a legitimate abstract historical comparator because its interpretation explicitly depends on base signs plus modifiers/context. That comparison must remain at the level of mechanism unless a much closer Paduan technical precedent is found.
