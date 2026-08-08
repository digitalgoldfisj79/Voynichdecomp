# PROTOCOL — Alpine–Venetian Corridor Programme v0.1

Frozen 2026-08-08 before similarity inspection.

## 1. Hypotheses

### H0
After matching for date, broad content/genre and digitisation availability, manuscripts produced in the specified Alpine–Venetian corridor show no greater VMS affinity than control manuscripts.

### H1
The corridor has higher manuscript-level cross-family VMS affinity than controls.

### H1a — antecedent
Among manuscripts whose catalogue dating permits antecedence (`date_end <= 1438`), corridor manuscripts have higher affinity than matched controls.

### H1b — ecology
Within 1390–1450, corridor manuscripts show a coherent multi-family illustration ecology closer to the VMS than controls. This does **not** imply direct copying.

### H1c — reception
1451–1475 and 1476–1500 material may reconstruct persistence/transmission but cannot support antecedence.

## 2. Geography

Core route nodes are frozen as:

1. Brixen/Bressanone
2. Bolzano/Bozen
3. Trento
4. Rovereto
5. Verona
6. Padua/Padova
7. Venice/Venezia

A candidate is `corridor_core` only when an authoritative production/origin statement places it in one of these nodes or in a locality demonstrably on the connecting route. Fuzzy regional assignments (e.g. `Tyrol?`, `Veneto?`) are `corridor_buffer`, not core. Holding location never substitutes for production location.

### Controls

Four control ecologies are frozen:

- `control_lombardy`: Milan/Pavia and surrounding north-Italian production.
- `control_tuscany`: Florence/Siena and surrounding central-Italian production.
- `control_bavaria_swabia`: Munich/Augsburg/Regensburg and surrounding production.
- `control_east_alpine`: Salzburg/Vienna/Graz and surrounding production.

Controls are selected without using VMS similarity, then matched to corridor manuscripts by chronology bin, broad genre/content tags, substrate where known, and image availability.

## 3. Time bins

- `A_primary`: 1390–1450
- `B_antecedent`: 1350–1389
- `C_reception`: 1451–1475
- `D_late_reception`: 1476–1500

If a catalogue range crosses bins, preserve the verbatim date and both integers; do not force a precise year. For causal antecedent analyses, require `date_end <= 1438`.

## 4. Candidate inclusion

Required:

1. institutional or scholarly catalogue identity/shelfmark;
2. date range intersecting 1350–1500;
3. production/origin evidence sufficient for one frozen geographic class, or explicit `unresolved`;
4. evidence of relevant illustration OR a digitised facsimile that can be inspected for incidental drawings.

Relevant content is defined before inspection as: botanical/herbal, medical/pharmaceutical, astronomical/astrological, calendrical, cosmological, balneological, alchemical, natural-philosophical, cartographic/geographical, architectural/technical, practical miscellany, or incidental drawings in a working manuscript.

### Exclusion

- Selection solely because a source calls an image "Voynich-like".
- Holding-place-only geographic assignment.
- Undated records with no defensible 1350–1500 range.
- Modern facsimiles/reconstructions.
- Printed books for the primary manuscript test.

## 5. Discovery rule

Discovery and ranking are separated.

The discovery stage may query catalogue metadata for date/place/content/illustration terms, but may not query for `Voynich`, `VMS`, known Voynich folio labels, or visual similarity. Existing Manucomp records originally discovered through Voynich research remain usable only if their inclusion can be reproduced from the frozen neutral criteria; provenance of discovery is retained.

Every candidate receives:

- stable candidate key;
- shelfmark/title;
- holding institution;
- catalogue URL;
- production/origin statement verbatim;
- place authority URL where available;
- verbatim dating;
- parsed date bounds;
- source of illustration evidence;
- facsimile/IIIF URL;
- discovery source and timestamp;
- geography class and confidence.

## 6. Image triage

Triage is blind to geographic class and manuscript identity where technically possible. Page images are assigned only opaque IDs.

Frozen classes:

- `plant`
- `root`
- `flower`
- `zodiac`
- `star_astronomy`
- `bath_human`
- `architecture_cartography`
- `diagram_geometry`
- `other_relevant`
- `none`

The triage model/coder may localise candidate regions and describe visible morphology, but may not identify the manuscript, place, artist, school, or speculate about relation to the VMS.

## 7. Image normalisation and the known confound

Existing Manucomp measurements show raw image embeddings can classify manuscript/institution/scan pipeline essentially perfectly. Therefore:

- raw full-page RGB embeddings are prohibited as inferential evidence;
- use object crops or tightly bounded relevant regions;
- background divide/flatten;
- greyscale structural arm;
- optional ink-mask/skeleton arm;
- fixed resize/padding policy;
- retain a text-description arm independent of pixel colour.

### Confound gate

Before the image arm can support H1, train a manuscript/institution classifier on the final normalised representations using grouped cross-validation.

- AUC `<= 0.65`: pass.
- `0.65 < AUC <= 0.70`: caution; sensitivity analyses mandatory.
- AUC `> 0.70`: image-embedding arm fails and is excluded from inferential composite. Text/geometry/codicology arms remain available.

No threshold may be relaxed after seeing corridor effects.

## 8. Feature families

Within each visual class, comparisons use at least two representation families where feasible:

1. structure-oriented image embedding (DINOv3 or frozen successor, normalised crop only);
2. blind structured visual description -> text embedding;
3. explicit geometry/morphology features appropriate to class.

Examples:

- plants: branching topology, leaf attachment, flower/root arrangement;
- roots: count, topology, tuber/strand geometry, symmetry;
- zodiac: sign pose, figure arrangement, band/ring geometry, emblem structure;
- astronomy: star glyph geometry, radial structure, labels/figure topology;
- bath/human: vessel/tub geometry, figure count, posture/topology;
- architecture/cartography: tower/roof/crenellation/flag geometry, enclosure topology;
- diagrams: circles, spokes, nested rings, connectors, compartment topology.

Colour/pigment is a descriptive secondary arm unless acquisition is demonstrably calibrated.

## 9. VMS reference set

The VMS reference corpus is frozen from existing reviewed/manually corrected Manucomp objects where available. Automatic/unreviewed detections may be used for discovery but not the primary endpoint unless their error rate is quantified before analysis.

Class matching is strict: cross-class nearest neighbours are invalid.

## 10. Scoring

Unit of inference is the **manuscript**.

For each manuscript `m` and feature family `f`:

1. compute class-matched VMS-to-comparandum object similarities;
2. convert object similarities to null-calibrated z/rank scores using non-VMS manuscript pairs and/or permutation nulls;
3. aggregate within class robustly (median of top-k with k frozen by available object count; never all crop-pairs as independent observations);
4. aggregate classes to a manuscript-family score;
5. create the manuscript composite from available independent families with equal family weight.

Missing families remain missing; absence is not scored as dissimilarity unless the facsimile is complete and the class is verified absent.

## 11. Primary statistical test

A stratified permutation test compares manuscript composite scores for `corridor_core` versus matched controls.

Strata:

- time bin;
- broad content/genre;
- substrate when known;
- usable image coverage band.

Permutation unit: manuscript.

Primary Monte Carlo permutations: 100,000 with fixed seed `20260808`.

Primary significance rule: two-sided `p < 0.01` **and** positive effect direction.

Convergence requirement: at least three independent feature families must show corridor-positive effects, with at least two surviving Benjamini–Hochberg FDR `q < 0.05`. A single dominant family cannot establish H1.

Effect sizes and confidence intervals are reported regardless of p-value.

## 12. Negative controls / falsifiers

H1 is materially weakened by any of:

1. corridor composite not above matched controls;
2. signal disappears when holdings from the same digitising institution are excluded;
3. signal exists only in colour/raw-pixel representations;
4. signal is driven by one manuscript or one visual class;
5. Bavaria/Swabia or Lombardy performs equally or better under the frozen test;
6. production-place uncertainty explains the result;
7. leave-one-manuscript-out changes the sign of the corridor effect;
8. model descriptions leak place/manuscript identity;
9. primary result fails after restricting to institutional dating/place authorities.

## 13. Coverage gates

No strong geographic inference if:

- fewer than 12 verified illustrated `corridor_core` manuscripts in A+B combined; or
- fewer than 8 usable matched controls in at least two control ecologies; or
- >30% of otherwise eligible corridor candidates lack inspectable images and missingness is demonstrably geographically non-random.

If underpowered, report `UNDERPOWERED`; do not expand the primary time window post hoc. D can be reported only as supplementary reception evidence.

## 14. Sensitivity analyses

Predeclared:

- core only vs core+buffer;
- `date_end <= 1438` antecedent restriction;
- IIIF-only / complete-facsimile-only;
- exclude each holding institution in turn;
- exclude each corridor node in turn;
- leave-one-manuscript-out;
- image arm removed;
- text arm removed;
- geometry arm only;
- Italian controls only;
- Germanic controls only.

## 15. Interpretation ladder

- **Tier 0:** no enrichment.
- **Tier 1:** one-family resemblance only; descriptive.
- **Tier 2:** multi-family corridor enrichment, robust to controls; supports regional ecology.
- **Tier 3:** Tier 2 plus antecedent-only enrichment and independent palaeographic/codicological corroboration; meaningful provenance evidence.
- **Tier 4:** requires direct documentary/codicological linkage; this programme alone cannot establish it.

The programme is designed to distinguish Tier 0–3. It cannot prove a specific workshop, author, or source manuscript.
