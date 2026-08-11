# Q9/Q10 Structural Comparanda v0.3 — preregistration

Date: 2026-08-11
Branch: `experiment/q9q10-structural-comparanda-v0.3-20260811`

## Scientific aim

Replace generic image-nearest-neighbour matching with explicit, inspectable diagram primitives. The target is the astronomical/cosmological material in Q9/Q10. Zodiac-specific calibration is **not** a gate for this experiment.

The frozen Q9/Q10 descriptions in `Q9_Q10_DATA.xlsx` and the Q9/Q10 panel inventories are the target ground truth. No target feature is altered after candidate rankings are observed.

## Target set

Primary targets: f67r1, f67r2, f67v1, f67v2, f68r1, f68r2, f68r3, f68v1, f68v2, f68v3, f69r, f69v, f70r1, f70r2.

f70v1/f70v2 are retained only as non-gating contextual controls and are not used to decide whether the astronomical search is valid.

## Candidate universe

`comparanda_illuminations`, date interval fully inside 1250–1500, restricted to astronomical/cosmological classes:

- astro_diagram
- astrology_diagram
- sphere_heavens
- computus_table
- sun_moon
- sun
- moon
- star
- zodiac_wheel

One visual entity per resolved image URL. Candidate retrieval is manuscript-diverse: no target ranking may contain more than one image from the same manuscript before every other manuscript has had an opportunity to appear.

## Frozen structural primitive schema

Each candidate image is converted to this explicit vector by one deterministic VLM pass (`do_sample=false`). Unknown/ambiguous values are recorded as null/unknown rather than guessed.

### Counts

- radial_line_count
- sector_count
- compartment_count
- concentric_ring_count
- outer_repeated_unit_count
- axial_line_count
- diagonal_line_count
- corner_roundel_count
- luminary_count
- face_count
- human_figure_count
- pipe_or_tube_count
- spiral_band_count

Counts above 40 may be clipped to 40 for distance scaling but the raw count is retained.

### Categorical primitives

- center_type: none | sun | moon | star | earth_partition | geometric | animal | human | mixed | other | unknown
- layout_type: radial_rota | concentric_rota | open_field | annular | corner_network | grid_table | mixed | other | unknown
- text_layout: none | radial_inward | radial_outward | radial_mixed | circular | scattered_labels | mixed | unknown
- star_distribution: none | central | scattered | annular | outer_ring | alternating_sectors | clustered | mixed | unknown
- boundary_type: open | plain_circle | compartmented | lobed | wavy_nebuly | other | unknown

### Boolean primitives

- central_face
- central_luminary
- concentric_structure
- starry_annulus
- corner_network
- repeated_faces
- alternating_star_text
- connected_star_cluster
- pipe_ring
- t_o_partition
- colour_coded_axes

## Target encoding

Voynich target vectors are manually encoded from the frozen observations before retrieval. Counts that the inventory explicitly distinguishes (e.g. 16 drawn sectors but 8 text-bearing sectors on f68v1) are preserved in separate relevant fields rather than collapsed.

## Candidate extraction

Model: Qwen2.5-VL-7B-Instruct (or exact compatible Qwen VL checkpoint if the named checkpoint is unavailable). Output must be one JSON object matching the schema. Parsing is strict; invalid JSON gets one deterministic repair prompt and then is marked extraction_failed. No manual correction of candidate vectors is allowed before ranking.

## Distance

Missing-aware weighted Gower-style distance.

Numeric count distance: `min(|a-b| / scale, 1)`, scales fixed before run:
- radial/sector/compartment/outer unit: 32
- rings: 8
- axial/diagonal/corner: 8
- luminary: 2
- face/human: 30
- pipe/tube: 30
- spiral: 12

Categorical mismatch = 0 if equal, 1 if unequal. Boolean mismatch = 0/1. Unknown fields are omitted pairwise, never treated as a match.

Feature-group weights:
- topology/counts 0.50
- centre/layout categories 0.20
- text/star distribution 0.15
- distinctive booleans 0.15

The final score is `1 - normalized_distance` over observed shared features.

## Calibration gate — astronomical only

Before interpreting Voynich rankings, structural vectors must retrieve held-out known `astro_diagram` examples across manuscripts.

Deterministically select 8 `astro_diagram` entities with distinct manuscript keys. For each query, exclude its manuscript from candidates and calculate enrichment of `astro_diagram` among the top 20 versus the available held-out baseline.

**PASS iff:**
- median top-20 enrichment >= 1.5x, and
- at least 6/8 queries have enrichment > 1.0x.

No Aries/Pisces gate applies. This implements the user's correction that only astronomical calibration is required.

If the astronomical gate fails, rankings are archived as exploratory only and no semantic conclusion is promoted. If it passes, top candidates are manually image-audited against the frozen target morphology.

## Promotion rule

A candidate can be promoted only after manual audit shows at least three independent shared structural features **and** no single fatal mismatch on the feature that drove the retrieval. Generic circularity, shared parchment colour, decorative style, or a similar number alone cannot qualify.

Strength labels:
- strong structural comparator: >=5 independent matches, no fatal mismatch
- moderate structural comparator: 3–4 independent matches, no fatal mismatch
- weak/contextual: fewer than 3 or one major mismatch

No claim of genealogy, copying, plaintext identity, or subject identity follows from structural similarity alone.

## Compute discipline

GPU jobs have a hard timeout. Obsolete/failed jobs are cancelled explicitly. Full candidate vectors, calibration output and rankings are preserved before manual adjudication.