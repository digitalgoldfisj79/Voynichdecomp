# Amendment 010 — Uniform blind-description repair

Date: 2026-08-08
Run: `corridor_v01_20260808_run01`
Timing: frozen before any corridor-to-Voynich text-description similarity has been computed.

## Problem discovered

The legacy Stage 4 descriptions are not comparable across geography: in the live structural classes the control-side records are frequently the placeholder string `neutral morphology only`, while corridor-side map-recovery records contain substantive morphology descriptions. Using the legacy strings would therefore encode pipeline/provenance differences into the text family.

This is an implementation defect discovered before target similarity, not a scientific outcome.

## Repair

Generate a fresh description for **every one of the 78 sealed comparator crops** using one identical, identity-blind VLM pass. Generate the frozen VMS reference descriptions in the same job with the same image preprocessing and prompt.

The description model receives only the isolated crop. It does **not** receive:

- candidate/manuscript identity;
- corridor/control label;
- place, date, institution or language;
- visual class;
- VMS/reference status;
- previous descriptions or identifications.

### Model

`Qwen/Qwen2.5-VL-7B-Instruct`

Frozen decoding: deterministic (`do_sample=false`), maximum 120 new tokens.

### Crop preprocessing

- use the already sealed bbox only;
- RGB conversion;
- square white padding;
- deterministic upscale so the longest dimension is 768 px (no downscale below native if already larger is unnecessary; final square passed at bounded resolution);
- no colour manipulation, segmentation or hand retightening.

### Prompt

> Describe only the visible morphology and geometry of this isolated manuscript illustration crop. Do not identify or guess the manuscript, place, date, artist, language, subject name, species, zodiac sign, culture, or relationship to any known manuscript. Do not use proper nouns. Ignore text content and colour unless colour itself defines a visible boundary. Focus on topology and arrangement: shapes, branching, leaves/stems/roots if present, circles/rings/spokes/connectors, enclosures, towers/roofs/walls/flags, figures/vessels, repeated elements, symmetry and relative layout. Return JSON only: {"description":"8–35 neutral words","tags":["up to 8 short morphology tags"]}.

The same prompt is used for comparator and VMS crops.

## Frozen manifests

Comparator corpus:

- `stage5_confound_manifest.tsv` — 59 sealed multi-page objects;
- `stage5_singlepage_manifest.tsv` — 19 sealed single-page map objects.

Together these are the 78-object Stage 5 visual corpus sealed at SHA-256 `63a1dd441c11b92f7851d367c7cf58aca1a446e1b5ae25c1c9e93ee436504669`.

VMS description-reference manifest:

- `stage5_vms_text_refs.tsv`

It contains the two already verified structural references and the eight pre-existing reviewed/usable whole-plant cutouts. No new VMS object is selected from image appearance during this repair.

## Leakage gate

Before similarity, generated descriptions are scanned for forbidden manuscript/place/institution/Voynich identifiers. Any leaking record is excluded rather than edited. The pass is nonresolving if fewer than 90% of comparator crops or fewer than one target reference for each intended live class survive acquisition/parsing/leakage QA.

## Live confirmatory text classes

A class may enter text scoring only if, after uniform description generation:

1. it has at least one frozen VMS reference;
2. it has comparator objects from both corridor and control geographies;
3. a cross-manuscript non-VMS within-class similarity null can be estimated.

Expected potentially live classes from the sealed corpus are `architecture_cartography`, `diagram_geometry`, and `plant`. Root/zodiac/other classes remain unavailable for confirmatory geography comparison unless the already sealed corpus independently satisfies these rules; no new objects are added.

## Scoring remains Protocol §10

Descriptions will be embedded with a separately frozen text encoder; target/comparandum cosine similarities will be null-calibrated within visual class using only cross-manuscript non-VMS pairs before top-k manuscript aggregation. Raw cross-class similarity is prohibited.
