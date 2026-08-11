# Q9/Q10 Hybrid Structural Comparanda Protocol v0.4

Frozen 2026-08-11 before target rankings are exposed.

## Architecture
1. DINOv3 does broad candidate generation.
2. Explicit structural primitives rerank only the DINO shortlist.
3. Reranking is family-specific, not one global structural distance.
4. Calibration is held-out and target-blind.
5. Candidate metadata is interpretive context only after machine ranking.

## Candidate universe
Same date-bounded medieval comparanda universe as v0.2: 1250–1500, classes covering astronomical diagrams, astrology diagrams, Sun/Moon/star material, spheres/computus and relevant zodiac-wheel/sign material. Unresolved folios are never replaced by guessed neighbouring pages.

## Stage A — DINO retrieval
Model: `facebook/dinov3-vits16-pretrain-lvd1689m`.
Grayscale and edge streams are independently ranked and fused by reciprocal-rank fusion. Candidate pages use full and local views. Manuscript-diverse top 30 candidates are retained per Voynich target.

The previously successful astronomical-diagram calibration is repeated on eight deterministic held-out query manuscripts. Required retrieval pass: median top-20 enrichment >=2.0 and >1 enrichment for at least 6/8 queries.

## Stage B — Structural primitive extraction
Only the union of the top-30 DINO shortlists and the eight calibration-query shortlists is passed to `Qwen/Qwen2.5-VL-3B-Instruct`.

Primitive fields include counts for radial lines, sectors, compartments, rings, repeated outer units, axes, diagonals, corner roundels, luminaries, faces, figures, pipes and spiral bands; categorical centre/layout/text/star/boundary/object fields; and boolean flags for centrality, concentric structure, starry annulus, corner network, repeated faces, alternating star/text structure, connected star clusters, pipe rings, T-O partition and colour-coded axes.

Unknown values remain missing; they are never counted as matches.

## Stage C — Family-specific structural scoring
Frozen target families:
- F1 annular/concentric rota;
- F2 radial/compartment wheel;
- F3 star-field/heavenly-distribution diagram;
- F4 corner-linked/quadrant cosmology;
- F5 figure-bearing ring;
- F6 hybrid/other.

Each family has a fixed feature-weight table. Candidate family is derived from extracted primitives. Cross-family candidates are penalised but retained for audit. Hard mismatches (for example corner topology for F4 or gross count contradictions for count-defining F2 panels) receive an additional fixed penalty.

## Stage D — Hybrid weight calibration
The DINO/structure mixing coefficient is selected from `{0.35,0.45,0.55,0.65,0.75}` using only the eight held-out `astro_diagram` controls. Target rankings are not inputs to this choice.

For each control, DINO generates a manuscript-diverse top-30 shortlist and the hybrid reranker produces a top 10. Evaluation uses the externally stored `astro_diagram` class only as a held-out success label; class metadata is not an input to ranking.

Hybrid pass requires:
- median top-10 enrichment >=1.5;
- >1 enrichment in at least 6/8 held-out queries;
- median top-10 enrichment strictly greater than DINO-only on the same controls.

## Stage E — Target run
After the alpha is frozen on controls, the sixteen Voynich targets are reranked once. Ten results per target are retained with DINO rank, extracted primitive vector, structural score, family match, hard-mismatch reasons and final hybrid score.

## Promotion gate
New scale comparanda may be promoted only if:
- DINO retrieval calibration passes;
- hybrid reranker calibration passes;
- primitive extraction failure rate is <=10%.

If the promotion gate fails, rankings are preserved as exploratory retrieval output but are not historical evidence.

## Human adjudication
Only after the promotion gate is known are exact candidate folios inspected. Numerical rank alone cannot establish copying, subject identity, provenance or genealogy.
