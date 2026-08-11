# Q9/Q10 Structural Comparanda v0.3 — Results

Date: 2026-08-11

## Result

The explicit structural-feature extractor succeeded, but the standalone global primitive-distance retrieval metric failed the preregistered astronomical calibration gate.

### Corpus and extraction

- Frozen candidate universe: 903 `comparanda_illuminations` rows dated wholly within 1250–1500 and restricted to astronomical/cosmological classes.
- Final exact/resolved unique image URLs: 393.
- Final usable unique images: 315.
- Qwen2.5-VL-3B structural vectors extracted: **315/315**.
- Extraction failures: **0**.
- JSON repair prompts: **0**.

This validates the extraction layer as a stable way to convert manuscript diagrams into auditable structural metadata.

### Frozen astronomical calibration

Eight `astro_diagram` queries from distinct manuscripts were selected deterministically. Each query manuscript was held out. The top-20 `astro_diagram` fraction was compared with the available held-out manuscript baseline.

Frozen PASS rule:

- median enrichment >= 1.5x; and
- at least 6/8 queries above 1.0x baseline.

Observed:

- median enrichment = **0.9926168992387415x**
- queries above baseline = **3/8**
- verdict = **FAIL**

Therefore v0.3 Voynich rankings are exploratory only. They are not promoted as new comparanda.

## Interpretation

The negative result is specific to the global hand-designed distance metric, not to primitive extraction. A single weighted distance over centre type, radial/sector/ring counts, repeated units, text/star layout, boundaries and distinctive motifs does not organize the heterogeneous medieval astronomical-diagram corpus into sufficiently coherent families.

This does not overturn the earlier v0.2 DINOv3 result, where `astro_diagram` held-out calibration passed (median enrichment 2.6x; 7/8 above baseline). The technically defensible next architecture is therefore hybrid:

1. use calibrated DINOv3 for broad candidate retrieval;
2. use explicit primitive vectors inside the DINO shortlist for transparent reranking, mismatch analysis and explanation;
3. require direct image audit against the frozen Voynich morphology before promotion.

The failed v0.3 thresholds are not relaxed retrospectively.

## Reproducibility

Final Hugging Face job: `6a7a8c6d3b2516b29b154c34`

Runner SHA-256: `5f0ba4ab3ab021ed3c80064d156a14be13b9d09e31449bc8528edffff9bec52f`

Protocol commit: `434dfd7080c887a027bbdf86819ebe4ce73425ce`
