# Isaiah 1586 score-independent transcription-feasibility sample

Frozen 2026-09-12 before rendering or inspecting sampled PDF pages.

Source: HebrewBooks item 42163, *ישעיה עם אידיש טייטש*, Kraków 1586.
Direct-download object SHA-256: `8a33745204ff8454db5e704545bcf3135d08df44a3504d1954e2beaa8372d528`.
PDF pages: 203.

Selection rule: Python `random.Random(int(pdf_sha256[:16],16)).sample(range(5,199),6)`, sorted. This excludes likely covers/front/back matter without inspecting page content.

Frozen PDF page numbers (1-based):

- 41
- 49
- 85
- 131
- 170
- 174

Audit purpose only: determine print/layout quality, whether Yiddish and Hebrew layers are visually separable under a source-independent rule, and whether a diplomatic transcription pipeline is technically plausible. No language/invariant/target scores may be computed from this sample.

The source images are authoritative. OCR/vision output, if used, is draft-only until independently checked against the images.
