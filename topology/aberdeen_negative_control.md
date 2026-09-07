# Aberdeen Bestiary historical negative control

Date: 2026-09-07

## Purpose

Test whether ordinary nested parchment bifolia automatically show the textual-conjoint signal seen in MS 408.

Control manuscript: University of Aberdeen MS 24, Aberdeen Bestiary. Quires M (ff. 80-87) and N (ff. 88-95) are intact regular eight-folio gatherings. Folio-level Latin transcriptions were fetched directly from the University of Aberdeen site and frozen in Supabase with per-page SHA-256 hashes.

True conjoint mappings:

- M: 80/87, 81/86, 82/85, 83/84
- N: 88/95, 89/94, 90/93, 91/92

Null: every perfect matching of the eight folios inside each quire (105 exact matchings). Metrics: folio word-frequency cosine and within-token character-trigram cosine, matching the Voynich audit family.

## Results

Quire M:
- char3: observed .679457 vs null .682035 +/- .015797; effect -.002578 = -.163 null SD; exact p=.504762.
- word: .642316 vs .640548 +/- .013413; effect +.001769 = +.132 SD; p=.438095.

Quire N:
- char3: .643225 vs .645683 +/- .021154; effect -.002459 = -.116 SD; p=.485714.
- word: .590184 vs .611575 +/- .019794; effect -.021391 = -1.081 SD; p=.904762.

Pooled exact product null (11,025 M x N matching combinations):
- char3: observed .661341 vs null .663859 +/- .013200; effect -.002518 = -.191 SD; p=.537596.
- word: .616250 vs .626061 +/- .011955; effect -.009811 = -.821 SD; p=.789478.

**Mandatory conclusion: the metric does not resolve conjoint bifolia in either conventional Aberdeen quire.**

Therefore the strong Voynich bifolium signal is not a generic consequence of two leaves sharing one parchment sheet in an ordinary nested quire. This supports treating the Voynich bifolium as a production-functional unit, but does not by itself establish universal singulions or recover a linear order.

## Separate sequence calibration failure

Maximum textual-similarity open-path ordering remains disqualified. On matched-length continuous Latin (Caesar, De Bello Gallico I-IV), mean true-adjacency recovery was only .5921 for Q13-shaped windows and .6286 for Q20-shaped windows, with false-edge rates .4079 and .3714. Hence Q13/Q20 text paths are production-proximity summaries only.

## Downstream topology triage

The persistent Supabase registry currently marks 23 canonical result families:
- 8 SAFE
- 2 PROVISIONALLY_SAFE
- 6 AUDIT_REQUIRED
- 4 SUSPENDED
- 3 SUSPENDED_FOR_MECHANISM

The three suspended-for-mechanism families are R64 exact cache, hierarchical working-set repertoire, and latent-state trajectory.
