# Q9/Q10 Scale Comparanda Programme — v0.2 Result

Date: 2026-08-10
Branch: `experiment/q9q10-scale-comparanda-v0.1-20260810`
HF v0.2 job: `6a7a4b963b2516b29b154689`
Runner SHA-256: `ae2507cfafdabdbbe7d01f1e1aa661b41e99b9fb473f73314f118f78a83343b6`

## Verdict

**NO-GO for generic DINOv3 visual-nearest-neighbour retrieval as a validated scale-comparanda engine for all Q9/Q10 panel types.**

The corrected v0.2 representation passed the independent `astro_diagram` known-answer calibration but failed both zodiac controls. Under the frozen protocol, overall calibration therefore failed and the downstream Qwen morphology-adjudication stage was not run.

This does **not** falsify the existing source-secure comparanda literature audit. It means the tested embedding retrieval representation is not reliable enough to promote new target rankings to historical comparanda.

## Candidate universe and acquisition

Frozen candidate predicate:
- `public.comparanda_illuminations`
- `date_start >= 1250`
- `date_end <= 1500`
- classes: `astro_diagram`, `sun_moon`, `sun`, `moon`, `star`, `sphere_heavens`, `computus_table`, `astrology_diagram`, `zodiac_wheel`, `zodiac_aries`, `zodiac_taurus`, `zodiac_pisces`

Results:
- database rows: **1,340**
- unique relevant manifests: **238**
- manifests fetched successfully: **171**
- unique image URLs resolved without guessed pagination: **500**
- unique visual entities successfully downloaded after duplicate-image collapse: **420**
- unresolved rows: **645** manifest unavailable + **142** folio unresolved
- image download failures: **80** (43 HTTP, 37 connection)

The effective visual coverage is therefore 420 unique images from a 1,340-row frozen candidate universe. Missingness is substantial and non-random across repositories.

## Representation

Model: `facebook/dinov3-vits16-pretrain-lvd1689m`.

v0.2 deliberately removed colour and used two independently ranked streams:
1. grayscale/autocontrast;
2. deterministic edge/line-art transformation.

Rows resolving to one image were collapsed. Rankings shown to the target stage were manuscript-diverse: at most one image per manuscript. Candidate pages were represented by full-page, central and overlapping local crops; Voynich targets by frozen full and central panel views. Streams were fused by reciprocal-rank fusion.

## Independent known-answer calibration

Calibration used labelled medieval images, never Voynich targets. Each class used 8 deterministic query manuscripts; the query manuscript was held out entirely. Pass rule per class: median top-20 enrichment >= 2.0 and >1 enrichment in at least 6/8 queries.

| Class | Median enrichment | Queries >1 | Required | Verdict |
|---|---:|---:|---:|---|
| zodiac_aries | **0.90** | **3/8** | >=2.0; >=6/8 | FAIL |
| zodiac_pisces | **1.256** | **5/8** | >=2.0; >=6/8 | FAIL |
| astro_diagram | **2.60** | **7/8** | >=2.0; >=6/8 | PASS |

Overall calibration: **FAIL**.

The failure is informative: DINOv3 grayscale/edge retrieval can detect a broad astronomical-diagram family across manuscripts in this corpus, but it does not robustly recover even conventional Aries/Pisces iconography under manuscript holdout. It is therefore not calibrated for the heterogeneous Q9/Q10 problem as a whole.

## Target rankings

Target rankings were still computed before the gate, as preregistered, but are **exploratory only**. They were not passed to Qwen and must not be described as validated comparanda.

Repeated machine-retrieved leads included:
- Lyon BM Ms 172 astronomical diagrams (especially ff. 20v, 29, 53);
- Munich BSB Clm 14504 scientific miscellany (especially ff. 14r, 137v);
- Leipzig UB Ms 1483 f. 2r (1442);
- Leipzig UB Ms 1164 astronomical/medical-astrological material;
- Munich Cgm 120 f. 1v;
- Tournai Séminaire Ms 12 f. 3;
- Erfurt CA 4° 374 f. 60v;
- Gotha Chart. A 472 astronomical-instrument diagrams.

Because calibration failed, these identities are recorded only to make the run auditable. Their recurrence cannot be interpreted as evidence of copying, subject identity, workshop relation, regional origin, or even close iconographic similarity.

## v0.1 audit

v0.1 was also a NO-GO, for two reasons:
1. its target-specific 'calibration' failed badly for the two zodiac targets (Aries 0/20; Pisces 1/20 among top 20), while f68v3 showed broad astronomical enrichment;
2. the raw rankings were visibly distorted by duplicate database rows and same-codex/style dominance.

The attempted Qwen stage in v0.1 had a parser defect that produced zero numerical totals. Those zeros are **invalid scores**, not evidence of no visual similarity. The job was cancelled rather than interpreted.

v0.2 corrected those defects before rerunning. Its independent calibration still failed, establishing the stopping point without relying on the defective v0.1 scorer.

## Scientific consequence

The scale experiment supports a narrow conclusion:

> **Generic foundation-model image similarity is not presently validated as a universal discovery engine for the Q9/Q10 medieval comparanda problem.**

It does not alter the existing evidence hierarchy in which source-secure subject matches, exact morphological inspection, and programme-level manuscript comparisons carry the weight. The next defensible scale approach would be explicit diagram-feature retrieval (centre type, partition count, radial/ring topology, text placement, figure/object class) with independently labelled controls, rather than further post-hoc tuning of the failed embedding threshold.

## Compute discipline

- Initial sequential-acquisition v0.1 job was cancelled once network latency made the implementation inefficient.
- Optimized v0.1 was cancelled after the calibration gate had already failed and the Qwen parser defect was observed.
- v0.2 completed normally in ~90 seconds with its hard timeout intact.
- No HF job from this programme was left running.
