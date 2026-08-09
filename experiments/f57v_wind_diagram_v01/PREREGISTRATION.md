# PREREGISTRATION — f57v Wind-Diagram Comparator v0.1
Date frozen: 2026-08-09

## Question
Is Voynich MS f57v morphologically closer to independently attested medieval wind diagrams than to non-wind circular diagrams, after controlling for generic medieval rota geometry and manuscript style?

## Claims deliberately separated
H1 (family-level): f57v instantiates visual grammar characteristic of medieval wind diagrams.
H2 (identification): f57v actually denotes winds / atmospheric directions.
H3 (direct copying): f57v descends from a particular exemplar.

v0.1 can test H1. It can provide supporting but not decisive evidence for H2. It cannot establish H3.

## Target
Voynich f57v, supplied colour image. SHA-256 recorded in DATA_PANEL.md.
The target is not used to define features, weights, thresholds, corpus classes, or acquisition rules.

## Primary comparator design
Use matched manuscript strata. Each qualified stratum contains:
- >=1 independently catalogued/identified wind diagram; and
- >=1 non-wind circular or rota-like diagram from the same manuscript when available.

Hard negatives are intentionally included: zodiac wheels, lunar/computus diagrams, elemental/humoral cosmologies, constellation/head medallion circles, planetary wheels, Apuleian spheres, and other circular diagrams with anthropomorphic imagery.

Minimum qualified primary panel:
- >=6 independent manuscript strata;
- >=12 wind-positive images total;
- >=24 matched controls total;
- >=3 late-medieval wind images dated 1350–1500;
- >=2 northern-Italian/Alpine/Central-European wind images if available from authoritative repositories.

No Voynich-adjacent website may supply a comparator label.

## Primary feature space
All primary features are visual and must be codeable without reading inscriptions:
- dominant circular/rotational layout
- concentric annular organization
- radial sectoring/spokes
- fourfold orthogonal human layout
- human/face personifications at radial positions
- inward/radial orientation of personifications
- explicit exhalation/breath stream
- central emblem/body
- centre-directed relation between outer figures and centre
- modular 4/8/12 organization
- annularly arranged writing/marks (presence only; semantics ignored)
- symmetry around a common centre

Each feature is scored in {0, 0.5, 1}; missing/indeterminate = blank. Equal weights. Feature definitions are frozen in feature_schema.csv.

Semantic features (readable wind names, cardinal labels, textual descriptions) are metadata-only and excluded from the primary statistic.

## Blinding
Before coding, image files are assigned random opaque IDs. The coder receives image pixels only, without shelfmark, repository, filename, class label, catalogue text, or transcription. A sealed key maps opaque IDs to source metadata.

The target is mixed into the blind packet. Its identity may be visually recognizable, but no scoring rule can be altered after coding begins.

## Primary statistic
Distance = equal-weight Gower distance over available frozen primary features.

For manuscript stratum m:
  Delta_m = mean distance(target, controls_m) - mean distance(target, winds_m).

Delta_m > 0 means the target is closer to the wind exemplar(s) than to the matched controls in that manuscript.

Primary inferential test: exact one-sided sign test over independent manuscript strata, H0 P(Delta_m > 0)=0.5.

Primary success requires ALL:
1. positive-control qualification passes;
2. >=6 qualified independent strata;
3. exact one-sided sign-test p <= 0.05;
4. median Delta_m >= 0.08;
5. at least one anthropomorph-specific ablation remains positive.

## Positive-control qualification
Leave-one-stratum-out pseudo-target test:
For each wind-positive image, compare its mean Gower distance to wind images in other strata with its mean distance to controls in other strata. A pseudo-target is correctly classified if it is closer to the other wind images.

Qualification threshold: >=75% accuracy overall AND >=60% accuracy among late-medieval (1350–1500) positives. If this fails, f57v is not tested and the experiment returns NO TEST.

## Required ablations
A. Geometry-only: circle/annuli/sectoring/symmetry/annular writing.
B. Anthropomorph-only: fourfold humans, radial personifications, inward orientation, exhalation, centre-directed relation.
C. Remove exhalation feature.
D. Remove all centre-related features.

For support of H1, the primary result must not be explainable solely by A. At minimum the anthropomorph-only median Delta_m must be >0.

## Secondary machine-vision analysis
Only after the primary feature schema and thresholds are frozen:
- DINOv3 image embeddings (or closest current self-supervised vision model if DINOv3 is unavailable);
- no OCR and no text transcription;
- cosine distance;
- same matched-stratum statistic;
- full-diagram, text-masked, and figure/centre crops;
- leave-one-manuscript-out positive-control qualification.

Machine-vision results are corroborative, not replacements for the primary test.

## Temporal/geographic sensitivity
Repeat the descriptive Delta analysis on:
1. 1350–1500 only;
2. northern Italy / Alpine / Central Europe only;
3. 1350–1500 AND those regions, if >=3 independent strata exist.

No p-value is claimed for subsets smaller than 6 strata.

## Interpretation
Strong family-level support: primary endpoint passes + anthropomorph ablation positive + machine-vision direction concordant.
Weak/ambiguous support: primary endpoint positive but fails margin/ablation/model concordance.
No support: Delta median <=0 or sign test fails.
NO TEST: positive-control qualification fails or <6 qualified strata.

Even a strong result does not establish a direct exemplar, textual meaning, or wind names in Voynichese.
