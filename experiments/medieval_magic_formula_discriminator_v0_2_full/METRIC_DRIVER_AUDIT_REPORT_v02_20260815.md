# Medieval Magic v0.2 — Metric-driver post-hoc audit

Status: `METRIC_DRIVER_AUDIT_COMPLETE`

Primary whole-corpus verdict remains `NO_ROBUST_MAGIC_AFFINITY`. This audit is secondary and uses the frozen 18-metric A-vs-C model only; B is excluded from driver attribution because the frozen B reference has one held-out test block.

## Main finding

The short-folio C affinity is not produced by entropy becoming charm-like. Instead, two separable effects coexist:

1. **Persistent C-like local/token-family geometry**: F2 token-family metrics, near-copy structure (F3), and some line-position metrics (F4) place Voynich closer to productive voces class C in both short and long folios.

2. **Strong length-dependent A-like compression-transfer pressure**: several F7 metrics become much more A-like as folios get longer. This is what drives the folio-to-section/corpus sign reversal in RF, FULL_STA and AAA. STA_FAMILY is anomalous and behaves differently.

Therefore the earlier statement that folio C affinity is 'largely just length' needs refinement: the **sign reversal** is largely length/compression driven, while a real C-like local family geometry survives length.

## Persistent C-like drivers

- `F3_nearcopy_lag10`: Q1 C-support 3/4 layers, Q4 C-support 3/4; median leave-one-out support in Q1 +0.1187.
- `F2_oneedit_component_frac`: Q1 C-support 4/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.1017.
- `F2_shared_core_ratio`: Q1 C-support 3/4 layers, Q4 C-support 3/4; median leave-one-out support in Q1 +0.0997.
- `F2_oneedit_degree`: Q1 C-support 4/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.0839.
- `F7_tok_bz2_ct_A`: Q1 C-support 4/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.0770.
- `F2_tok_len_std`: Q1 C-support 4/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.0622.
- `F2_tok_len_mean`: Q1 C-support 4/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.0438.
- `F4_line_medial_len`: Q1 C-support 3/4 layers, Q4 C-support 3/4; median leave-one-out support in Q1 +0.0339.
- `F4_init_final_jsd`: Q1 C-support 3/4 layers, Q4 C-support 4/4; median leave-one-out support in Q1 +0.0282.

The most robust are `F2_oneedit_component_frac`, `F2_oneedit_degree`, `F2_tok_len_std`, `F2_tok_len_mean`, `F3_nearcopy_lag10`, and `F2_shared_core_ratio`. Their C-like signal does not vanish in the longest quartile.

## What causes the length inversion

- RF_WORD: F7 family median mean qdiff Q1 -4.748 → Q4 -12.768.
- FULL_STA: F7 family median mean qdiff Q1 -5.080 → Q4 -13.771.
- AAA_CONNECTED: F7 family median mean qdiff Q1 -8.793 → Q4 -20.170.
- STA_FAMILY: F7 family median mean qdiff Q1 +2.614 → Q4 +13.369.

Positive qdiff means closer to C than A. In RF/FULL/AAA, F7 becomes dramatically more negative with length; in STA_FAMILY it instead becomes more C-like, explaining that representation's anomalous section-level behaviour.

The strongest individual length-decay metrics are `F7_tok_bz2_ncd_A`, `F7_tok_lzma_ncd_B`, `F7_char_bz2_ct_C`, `F7_tok_zlib_ct_C`, plus `F5_local_global_gain`. `F1_H0` is A-like at both short and long lengths and is not the source of the short-folio C effect.

## Section residuals after quadratic length adjustment

- Herbal-A: residual positive in 4/4 layers; median residual +0.0095; raw median ΔC +0.1858.
- UNMAPPED: residual positive in 3/4 layers; median residual +0.0688; raw median ΔC -0.0699.
- Stars: residual positive in 3/4 layers; median residual +0.0408; raw median ΔC -0.0785.
- Herbal-B: residual positive in 3/4 layers; median residual +0.0073; raw median ΔC +0.1995.
- Astronomical: residual positive in 2/4 layers; median residual -0.0045; raw median ΔC +0.1660.
- text-only: residual positive in 1/4 layers; median residual -0.0082; raw median ΔC -0.0516.
- Pharmaceutical: residual positive in 1/4 layers; median residual -0.0184; raw median ΔC +0.0667.
- Rosettes: residual positive in 1/4 layers; median residual -0.0474; raw median ΔC +0.0031.
- Zodiac: residual positive in 1/4 layers; median residual -0.0657; raw median ΔC +0.0630.
- Cosmological: residual positive in 0/4 layers; median residual -0.0199; raw median ΔC +0.1025.
- Balneological: residual positive in 0/4 layers; median residual -0.0277; raw median ΔC -0.0881.

After length adjustment, Herbal-A remains slightly C-like in 4/4 layers and Herbal-B in 3/4. Astronomical is essentially neutral. Cosmological and Zodiac lose their apparent positive affinity; Balneological remains negative. Stars becomes positive relative to its very long length in 3/4 layers even though its raw ΔC is A-like.

## Interpretation

The defensible mechanism-level statement is now narrower but more interesting: **Voynich has a representation-robust local family/near-copy geometry that resembles productive magical vocables more than ordinary prose, but its long-range/compression behaviour is ordinary-prose-like in three of four representations.** This does not identify magic as the cause, because non-magical family-structured systems could share the same F2/F3 geometry.

The decisive next control is therefore to challenge the persistent F2/F3 signal against non-magical structured morphology/slot systems, matched for token length and type count. If C remains uniquely close after that, the magical-formula mechanism gains real specificity; if not, the resemblance is generic family morphology.
