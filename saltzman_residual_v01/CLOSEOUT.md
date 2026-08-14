# ASC Section-Conditioned Residual Fingerprint v0.1 — closeout

Date: 2026-08-14

## Execution status

Scientific scoring completed successfully with frozen protocol SHA-256 `b600d0e80beecc609823e2b24f303eec45c7e8862fae7b14ca6f725ff7b67dd0` and runner SHA-256 `36f727cf1594fe015eb4bcb30c2d564a42641997047f73f2d0dc8e9eee11154a`.

The main workflow run `31810009033` passed freeze, canonical ReM reconstruction, and all 12 synthetic scoring shards (190 distinct ReM documents). Its RF source job stopped before Voynich scoring because the workflow environment contained a malformed `bitrans.c` checksum string. The preserved 12 shards were not rerun.

Recovery run `31810157523` used the correct previously verified `bitrans.c` SHA-256 `3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a`, reverified RF1b / STA-aaa / regenerated connected-aaa provenance, computed the frozen Voynich fingerprints, downloaded the 12 immutable score shards from run `31810009033`, verified 12/12 shards and 190/190 documents, and executed the unchanged frozen merge.

Final artifact:
- name: `asc-section-residual-v01-final-recovered`
- artifact ID: `9222783937`
- digest: `sha256:15e2cd4aa892aa64f3dd9fdd860adac030ee15b7d2a274e485510fbfcee1a052`

## Preregistered adjudication

`SHARED_PLUS_SECTION_SPECIFIC`

Counts:
- representation-robust manuscript-shared residual metrics: **12**
- representation-robust section-modulated residual metrics: **12**

The phrase `SECTION_SPECIFIC` must be interpreted carefully: **none of the 12 section-modulated metrics showed a robust sign reversal between sections.** Every modulation call was caused by magnitude differences under the preregistered range criterion. The result therefore supports a common residual architecture whose strength varies substantially by section, not separate section-specific mechanisms pointing in opposite directions.

## Shared residual metrics

A shared metric requires the same RF-member / STA-family / connected-aaa residual sign in at least six of nine sections, surviving both ATOMIC and LITERAL synthetic renderings at robust |z| >= 2.

| metric | sections | residual sign (Voynich - frozen mechanism) |
|---|---:|---|
| `ed1_rate_l1` | 9/9 | positive |
| `ed1_rate_l2` | 7/9 | positive |
| `ed1_rate_l3` | 8/9 | positive |
| `ed1_rate_l4` | 6/9 | positive |
| `ed1_directional_asym` | 6/9 | negative |
| `len_close_l1` | 9/9 | positive |
| `len_diff_mean_l1` | 9/9 | negative |
| `start_len_minus_interior` | 6/9 | positive |
| `withinline_ed1_rate` | 9/9 | positive |
| `start_first_unit_js` | 9/9 | positive |
| `ed1_chain_lift` | 7/9 | positive |
| `h1_token_bits` | 7/9 | negative |

Thus the frozen mechanism systematically underproduces raw local ED1 relationships, underproduces similarity in adjacent token length, underproduces line-start differentiation, and underproduces ED1-chain dependence. Where ED1 directionality is estimable, the synthetic model is more one-way than Voynich.

## Section modulation

Twelve metrics met the magnitude-modulation criterion; none reversed robust sign:

- `ed1_rate_l1`
- `ed1_rate_l2`
- `ed1_rate_l3`
- `ed1_rate_l4`
- `ed1_directional_asym`
- `len_equal_l1`
- `start_len_minus_interior`
- `withinline_ed1_rate`
- `start_first_unit_js`
- `ed1_chain_lift`
- `h1_token_bits`
- `h2_cond_ratio`

The strongest outlying section is Cosmological. In RF-member units its raw ED1 rates are approximately 0.138, 0.140, 0.157, 0.164 at lags 1–4, versus frozen canonical synthetic medians around 0.007–0.009. Its ED1-chain lift is also exceptionally large. This is a magnitude result only; it is not a semantic interpretation.

## Family-level macro result

Equal-section macro summaries (positive gain = origin-state mechanism improves over cipher-only):

| family | median canonical distance | median mechanism gain |
|---|---:|---:|
| repeat recurrence | 0.900 | -0.1875 |
| ED1 lag | 5.919 | -0.5010 |
| ED1 topology | 1.643 | -0.2514 |
| token family | 9.047 | 0.0000 |
| boundary | 3.570 | -0.0543 |
| higher order | 1.533 | 0.0000 |

Therefore the origin-state operator that improved the earlier normalized positional-attenuation Q statistic does **not** explain the broader residual fingerprint. On the raw ED1-lag family it moves the model farther from Voynich. This distinction is central: matching the Q attenuation shape is not the same as reproducing Voynich's absolute local-neighbour density or higher-order relational grammar.

Exact-repeat recurrence is not a strong representation-robust residual discriminator in this panel: no repeat-recurrence metric met the shared or section-modulated robust residual criterion.

## Main mechanistic implications

The missing structure is concentrated in four linked properties:

1. **Dense local word-family neighbourhoods.** Voynich has substantially more raw ED1 relations at lags 1–4 than the frozen transformed ReM model, across essentially every section.
2. **Adjacent length coupling.** `len_close_l1` is higher and mean adjacent length difference lower in all nine sections. This survives RF/STA/aaa and is untouched by the origin-state operation.
3. **Physical line-start grammar.** Line-start tokens are longer relative to interiors in at least six sections and their first-unit distribution is much more distinct from interior positions in all nine sections. The origin-state operation barely affects this.
4. **Higher-order relational chaining.** ED1 transition lift is higher in at least seven sections; Voynich is not just rich in isolated one-edit pairs but in local chains of related forms.

The shared direction with section-dependent magnitude is more compatible with a manuscript-wide production architecture whose activation/intensity is register- or section-dependent than with wholly separate section mechanisms.

## Secondary Currier / hand sensitivity

Currier A and B and hands S1–S5 retain many of the same dominant residuals. In particular, reduced ED1 directionality, line-start first-unit divergence, and adjacent-length coupling recur across Currier/hand slices. These are descriptive sensitivities only and do not affect the primary adjudication. No claim is made that Currier or hand independently causes the section effects.

## Stopping / next-step discipline

Do not tune the Saltzman origin-state mechanism against these residuals. The next one-variable mechanism experiment, if pursued, should target a residual family that is manuscript-shared, RF/STA/aaa robust, and mechanistically distinct from the already-tested origin-state operation.

The cleanest next target is **local token-family production / modification**: a mechanism that can increase raw ED1 density and ED1-chain lift while preserving adjacent length coupling, with line-start conditioning tested separately rather than bundled. The especially strong Cosmological magnitude is a preregistered section result to explain, not a parameter-selection target.
