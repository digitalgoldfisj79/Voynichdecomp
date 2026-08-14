# ASC Section-Conditioned Residual Fingerprint v0.1 — preregistration

Date: 2026-08-14

## Question

After freezing the ReM + SWITCH_LINE cipher + short-lived K2 origin-state mechanism, what reproducible structure remains in Voynich that the mechanism does not explain, and is that residual shared across manuscript sections or section-dependent?

This is a diagnostic residual programme, not a new cipher search and not a decipherment attempt.

## Binding upstream constraints

1. Voynich is measured only from verified RF1b and its correlated RF-member/full-STA, STA-family, and connected-`aaa` projections.
2. `enriched_records.pkl` and the legacy 85-metric battery are prohibited.
3. The synthetic mechanism is not tuned: ReM v2.1, first 2000 tokens/document, W10, SWITCH_LINE, K2, tau=3, fixed line-reset POST canonical arm, fixed continuous POST sensitivity, and CIPHER_ONLY baseline.
4. Synthetic replicate indices are fixed prospectively to `[0,4,8,12,16]`, using the exact Phase-8 `P5-plan` and `P5-state` namespaces.
5. Both ATOMIC and LITERAL synthetic renderings are retained; robustness means surviving both.

## Primary conditioning

Primary unit is Voynich section, using the existing Voynich.nu-derived section map committed unchanged as `SECTION_MAP.json`.

Nine sections are tested:

- Herbal-A
- Herbal-B
- Astronomical
- Cosmological
- Zodiac
- Rosettes
- Balneological
- Pharmaceutical
- Stars

`text-only` is excluded, not reassigned. Every section must have >=400 clean RF words and >=15 clean segments. The manuscript-level primary aggregate is an **equal-section macro average**; pooled whole-manuscript results are secondary.

## Secondary conditioning

- Currier A/B comes directly from RF1b page-header `$L` metadata.
- Scribal hand comes directly from RF1b `$H=1..5`; the only `$H=@` page, f115r, follows the prior Davis sensitivity split: lines 1-12 S2, remainder S3.
- Section×hand and section×Currier cells are reported when >=300 clean tokens and >=10 clean segments.
- None of these secondary slices can alter the primary adjudication.

## Frozen residual panel

Six families are fixed in `protocol.json`:

1. exact-repeat recurrence at lags 1,2,3,4,8,12 plus decay contrasts;
2. ED1 lag profile at the same lags plus decay contrasts;
3. ED1 topology: substitution/insertion/deletion, first/interior/final position, equal-length share, directionality asymmetry;
4. token-family dynamics: shared first/last unit, local-vs-lag12 decay, length equality/closeness/difference;
5. physical boundary coupling: line-start/end length effects, cross-line versus within-line E1/ED1, and start/end unit-distribution JS divergence;
6. higher-order relational grammar: ED1 chains/lift, ABA return, ED1 return, triple repeat, token entropy, empirical H(next|prev)/H(token).

The panel is intentionally relational and representation-portable. No semantic word classes are used.

## Residual scoring

For each metric and synthetic representation, the 190-document CIPHER_ONLY distribution defines a frozen centre and scale:

`scale = max(1.4826 * MAD_baseline, metric_floor)`.

Every Voynich target is scored against CIPHER_ONLY, canonical mechanism, and continuous-state sensitivity using that same scale.

A section/metric is representation-robust residual evidence only when:

- ATOMIC and LITERAL residual signs agree;
- the weaker synthetic-representation absolute z is >=2;
- RF_MEMBER, STA_FAMILY, and AAA_CONNECTED all agree in sign and each passes the synthetic robustness condition.

A metric is **shared** if the same residual sign survives in >=6/9 sections.

A metric is **section-modulated** if robust section residuals reverse sign between at least two sections, or the robust absolute-z range is >=3 with a maximum >=4.

Mechanism gain is `baseline absolute distance - canonical absolute distance`; positive is improvement. Robust gain is the minimum across ATOMIC/LITERAL, then across the three Voynich projections.

## Adjudication

- >=3 shared and >=3 section-modulated metrics: `SHARED_PLUS_SECTION_SPECIFIC`
- >=3 shared only: `SHARED_RESIDUAL_DOMINANT`
- >=3 section-modulated only: `SECTION_SPECIFIC_RESIDUAL_DOMINANT`
- otherwise: `MIXED_OR_WEAK_RESIDUAL_FINGERPRINT`

The label is descriptive of the residual structure. It does not promote a historical mechanism.

## Interpretation rule

The next mechanism experiment, if any, must be chosen from a residual family that is:

1. representation-robust;
2. not already explained by the frozen origin-state arm;
3. either manuscript-shared or clearly section-conditioned under this programme;
4. testable by a one-variable intervention.

No post-hoc tuning of the present mechanism follows from this run.

## Execution note

A local implementation dry-run was used before this preregistration only to verify parsing, section support and code execution. It generated a target file but no residual metric values were inspected or displayed; only corpus-support counts were examined. The target file is discarded and the GitHub workflow recomputes all targets after the freeze gate.
