# VSN-B2-v1 — RF / STA / AAA Representation Robustness

Date: 2026-08-12
Status: robustness only; RF remains primary.

## Source

Existing frozen STA machinery was reused from `experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py`.

Reference Transliteration source:

- `RF1b.txt`;
- verified SHA-256: `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`.

AAA was produced with the pinned `bitrans.c` / `STA-aaa.bit` transformation used by the prior STA programme. Generated AAA SHA-256 in this run: `c14f43c731f46274f35b604356c6bb96a1186e0836aa9aa2b518666cce854167`.

Bounded CPU job: `6a7bc0c627caad61c6eaca91`. Final HF status: no running jobs.

## Type-level unit metrics

These representations have different token/unit alphabets and different exact type counts. They are therefore a robustness check on qualitative morphology and positional architecture, not a requirement for identical raw graph density.

| representation | types | units alphabet | mean unit len | edit pairs | mean degree | isolated | prefix | internal | suffix | H(unit|left pos) | H(unit|right pos) | H(next|prev) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF Basic EVA primary | 7,893 | 24 | 6.497 chars | 28,435 | 7.205 | 0.1491 | 0.2907 | 0.5278 | 0.1815 | 3.5734 | 3.4207 | 2.5355 |
| STA family | 4,940 | 23 | 5.677 | 28,047 | 11.355 | 0.0749 | 0.2757 | 0.5439 | 0.1804 | 2.9114 | 2.8642 | 2.4210 |
| full STA | 9,749 | 164 | 5.225 | 51,353 | 10.535 | 0.1331 | 0.3549 | 0.4548 | 0.1903 | 4.0164 | 3.8017 | 3.1293 |
| connected AAA | 8,298 | 135 | 5.111 | 47,449 | 11.436 | 0.1161 | 0.4063 | 0.4146 | 0.1791 | 3.8846 | 3.6901 | 2.9821 |
| Matteo K2 Latin chars | 7,893 | 24 | 4.913 | 26,094 | 6.612 | 0.1263 | 0.3048 | 0.5095 | 0.1857 | 3.8971 | 3.9713 | 3.3818 |

## Robust findings

### 1. Suffix share of exact one-edit neighbourhoods is highly stable

Voynich suffix-location share:

- RF: 18.15%;
- STA family: 18.04%;
- full STA: 19.03%;
- AAA: 17.91%.

Matteo K2: 18.57%.

Thus the K2 suffix-location agreement is not specific to literal EVA/RF character splitting.

Prefix/internal allocation is more representation-sensitive:

- RF: 29.07 / 52.78%;
- STA family: 27.57 / 54.39%;
- full STA: 35.49 / 45.48%;
- AAA: 40.63 / 41.46%.

The strongest like-for-like agreement remains at RF and STA-family scale. Full STA/AAA expose finer connected-unit distinctions and therefore alter exact one-unit edit topology substantially.

### 2. Voynich right-edge positional constraint survives every representation

`H(unit | right position) - H(unit | left position)`:

- RF: -0.1527 bits;
- STA family: -0.0472;
- full STA: -0.2147;
- AAA: -0.1945;
- Matteo K2: **+0.0742**.

The sign of the Matteo mismatch is therefore robust. Literal two-first-syllable concatenation does not generate the deeper right-edge positional architecture of Voynich.

### 3. Voynich local transition constraint also survives representation changes

`H(next unit | previous unit)`:

- RF: 2.5355;
- STA family: 2.4210;
- full STA: 3.1293;
- AAA: 2.9821;
- Matteo K2: 3.3818.

Absolute values shift with alphabet granularity, but Matteo remains less locally constrained than Voynich in every representation.

## Robustness verdict

The RF conclusion survives:

- the **minimal-pair / edit-location transfer is real enough to persist at the broad STA-family level**, especially the remarkably stable suffix share;
- the **right-edge and transition-constraint failures also persist** and are therefore not RF/EVA artefacts.

Accordingly the historical grammar remains **PARTIAL TRANSFER**. Representation robustness gives no basis to promote it to STRUCTURAL TRANSFER and no basis to dismiss the edit-graph result as a transliteration artefact.