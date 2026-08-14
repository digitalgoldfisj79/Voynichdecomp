# Voynich legacy ED1-chain replay v0.1 — scientific closeout

Date: 2026-08-14

## Authoritative execution

Corrected workflow run: `31817598862`

Head: `552c9443b59888104c0d48ae60bb424790f7c8d7`

Frozen protocol SHA-256: `d464cdc717e55d4233e2e5700be85b14fa2bc62a7691ac024b9e9bf98949533f`

Final artifact:
- name: `voynich-legacy-chain-replay-v01-corrected-final`
- artifact ID: `9225745769`
- digest: `sha256:c3c7ffd9647581b3236dbfa1195904224ef271557c3b26d976f2abf4fdea4cf8`

Run `31816720864` is superseded because its Timm parser admitted the published generator's `#Properties/#Statistics` metadata preamble as pseudo-text. The correction is parser-only and independently verified: pinned Timm seed 19 yields exactly 1,200 generated lines and 10,832 tokens, matching the previous faithful audit.

## Preregistered formal adjudication

`CHAIN_LIFT_NEW_MECHANISTIC_GAP_WITHIN_TESTED_LEGACY_ARMS`

| arm | median raw `ed1_chain_lift` | ratio to VMS | own within-line-shuffle z | formal status |
|---|---:|---:|---:|---|
| VMS | 0.08784431 | 1.000 | +0.29 | target |
| G-prime | 0.06318411 | 0.719 | +0.02 | MODEL_PARTIAL |
| faithful Timm default | 0.02583360 | 0.294 | -0.73 | MODEL_FAIL |
| Timm `reuse_last=0` | 0.05034252 | 0.573 | +0.25 | MODEL_PARTIAL, diagnostic only |
| Timm random source | -0.00453013 | -0.052 | -0.82 | MODEL_FAIL |
| Timm position source | 0.01945435 | 0.221 | -0.30 | MODEL_FAIL |

Q57b is `NOT_EXACTLY_REPLAYABLE` because the archived `q56_injective_anonymous_realiser.py` dependency was not recovered. No successor generator was substituted.

## Load-bearing control result

The formal label above must not be read as evidence for a new sequential chain-walk mechanism.

The Voynich target's raw `ed1_chain_lift` is **0.08784431**, but 500 exact within-line token shuffles have mean **0.08524750**. The frozen z score is only **+0.29**. Thus destroying the order of tokens within every physical line leaves essentially all of the apparent chain lift intact.

The same is true of the legacy mechanisms:
- G-prime: z **+0.02** against its own within-line shuffles;
- faithful Timm: z **-0.73**;
- no-reuse Timm: z **+0.25**;
- random-source Timm: z **-0.82**;
- position-source Timm: z **-0.30**.

Therefore `ed1_chain_lift` in this form is primarily a statistic of **line-local ED1 graph composition / inventory topology**, not transition ordering.

## Section sensitivity of the control — descriptive posthoc diagnostic

The same 500-shuffle calculation was repeated section by section after the primary run. No section has |z| >= 2:

| section | observed lift | shuffle mean | z |
|---|---:|---:|---:|
| Astronomical | -0.0284 | 0.0097 | -0.69 |
| Balneological | 0.0156 | 0.0316 | -0.90 |
| Cosmological | 0.5547 | 0.5651 | -0.25 |
| Herbal-A | 0.0413 | 0.0263 | +1.05 |
| Herbal-B | 0.0177 | 0.0008 | +0.53 |
| Pharmaceutical | 0.0911 | 0.0392 | +1.58 |
| Rosettes | 0.0272 | 0.0241 | +0.05 |
| Stars | 0.0318 | 0.0256 | +0.45 |
| Zodiac | 0.0395 | 0.0036 | +0.92 |

The extreme raw Cosmological value is especially diagnostic: its random-within-line expectation is equally extreme. It reflects an unusually ED1-dense line inventory, not an unusual sequence traversal.

## Revised novelty judgment

The prior crosswalk's statement that `ed1_chain_lift` was the clearest likely-new structural property is **withdrawn as a substantive novelty claim**.

The exact statistic remains a new measurement in this programme, and the RF-member / STA-family / connected-aaa residual against transformed ReM remains a valid representation-robust model discrepancy. But the permutation control shows that the discrepancy does not isolate higher-order sequential dependence.

Its substantive content collapses onto already-established programme findings:
- local/page/line-scoped word-family concentration;
- ED1-rich working sets;
- section-dependent magnitude;
- non-exchangeable placement of individual variants under page-preserving permutation controls.

Hence there is **no earned basis here for inventing a new chain-continuation generator**.

## Mechanistic reading of the legacy replay

G-prime's raw partial fit is informative but unsurprising: its regional working-set construction creates ED1-rich local inventories, so it inherits positive raw chain lift. Its own shuffle z≈0 shows it does not generate special chain ordering either.

Faithful Timm is more revealing. Its default output has a higher ordinary adjacent ED1 rate than Voynich, but much smaller raw chain lift and negative excess over its own shuffle expectation. The frozen `reuse_last` ablation actually raises raw chain lift while the prior tight-null audit showed that removing `reuse_last` lowers ordinary adjacent ED1 enrichment. Thus Timm's explicit adjacent derivative channel creates **isolated near-neighbour adjacency**, not sustained ED1-chain continuation.

That is a useful mechanism distinction, but it does not create a new Voynich anomaly because the target's chain continuation itself is null after conditioning on line inventory.

## Stop condition

Do not launch a chain-walk / Markov-ED1 continuation mechanism.

If further work is desired, the remaining legitimate question is narrower: quantify **how ED1-rich forms become co-located within physical lines relative to page inventory**, while conditioning away sequence order. Before constructing such a test, cross-check the existing conditional-placement/page-ensemble programme, which already fixed page multisets and line layout and rejected exchangeability of observed variants. Any new statistic must add information beyond that prior result.
