# Anonymous whole-word equality source-transfer v0.2 — DEVELOPMENT RESULT

Status: **`ANON_WORD_V02_DEVELOPMENT_PASS`**.

This result was completed using only consumed Bovo 1541 and Basel 1599 material. No Shmuel-1544 confirmation image was loaded or viewed before this pass was recorded. Voynich remains sealed.

Protocol freeze: `f683611685f310447110ee2d9415885f4e751012`.
Visual-audit rubric freeze: `7c4416a9a798cb24cc999d573311e9f765897089`.
Executable v0.2 implementation: `5f593d881a981e32bebd210f0f392843d28cdc29`.

The sole v0.2 algorithmic repair is replacement of the Bovo-specific absolute vertical admissibility rule by `0.03*H < line_start < 0.97*H`. Descriptor, Bovo BUILD scaler definition, complete-link cosine clustering and `tau=0.410` are unchanged.

## Bovo numerical requalification

BUILD n51 under v0.2 is unchanged from v0.1:

- image N=157; Penn N=159;
- image TTR 0.719745223 vs Penn 0.666666667;
- image hapax fraction 0.522292994 vs Penn 0.522012579;
- image repeat fraction 0.477707006 vs Penn 0.477987421;
- image max-frequency fraction 0.038216561 vs Penn 0.062893082;
- image equal-pair density 0.005389515 vs Penn 0.010508717.

HOLDOUT n54 under frozen `tau=0.410` is also unchanged:

- image N=179; Penn N=175; relative token-count error **0.022857143** <=0.03 PASS;
- |delta TTR| **0.013120511** <=0.05 PASS;
- |delta hapax fraction| **0.039297686** <=0.05 PASS;
- |delta repeat fraction| **0.039297686** <=0.05 PASS;
- |delta max-frequency fraction| **0.012067039** <=0.02 PASS;
- |delta equal-pair density| **0.003011856** <=0.010 PASS;
- nondegeneracy PASS.

Bovo registered scan nuisance pages n82/n86/n120, eight perturbations each: **24/24 PASS** against >=22/24. Minimum ARI across the 24 cells = **0.968220554**; all token-count and matched-position gates pass.

Bovo source image SHA-256:

- n51 `fd38f8246786d962690484d746d67c8759f2883f046898961970fcf912935c9c`
- n54 `15c59bea51116e9e776d75b9407949f3f2f3c0b87afc34243d30696f9b517d07`
- n82 `c0272b51547495dd95ee556efb3b3ed6ef94dccc51c2c35a5873d74c88a0158f`
- n86 `e253f4f49ad95e480e493c00319c714bb857a2f2298e977696111a04c990d9c5`
- n120 `da4d4b4493265861f6cc549fd3adb336c5b03b50014eb51bc9d2af937ea7f156`

## Basel 1599 consumed DEVELOPMENT audit

The six v0.1 audit canvases were re-run under v0.2 and visually inspected using the separately frozen `AUDIT_RUBRIC.md`. Counts below are lines containing at least one retained word / retained words:

| Canvas | v0.2 lines | v0.2 words | Missing substantive running-body line | Included non-text substantive line | Obvious body word-boundary errors | Gate |
|---:|---:|---:|---|---|---:|---|
| 5 | 33 | 265 | no | no | 0 | PASS |
| 11 | 33 | 285 | no | no | 0 | PASS |
| 17 | 33 | 294 | no | no | 0 | PASS |
| 23 | 33 | 273 | no | no | 0 | PASS |
| 29 | 33 | 293 | no | no | 0 | PASS |
| 35 | 21 | 146 | no | no | 0 | PASS |

The visually unretained standalone material on canvas 5 (a separated section label) and canvas 35 (short separated heading continuation / closing formula) is paratext under the rubric frozen before confirmation access. It is not part of the registered running-body sequence. All running body lines are retained.

Basel audit source-image SHA-256:

- 05 `6eb8e5620c34e1328b7797186e5559b566fe19fa360df789573c86d9c9920502`
- 11 `141b19010e55ad3d8d34052e408f1cc57f0310bed990d24e640aa9f1a5e9a3b0`
- 17 `4f2882a45fd2f867b7f28cc7c92c06855f1cfb74d8532205651acb9956c32275`
- 23 `e093729b6895dbe0b179baaa331ae3f188bd27e15e16e6b5c4690db1d9422bbe`
- 29 `eb77d343f167e05baccf642f3b8d6541005c45d4e00f0360892796a687802bc0`
- 35 `c7c9f9ca0fab1a2555db604e8acb0ba413fd0ee4be09ec9ad09c8162c1bbe15a`

## Decision

Both required consumed-source development gates pass. Therefore the already-preregistered fresh confirmation witness, Augsburg 1544 `Shmuel-bukh` / BSB `bsb10170884`, may now be accessed under `PROTOCOL.md` without further detector or threshold changes.

This DEVELOPMENT pass is not historical-Yiddish population evidence, not R/L/T completion and not a Voynich result.