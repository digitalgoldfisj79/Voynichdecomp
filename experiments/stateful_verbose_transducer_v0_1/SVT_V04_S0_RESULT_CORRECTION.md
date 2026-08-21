# SVT v0.4 S0 — corrected binding result

Date: 2026-08-21

## Retraction / correction

The GitHub Actions run `32473481147` is red, but that status is **not** a scientific failure of the S0 hidden-segmentation gate.

All eight binding trial jobs completed successfully and uploaded their immutable JSON artifacts. The aggregate computation step also completed successfully. The workflow then failed in `actions/upload-artifact@v4` while attempting to upload the aggregate summary, so the subsequent enforcement step was skipped. The red run therefore reflects CI plumbing after the scientific computation, not the S0 verdict.

This correction is binding and should be read before any earlier statement that “v0.4 failed”.

## Reconstructed frozen S0 gate

The eight preserved trial artifacts are sufficient to reconstruct the preregistered aggregate exactly.

| arm | replicate | boundary F1 | unit-count relative error |
|---|---:|---:|---:|
| periodic | 17000 | 0.9447568640 | 0.0305989583 |
| periodic | 17001 | 0.9280958722 | 0.0429687500 |
| periodic | 17002 | 0.9720486682 | 0.0188802083 |
| periodic | 17003 | 0.9569190601 | 0.0039062500 |
| line_reset | 17000 | 0.9605263158 | 0.0195312500 |
| line_reset | 17001 | 0.9479578393 | 0.0221354167 |
| line_reset | 17002 | 0.9219009638 | 0.0397135417 |
| line_reset | 17003 | 0.9608753316 | 0.0351562500 |

Aggregate:

- mean boundary F1: **0.9491351144**;
- median boundary F1: **0.9524384497**;
- minimum boundary F1: **0.9219009638**;
- trials with F1 >= 0.85: **8/8**;
- mean unit-count relative error: **0.0266113281**.

Frozen S0 requirements were:

- mean F1 >= 0.90;
- median F1 >= 0.90;
- minimum F1 >= 0.85;
- 8/8 trials >= 0.85;
- mean unit-count relative error <= 0.05.

**Binding S0 verdict: PASS.**

The legacy transition-surprisal segmenter remained around 0.68 F1; the semi-Markov S0 model is the qualified segmenter.

Voynich was not opened in this gate.
