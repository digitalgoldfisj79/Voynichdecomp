# v0.6 Family P1 — initial fully blind development result

Date: 2026-07-16

Verdict: **FAIL. SINGLE DEVELOPMENT AMENDMENT ACTIVATED.**

No test data or Voynich text was scored.

## Job

Hugging Face job: `Digitalgoldfish79/6a588abdb1669a49bf076fe7`

Scientific SHA-256: `4d2999963a1627257673f2e72d0f7cdfd46bd583f22ee45d1856550fafdc7725`

## Frozen configuration

- English development split;
- 16 ciphertexts of length 384;
- 8 continuous-periodic and 8 line-reset schedules;
- all 22 mode-period structures compared;
- `250,000 × 24` joint simulated-annealing search per structural candidate;
- train-only trigram-plus-unigram objective;
- BIC-like period penalty.

## Results

- mean plaintext recovery: **14.5996%**;
- median: **4.6875%**;
- minimum: **1.3021%**;
- trials at least 80%: **1/16**;
- trials at least 90%: **1/16**;
- exact plaintexts: 0/16;
- operating-mode accuracy: **13/16**;
- period accuracy: **12/16**;
- exact mode-plus-period accuracy: **11/16**.

One line-reset period-two trial reached 99.48% recovery, demonstrating that the correct joint basin exists, but the hit rate is only 6.25%.

## Diagnosis

The component oracles recover both channels almost perfectly:

- true wheel, unknown structure and shifts: 100%;
- true schedule, unknown wheel: 99.51% mean.

The blind solver also identifies the exact structural schedule in 68.75% of trials, including many trials whose plaintext recovery remains near random. Therefore the primary failure is not mode or period identifiability. It is the simultaneous high-dimensional search over the mixed wheel and its phase shifts.

## Permitted amendment

The single development amendment replaces simultaneous local moves with an explicit coordinate architecture:

1. derive phase-shift seeds from circular alignment of phase-specific symbol histograms;
2. alternate a validated monoalphabetic wheel solve with conditional shift optimisation;
3. rank all 22 structures using short coordinate runs;
4. fully refine only the strongest candidates;
5. retain the original corpus, ciphertexts, objective, complexity penalty and development gates.

No further development amendment is permitted after this coordinate run.