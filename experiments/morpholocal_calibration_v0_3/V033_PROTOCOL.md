# Morpholocal calibration v0.3.3 latent-order protocol

The v0.3.2 score audit found no decision threshold that generalised to an
unseen permuted-cipher control. v0.3.3 therefore tests latent sequence order
directly, conditional on the complete model fitted to training data.

## Statistic

For each held-out line, map surface cells to fitted latent units. Compute the
transition-only negative log2 likelihood under the external transition matrix
selected using training data.

## Conditional null

For each randomization:

1. retain every held-out line and its length;
2. retain the first latent unit of each line;
3. randomly permute the remaining latent units within that line;
4. therefore preserve each line's exact latent-unit multiset and its stationary
   line-start contribution;
5. recompute transition codelength without refitting any model component.

The empirical one-sided p-value is

`(1 + number(null_bits <= observed_bits)) / (B + 1)`.

The test passes at alpha 0.05 only when p <= 0.05 and the mean randomized
transition codelength exceeds the observation.

## Development sequence

1. smoke: beam, positives 0:16, controls 0:16, B=199;
2. if technically and scientifically coherent, beam-only full development
   suite: 96 positives and 64 controls;
3. assess sensitivity by positive policy and false positives by control family,
   with permuted-cipher controls primary;
4. do not use Voynich data;
5. do not run heuristic, parallel tempering or neural models unless the
   beam-only conditional test demonstrates control-family generalisation.

This remains a development diagnostic, not a locked formal test.
