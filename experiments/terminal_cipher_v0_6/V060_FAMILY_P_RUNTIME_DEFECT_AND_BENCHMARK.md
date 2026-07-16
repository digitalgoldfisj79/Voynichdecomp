# v0.6 Family P — runtime defect audit and corrected benchmark

Date: 2026-07-16

## Cancelled benchmark

Job: `Digitalgoldfish79/6a58d06885d9643ce16d638d`

The job emitted only:

- dependency installation;
- repository clone;
- Git head `5a80c33ad23e331195157a4b5eb710375be300c9d`.

It emitted no seed preflight, annealing progress, trial row or scientific SHA before cancellation.

## Root cause

`phase_histogram_seeds` requested eight unique shift seeds for every candidate period. At period 2 there is one free shift slot restricted to five ranked choices, so at most five distinct vectors exist. The unbounded uniqueness loop could therefore never reach eight. The solver stopped before its first annealing proposal.

The cancelled runtime is not scientific compute and does not constitute a Family P result.

## Corrected execution

Execution clarification commit: `b4f7f556766e29e8aeadf85b3f596e7f81bfac7b`

Termination-safe runner commit: `6dcf2f031a1c1fc701413869c3265612c6f17762`

The frozen eight-start budget is preserved. Distinct seeds are used until the finite proposal space is exhausted; remaining starts are sampled with replacement and receive independent downstream annealing seeds through their original `seed_index`.

## Corrected benchmark result

Job: `Digitalgoldfish79/6a5932beb1669a49bf07830e`

Trial: English development, true mode `periodic`, replicate 0, true period 6.

- elapsed: `472.36716659700323` seconds;
- wrapper elapsed: `472.3682440689954` seconds;
- plaintext accuracy: `0.9947916666666666`;
- exact: `false`;
- selected mode: `periodic`;
- selected period: `6`;
- mode correct: `true`;
- period correct: `true`;
- structure correct: `true`;
- screening-best accuracy: `0.9947916666666666`;
- refined candidates: `4`;
- result SHA-256: `0993b981e252dfbc8e843c38138bbc1c129cae034579b430edada7d4281e6a3a`.

## Interpretation

The first valid shard is strongly positive. It exceeds the Family P per-trial recovery target and correctly identifies the hidden schedule structure. It does not by itself pass the 16-trial development gate; the full mode-crossed development set remains required.
