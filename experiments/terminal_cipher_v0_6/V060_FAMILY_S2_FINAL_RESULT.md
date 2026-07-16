# v0.6 Family S2 — final segmented-oracle result

Date: 2026-07-16

Verdict: **PASS. FULLY JOINT S3 IS AUTHORISED.**

No test data or Voynich text was scored.

## Job

Hugging Face job: `Digitalgoldfish79/6a588f4185d9643ce16d5f4e`

Scientific SHA-256: `aae7812cf819e2014cfaf76b6e5ca072a5166d3ca022185cdc7c3141f9bf5486`

## Frozen amendment

- train-only polygraphic-unit trigram-plus-unigram objective unchanged;
- 700,000 proposals per restart unchanged;
- independent restarts increased from 50 to 200;
- corpus, ciphertexts, unit inventory, proposal distribution, cooling schedule and gates unchanged.

## Results

- mean plaintext recovery: **93.9004%**;
- median: **98.3073%**;
- minimum: **29.6875%**;
- trials at least 80%: **15/16**;
- trials at least 90%: **15/16**;
- exact plaintexts: 2/16;
- mean observed mapping accuracy: **88.6169%**;
- median observed mapping accuracy: **94.4135%**;
- mean frequency-baseline recovery: 26.1638%.

Fifteen trials recover 94.8–100% plaintext. One trial remains in a wrong basin at 29.69% recovery and 9.76% mapping accuracy.

## Decision

The registered S2 gate passes:

- mean recovery exceeds 80%;
- median exceeds 90%;
- 15/16 trials exceed 80%;
- mean observed mapping accuracy exceeds 75%.

The fully joint S3 stage may now attempt to infer both variable-length code-group boundaries and the fresh code-group-to-polygraphic-unit mapping. The test split remains sealed.