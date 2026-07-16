# v0.6 Family S2 — initial segmented-oracle result and amendment

Date: 2026-07-16

Verdict: **INITIAL SEARCH FAILS; FINITE RESTART-RELIABILITY AMENDMENT AUTHORISED.**

No test data or Voynich text was scored.

## Job

Hugging Face job: `Digitalgoldfish79/6a588eac85d9643ce16d5f4c`

Scientific SHA-256: `caad577a86baa793455b8d96d787e860b8f5bb88981c37cad975df64f5ec1ade`

## Configuration

- English development split;
- 16 segmented polygraphic ciphertexts;
- 63 candidate plaintext units;
- fresh one-to-one code-group mapping per trial;
- true group boundaries supplied;
- train-only polygraphic-unit trigram-plus-unigram model;
- validated simulated annealing at `700,000 × 50`.

## Results

- mean plaintext recovery: **73.7206%**;
- median: **97.1502%**;
- minimum: **14.6392%**;
- trials at least 80%: **11/16**;
- trials at least 90%: **11/16**;
- exact plaintexts: 2/16;
- mean observed mapping accuracy: **65.6879%**;
- median observed mapping accuracy: **89.8737%**.

The 11 successful trials recover 94.8–100% plaintext. The five failures recover only 14.6–29.7%. There is no intermediate degradation regime.

## Diagnosis

The unit-language objective is adequate when the correct basin is reached. The failure is restart reliability in a 63-unit assignment space, not a systematic incompatibility of the polygraphic model.

## Frozen amendment

The only changed variable is independent restart count:

- proposals per restart remain 700,000;
- restart count increases from 50 to 200;
- corpus, ciphertexts, unit inventory, initial frequency key, proposal distribution, cooling schedule, objective and gates are unchanged.

The 200-restart run is the final S2 development attempt. If it fails, Family S closes without joint segmentation-decoding. If it passes, the smallest empirically supported restart budget is frozen for S3 and any later locked test.