# v0.6 Family P — oracle-stage result

Date: 2026-07-16

Verdict: **BOTH ORACLE COMPONENTS PASS; FULLY BLIND DEVELOPMENT IS AUTHORISED.**

No Voynich text was scored.

## Family

Family P models a fresh mixed-alphabet rotating wheel with either:

- continuous periodic stepping; or
- line-reset periodic stepping.

The development periods are drawn from 2, 3, 4, 6 and 8. Candidate periods 2–12 are compared under a fixed BIC-like complexity penalty. Plaintext length is 384 characters.

## Job

Hugging Face job: `Digitalgoldfish79/6a588346b1669a49bf076f6d`

Scientific SHA-256: `81316cce3590db9f7329d4b7e18c7c31231a3e5b0228cfc7c3e578d6aa57b9a0`

Configuration:

- English development split;
- 8 periodic and 8 line-reset ciphertexts;
- true-wheel schedule search: `50,000 × 12`;
- true-schedule mixed-alphabet search: validated v0.5.1 mono solver at `700,000 × 50`;
- 16 independent trials.

## Results

### True mixed wheel; unknown period and shifts

- mean recovery: **100%**;
- median recovery: **100%**;
- minimum recovery: **100%**;
- period recovery: **16/16**.

### True schedule; unknown mixed wheel

- mean recovery: **99.5117%**;
- median recovery: **99.6094%**;
- minimum recovery: **98.1771%**;
- all 16 trials exceeded 98%;
- exact plaintext recovery: 4/16.

The non-exact cases contain only small residual key errors and are far above the component reliability threshold.

## Decision

Both component channels are strongly identifiable at the registered observation length. Family P therefore proceeds to fully blind development, jointly estimating:

- operating mode;
- period;
- wheel alphabet;
- state shifts;
- plaintext.

The test split remains sealed. Failure of fully blind development after the single permitted development amendment closes Family P without a locked test.