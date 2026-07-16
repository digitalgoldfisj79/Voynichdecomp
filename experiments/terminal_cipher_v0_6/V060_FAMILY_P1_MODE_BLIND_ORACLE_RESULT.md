# v0.6 Family P1 — corrected mode-blind oracle result

Date: 2026-07-16

Verdict: **PASS. FULLY BLIND JOINT DEVELOPMENT IS AUTHORISED.**

No test data or Voynich text was scored.

## Correction applied

Every ciphertext carries independently generated observed line boundaries. The oracle structure search compares both operating modes (`periodic` and `line_reset`) and all periods 2–12. The earlier mode-supplied oracle report is superseded by this result.

## Job

Hugging Face job: `Digitalgoldfish79/6a5889f3b1669a49bf076fdf`

Scientific SHA-256: `1b0b3c35e0458bb40d11f52475f342e0255dba1b9be2f957da1573e5b00071f9`

Configuration:

- English development split;
- length 384;
- 8 continuous-periodic and 8 line-reset ciphertexts;
- true schedule, unknown mixed wheel: validated mono search at `700,000 × 50`;
- true mixed wheel, unknown mode, period and shifts: `50,000 × 12` per each of 22 structural candidates.

## Results

### True mixed wheel; unknown mode, period and shifts

- mean plaintext recovery: **100%**;
- median: **100%**;
- minimum: **100%**;
- operating-mode recovery: **16/16**;
- period recovery: **16/16**;
- exact mode-plus-period recovery: **16/16**.

### True schedule; unknown mixed wheel

- mean plaintext recovery: **99.5117%**;
- median: **99.6094%**;
- minimum: **98.1771%**;
- all 16 trials exceeded 98%;
- exact plaintext recovery: 4/16.

## Decision

The corrected oracle gate passes every registered condition. The fully blind solver may now jointly estimate:

- operating mode;
- period;
- wheel alphabet;
- state shifts;
- plaintext.

The first blind schedule is frozen at `250,000 × 24` for every mode-period candidate. The test split remains sealed.