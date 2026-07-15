# Recoverability frontier v0.5.2 — pair-block development result

Date: 2026-07-15

Verdict: **PAIR-BLOCK MOVES SOLVE HEBREW BUT NOT ENGLISH**

## Flexible-inventory pair blocks

Development jobs:

- English: `Digitalgoldfish79/6a580d9d85d9643ce16d593f`, SHA `2220824725702a0959bdbb0346cf53d7191fbd25016dc59ee81cf82a3c5bfb9c`;
- Hebrew: `Digitalgoldfish79/6a580da785d9643ce16d5941`, SHA `5b586c0b0ae5324c4b517a414c6e4c7ef582eca0156fe430cb802f8edd988c6c`.

Best recovery:

- English: 28.2227%;
- Hebrew: 84.7656%.

Flexible block moves continued to damage English's initially strong inventory estimate.

## Fixed-inventory pair blocks

Development jobs:

- English: `Digitalgoldfish79/6a580e66b1669a49bf07641b`, SHA `ba465fce38bb05936f682994893dac4d61e9ce2f72dbe59ec9416b9fda15921b`;
- Hebrew: `Digitalgoldfish79/6a580e6fb1669a49bf07641d`, SHA `9d055dd2bd971759545346320248af443805fe2222677b9a7b895e2075847413`.

Best recovery:

- English: 47.8841% at 12 restarts × 8 sweeps;
- Hebrew: 99.5117% at 24 restarts × 12 sweeps.

The inferred inventory was preserved exactly.

## Interpretation

The Hebrew failure is resolved at the development level: its inventory estimate is sufficiently accurate, and coordinated pair-block assignments recover the plaintext almost completely.

English improves substantially but remains below the 70% development gate. Pair-block coordinate ascent and swap-only annealing solve different aspects of the search:

- block moves cross coordinated homophone-assignment barriers;
- annealing can refine within a promising basin and escape block-coordinate local optima.

The next permitted diagnostic is a fixed-inventory hybrid: block polish, swap-only annealing, then block polish again. The objective and inventory remain unchanged.
