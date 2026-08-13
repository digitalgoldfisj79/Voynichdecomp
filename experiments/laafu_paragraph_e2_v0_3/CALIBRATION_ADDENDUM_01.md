# Calibration addendum 01 — N3 instrument hardening

Frozen after the v0.3 run showed `C0_N3` missed its preregistered 20-replicate gate by one replicate: mean ratio 1.0383 was inside [0.95,1.05], but 3/20 pseudo-corpora had |z|>=2 versus the allowed <=2/20. No change is made to the v0.3 target result or original verdict.

This addendum tests **only null calibration**, not Voynich ordering.

## Extended calibration

- Corpus skeleton/token multisets: ZLZI P0 loci, as in v0.3.
- Generate 100 independent pseudo-corpora by one N1 boundary-class shuffle each.
- For each pseudo-corpus, evaluate N3 interior E2 against 500 fresh N1 permutations.
- Seeds begin at 20261813 and advance deterministically by 1009 per pseudo-corpus.

## Calibration criteria

Declare the N3 instrument adequately calibrated for scientific interpretation iff all hold:

1. mean E2 ratio in [0.97, 1.03];
2. mean z in [-0.25, +0.25];
3. SD of z in [0.80, 1.20];
4. no more than 8/100 have |z|>=2.

These limits are set without reference to the observed P0 N3 target value. Passing this addendum does **not** retroactively change the literal v0.3 preregistered control verdict; it determines whether the original 3/20 miss is consistent with a calibrated instrument or indicates systematic miscalibration.
