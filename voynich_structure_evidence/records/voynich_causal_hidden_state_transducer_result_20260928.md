# Causal hidden-state transducer result — encoded vs created — 2026-09-28

- key: `voynich_causal_hidden_state_transducer_result_20260928`
- status: active
- created: 2026-09-28T11:50:59.65285+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

# Causal hidden-state transducer result — 2026-09-28

RETRACT / DOWNGRADE:
- Reject as encoded-natural-language rescue.
- Do NOT claim no hidden state: a small reproducible past-ciphertext state signal exists for exact SG64 realization.
- Do NOT interpret downstream bigram gain as decryption; matched random compression explains it.
- The hidden state behaves like part of the surface production mechanism, not removable cipher noise.

Frozen HMM:
rho=.70, p0=.35, p1=.70 selected on CALIB only.
State q_t is inferred recursively from past majority-vs-alternative SG64 ciphertext emissions before observing current token.

CALIB exact realization:
n=1155.
accuracy +0.6926 pp / folio null SD .3240 = +2.14 SD.
log-loss benefit +.04131 bits/token /.01921 = +2.15 SD.
21 families selected prospectively.

TEST Gate A, binary hidden-regime prediction:
n=1207.
memoryless 1.00465 bits; HMM 1.00184.
benefit +.0028115 /.0125689 = +.224 SD. THE METRIC DOES NOT RESOLVE THIS.
q mean .4919, SD .0745, range .3916-.6000.

TEST exact all ambiguous:
n=1126.
accuracy 47.9574% ->48.6679%.
gain +.7105 pp /.3323 = +2.14 SD.
log loss 2.02475 ->1.97865.
benefit +.04610 /.02306 = +1.9988 SD: under frozen 2-SD rule THE METRIC DOES NOT RESOLVE THIS.

TEST CALIB-selected 21:
n=487.
accuracy gain 0.
log-loss benefit +.02064 /.01124=+1.84 SD unresolved.

Post-hoc q-shuffle sensitivity:
500 within-folio shuffles preserve q distribution.
Correct folio-cluster actual-vs-shuffle:
accuracy excess +.0036978 /.0017595=+2.10 SD.
log-loss excess +.0027646 bits/token /.0012958=+2.13 SD.
All five TEST folios have positive exact-state alignment excess.
LICENSED POSITIVE: past ciphertext carries a small real prospective signal about which exact SG64 surface realization is used next.
Magnitude is tiny; most gross exact-family log-loss improvement is recalibration/regularisation, not causal sequence alignment.

Downstream selective strip (21 families):
TRAIN->TEST within-line bigram raw 10.59804, strip 10.32940.
gain +.26864 /.13592=+1.976 SD: under frozen rule unresolved.

500 matched random family collapses:
mean gain +.20467, SD .05234.
actual excess +1.22 random SD; p(random>=actual)=.102.
No transducer-specific lexical simplification.

TEST positional:
P1 raw -2.9584 pp /1.6234=-1.82 SD; strip -6.1957/1.9779=-3.13 SD.
paired change -3.2372 pp /.8375=-3.86 SD WRONG WAY.
P3 raw +.4585 pp; strip -1.5468; paired -2.0054/1.818=-1.10 SD unresolved; absolute magnitude grows.
E-2 raw +10.9324 pp; strip +14.5185; paired +3.5860/4.2984=.83 SD unresolved; magnitude grows.
0/3 shrink.

ADJUDICATION:
Preregistered hidden-state encoded-natural-language rescue FAILS.
But a weak persistent causal state is real at exact surface-realization level.
Best interpretation: hidden/local state is part of Voynich surface production grammar, not a removable wrapper around recovered plaintext.

Current text-level mechanism:
real lexical/family identities + native positional/frame grammar + strong local graphotactic compatibility + weak persistent causal realization state + no recovered canonicalisation that exposes ordinary lexical grammar.

This substantially weakens simple/modestly stateful natural-language encoding.
It does not exclude arbitrarily powerful hidden transduction, but that hypothesis now requires a mechanism that creates much of the Voynich grammar rather than merely hides plaintext.
A native meaning-bearing constrained notation/formal/generative system is the parsimonious working model; meaningless/asemic generation is NOT established.

