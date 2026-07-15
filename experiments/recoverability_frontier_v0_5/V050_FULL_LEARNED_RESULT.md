# Recoverability frontier v0.5.0 — full learned-decoder result

Date: 2026-07-15

Verdict: **STOP THE MONOLITHIC LEARNED-DECODER ARCHITECTURE**

No Voynich text was scored.

## Execution provenance

- Branch: `experiment/recoverability-frontier-v0.5.0-20260715`
- Frozen protocol: `RECOVERABILITY_PROTOCOL_V050.md`
- Pre-run amendment: `PROTOCOL_AMENDMENT_V050_A.md`
- Full family-known job: `Digitalgoldfish79/6a57f1a885d9643ce16d5648`
- Full blind-family job: `Digitalgoldfish79/6a57f1b085d9643ce16d564a`
- Hardware: one `a100-large` per arm, running in parallel
- Training examples per arm and epoch: 120,000 positives plus 120,000 controls
- Epochs: 4
- Test set per arm: 8,640 positives plus 8,640 controls
- Ciphertext input: first-occurrence recurrence canonicalisation

Both jobs completed successfully.

## Family-known result

Development threshold was frozen at `0.5730006098747253`, producing development FPR 5.00% and sensitivity 8.125%.

Test result:

- sensitivity: **9.0278%**;
- control false-positive rate: **5.1620%**;
- mean character accuracy over all positive trials: **2.1612%**;
- mean character accuracy among positives declared message-bearing: **23.9394%**;
- exact recovery: **0/8,640**.

| Family | Sensitivity | Mean accuracy over all positives | Exact recovery |
|---|---:|---:|---:|
| `mono` | 15.93% | 3.85% | 0% |
| `homophonic` | 0.56% | 0.13% | 0% |
| `null_homophonic` | 0.93% | 0.24% | 0% |
| `polyalphabetic` | 0.00% | 0.00% | 0% |
| `feedback` | 0.09% | 0.02% | 0% |
| `nomenclator` | 42.22% | 9.98% | 0% |
| `transposition` | 10.74% | 2.63% | 0% |
| `fractionated` | 1.76% | 0.43% | 0% |

Frozen gate:

```json
{
  "five_families_accuracy_pass": false,
  "fpr_pass": false,
  "pass": false,
  "sensitivity_pass": false
}
```

## Blind-family result

Development threshold was frozen at `0.546865701675415`, producing development FPR 4.9884% and sensitivity 5.2894%.

Test result:

- sensitivity: **7.0139%**;
- control false-positive rate: **4.8611%**;
- mean character accuracy over all positive trials: **1.6731%**;
- mean character accuracy among positives declared message-bearing: **23.8539%**;
- exact recovery: **0/8,640**.

| Family | Sensitivity | Mean accuracy over all positives | Exact recovery |
|---|---:|---:|---:|
| `mono` | 8.43% | 1.97% | 0% |
| `homophonic` | 5.28% | 1.27% | 0% |
| `null_homophonic` | 5.28% | 1.28% | 0% |
| `polyalphabetic` | 0.56% | 0.13% | 0% |
| `feedback` | 1.48% | 0.35% | 0% |
| `nomenclator` | 21.85% | 5.27% | 0% |
| `transposition` | 7.59% | 1.79% | 0% |
| `fractionated` | 5.65% | 1.33% | 0% |

Frozen gate:

```json
{
  "five_families_accuracy_pass": false,
  "fpr_pass": true,
  "pass": false,
  "sensitivity_pass": false
}
```

## Interpretation

The monolithic Transformer did not learn general fresh-key cryptanalysis. Family information improved results only modestly. Sequence loss declined throughout training, but held-out plaintext recovery remained extremely poor and no sequence was recovered exactly.

The separate `MESSAGE` / `NO_MESSAGE` classification target is also conceptually invalid for these controls. A Markov, motif, copy/mutate or slot sequence passed through a cipher is still an encoded latent sequence. It differs in provenance, not in the existence of decodable content. Consequently, near-chance classification is expected and must not be used as the gate for plaintext recovery.

This result rules out:

- further scaling of the same shared sequence-to-sequence architecture;
- additional epochs or larger GPUs as the primary remedy;
- broad language expansion under the same model;
- treating `MESSAGE` classification as evidence of generated versus cipher;
- application to the Voynich Manuscript.

## Required pivot

The next programme must separate two tasks:

1. **Cryptanalytic recoverability:** family-specific solvers recover the latent sequence under bounded cipher assumptions. Every encoded sequence, including generator-produced controls, has a recovery target.
2. **Bounded provenance comparison:** after recovery, explicit source-language and generator models are compared under common MDL/Bayesian accounting, with `NON_IDENTIFIABLE` permitted.

The immediate v0.5.1 diagnostic should implement a solver portfolio rather than another monolithic decoder:

- monoalphabetic substitution: n-gram scoring plus simulated annealing / tree search;
- homophonic and null-homophonic substitution: null-aware hill climbing or EM/beam search;
- polyalphabetic: period search and shift optimisation;
- feedback: bounded state/key search;
- nomenclator: joint word-code and character substitution search;
- transposition: block/permutation search combined with substitution scoring;
- fractionated: coordinate-system and symbol-mapping search.

Family-known solvers are evaluated first. A later blind-family layer selects among solver evidences using held-out MDL rather than a direct neural class label.

## Stop/go rule

Do not resume a blind-family or multilingual scale-up until at least five family-specific solvers exceed 70% mean character recovery on unseen keys and the family selector is calibrated on independently implemented test generators.
