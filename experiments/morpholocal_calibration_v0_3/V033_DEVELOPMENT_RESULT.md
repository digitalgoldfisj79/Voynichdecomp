# Morpholocal calibration v0.3.3 development result

Date: 2026-07-15

## Status

**PROMISING DEVELOPMENT RESULT — INDEPENDENT VALIDATION REQUIRED.**

Do not apply this result to the Voynich Manuscript. The same synthetic fixture
families used during v0.3.2 development were reused here, and the positive
fixtures were generated from the same external-model family used by the
scorer. The result demonstrates that the new statistic attacks the identified
permuted-cipher failure; it is not yet a generator-disjoint validation.

## Frozen implementation

Branch:
`experiment/morpholocal-calibration-v0.3.3-sequence-randomization-20260715`

Scientific code commit:
`2b48082191e5de767242ab25aaa92fb99cc02da6`

Protocol:
`experiments/morpholocal_calibration_v0_3/V033_PROTOCOL.md`

The conditional null preserves the fitted model, line membership, line length,
line-start unit and each line's exact latent-unit multiset. It permutes only the
remaining latent order within each held-out line. Each trial used 199
deterministic randomizations. The one-sided development threshold was alpha
0.05.

## Full beam-only development suite

- positives: 96;
- controls: 64;
- solver: beam only;
- no heuristic, parallel-tempering or neural rerun;
- positive passes: **73/96 = 76.04%**;
- control false positives: **6/64 = 9.38%**;
- specificity: **58/64 = 90.63%**;
- permuted-cipher false positives: **0/16**.

Approximate 95% Wilson intervals:

- sensitivity: 66.61%--83.47%;
- false-positive rate: 4.37%--18.98%;
- permuted-cipher false-positive rate: 0%--19.36%.

At a nominal 5% randomization level, observing at least six false positives in
64 controls has binomial tail probability approximately 0.100. The observed
9.38% false-positive rate is therefore not yet demonstrably above nominal, but
64 controls are insufficient to establish calibration tightly.

## Positive sensitivity by selection policy

| Positive policy | Passes | Trials | Rate |
|---|---:|---:|---:|
| iid_uniform | 20 | 24 | 83.33% |
| frequency_weighted | 18 | 24 | 75.00% |
| cyclic | 17 | 24 | 70.83% |
| sticky_line_reset | 18 | 24 | 75.00% |

The previous v0.3.2 composite gate detected 0/72 sticky-line-reset positives
across three solvers. The conditional latent-order statistic therefore repairs
the principal sensitivity failure at the level it was designed to test.

## Control false positives by family

| Control family | False positives | Trials | Rate |
|---|---:|---:|---:|
| context_iid | 3 | 16 | 18.75% |
| cell_markov | 2 | 16 | 12.50% |
| copy_mutate | 1 | 16 | 6.25% |
| permuted_cipher | 0 | 16 | 0.00% |

The primary adversarial family, `permuted_cipher`, is rejected in every trial.
This is the opposite of the v0.3.2 learned-composite failure, which accepted
nearly all permuted controls when that family was withheld from training.

## Comparison with the v0.3.2 beam decision

| Decision | Positive detections | Control false positives |
|---|---:|---:|
| v0.3.2 composite gate | 25/96 | 33/64 |
| v0.3.3 conditional latent order | 73/96 | 6/64 |

These are development comparisons on the same fixtures and must not be read as
locked performance estimates.

## Shard provenance

| Range | Job | Result SHA-256 | Temporary artifact |
|---|---|---|---|
| positives 0:16, controls 0:16 | `6a57d46bb1669a49bf075bdf` | `f89ccbad706988f0c3f4929948308cdec06195720c3f5859076917b14aa63fe2` | `https://n.uguu.se/dRTkCvSu.gz` |
| positives 16:32, controls 16:32 | `6a57d58085d9643ce16d53e5` | `1aef6338b71b7104e6f9717786efe3e81c0e3a2b77052d04066a0cd6b45ce8f9` | `https://h.uguu.se/gweaBjWm.gz` |
| positives 32:48, controls 32:48 | `6a57d5aa85d9643ce16d53e7` | `9632658ac9ea9de2729c759ed8112ad7bf243c5aecf18c9966b930a3d51143e2` | job log |
| positives 48:64, controls 48:64 | `6a57d5b385d9643ce16d53f1` | `94143619ff588c4790f98e57983bb3e20bd5ab830cfac7a76acbf20af1e97561` | job log |
| positives 64:80 | `6a57d5bf85d9643ce16d53f5` | `851444d325d2285b7447f42a81a5d6131a5d8fbe70fa608dea6bdc2050c17efb` | job log |
| positives 80:96 | `6a57d66085d9643ce16d542b` | `e51a6e04236c519037348637c4170648c1472c7bb57c674a5904365eb456dd22` | job log |

The final four development JSON files were hashed and summarized in immutable
Hugging Face job logs but were not uploaded to the temporary artifact host.
A formal validation must write all trial records to persistent authenticated
storage before execution.

## Interpretation and limit

The result supports the narrower claim that the recovered held-out latent
sequence often has non-random order under the selected external transition
model, while the permuted-cipher controls do not.

It does **not** by itself prove that a latent sequence is an enciphered human
message. A generative latent-state process can also create ordered hidden
states. Therefore the next validation must be generator-disjoint and include:

1. positives produced by independent cipher implementations and external
   corpora not used to fit the scorer;
2. hard controls with ordered latent states but no enciphered source message;
3. block-permuted and locally Markov-preserving cipher controls;
4. document-grouped locked evaluation and persistent artifacts;
5. beam only unless an explicit solver comparison is scientifically required.
