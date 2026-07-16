# v0.6 Stage 7 locked result

Date: 2026-07-16

Verdict: **LOCKED SYNTHETIC GATE FAILED. VOYNICH WAS NOT OPENED.**

## Provenance

- Hugging Face job: `Digitalgoldfish79/6a5940ecb1669a49bf078483`
- Git head: `f18e5a1ca016980d391545b22d3c39ce9d30ec5d`
- Running time: `76` seconds
- Scientific SHA-256: `63d1ffac893e2e0449a22c87d5ff92aee2ab2c6d02a28f93cad8980204091add`
- Implementation freeze: `V060_BLIND_MODEL_SELECTION_IMPLEMENTATION_FREEZE.md`

The process built train, calibration and locked-test feature sets. It loaded no Voynich transcription because the locked synthetic gate failed.

## Frozen calibration rule

The calibration split selected:

- P probability threshold: `0.51`;
- top-two probability margin: `0.02`;
- calibration P recall: `94.4444%`;
- calibration false-positive rate on generated, notation and none controls: `0%`.

## Locked synthetic results

| Metric | Result | Required | Outcome |
|---|---:|---:|---|
| Macro one-vs-rest AUC | **0.96505** | >=0.90 | Pass |
| Expected calibration error | **0.06071** | <=0.05 | Fail |
| P false-positive rate on structured controls | **0.00000** | <=0.05 | Pass |
| P precision | **0.96154** | >=0.90 | Pass |
| P recall | **0.34722** | >=0.80 | Fail |

The overall gate therefore failed.

## Locked confusion pattern

The class order was:

`P, S, T, generated, mixed, mono, none, notation, ordinary`.

For the 72 true P examples, the unthresholded top-class predictions were:

- P: `31`;
- S: `14`;
- mixed: `27`;
- every other class: `0`.

Only `34.7222%` of true P examples satisfied the frozen probability-plus-margin evidence rule. The actionable failure is therefore not broad false-positive inflation. It is failure to identify unseen P instances reliably and confidently.

The full confusion matrix by true row and predicted column was:

```text
31 14  0  0 27  0  0  0  0
 0 71  0  0  1  0  0  0  0
 0  0 56  2  0  0 14  0  0
 0  0  4 55  0  9  0  0  4
 1  2  0  0 69  0  0  0  0
 0  0  0 12  0 47  0  0 13
 0  0  6  0  0  0 66  0  0
 0  0  0  0  0  0  0 72  0
 0  0  0  9  1 52  0  0 10
```

## Interpretation

The high macro-AUC shows that the feature space contains substantial family information in aggregate. It does not establish an operational detector. At the preregistered evidence threshold, locked P sensitivity collapsed from `94.44%` on calibration to `34.72%` on unseen test examples.

The P test partition used unseen period regimes, while the calibration partition used the development regimes. The confusion of true P chiefly with S and mixed indicates that the label-invariant statistics did not transfer sufficiently across that structural shift. This is a likely explanation rather than a post-test repair claim.

The result also establishes a useful distinction:

1. the Family P solver can recover plaintext extremely well when the input is known to belong to the validated wheel family and carries an observed circular symbol order;
2. the frozen label-invariant evidence layer cannot reliably determine that an unseen sequence belongs to Family P;
3. EVA provides no independently observed circular glyph order and has incompatible representation cardinalities across transcriptions;
4. therefore applying the solver directly to Voynich would produce uninterpretable, output-selected plaintext rather than protocol-valid evidence.

## Protocol consequence

- no classifier, feature, threshold or calibration amendment is permitted after this locked result;
- Voynich transcription scoring remains blocked;
- no Family P plaintext may be generated or interpreted under v0.6;
- Families T, G and S remain closed negative;
- Family P remains a positive recoverability result but fails the required family-identification bridge to the manuscript;
- the terminal v0.6 programme is complete.

The programme-level conclusion is not that every cipher is impossible. It is that no represented mechanism completed the entire evidential chain from blinded synthetic recovery through calibrated family identification to protocol-valid Voynich application.
