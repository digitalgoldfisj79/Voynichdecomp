# Source-transfer v0.4.0 oracle identifiability result

Date: 2026-07-15

Verdict: **STOP BEFORE INFERRED MAPPING**

No Voynich text was scored.

## Frozen provenance

- Base result commit: `0962441a373ca4aa045c19ef56fd010dd926cca9`
- Branch: `experiment/source-transfer-v0.4-oracle-audit-20260715`
- Protocol: `experiments/source_transfer_v0_4/ORACLE_IDENTIFIABILITY_PROTOCOL_V040.md`
- Configuration: `experiments/source_transfer_v0_4/oracle_source_transfer_v040_config.json`
- Hash-verified implementation launcher: `experiments/source_transfer_v0_4/oracle_source_transfer_v040.py`
- Full scientific job: `Digitalgoldfish79/6a57ea62b1669a49bf075eb6`
- Deterministic repeat job: `Digitalgoldfish79/6a57eafb85d9643ce16d558a`

The two completed jobs reported the same scientific summaries and gate decisions.

Reported result JSON hashes were:

- first full run: `98bee197c53001dfe90930e419254c5518a79a1066d8a3f8a0351a4218a8e538`;
- repeat run: `5188b46ea8254d6c4c369f044fc9e1b65a2555ba56757404db5cb347b295c897`.

These artifact hashes are not expected to match because the primary JSON includes runtime metadata such as elapsed seconds. No bitwise-reproducibility claim is made. A future formal runner must separate the canonical scientific payload from runtime metadata before hashing.

## Design

The audit removed latent-mapping inference and tested whether six oracle source representations could distinguish held-out real source chunks from ten structured or source-conditioned non-message control families.

Representations:

1. current 12-class word projection;
2. initial orthographic unit;
3. full normalized character stream with word boundaries;
4. deterministic syllable-like units;
5. pooled-training-only BPE;
6. capped word identity.

For each representation, smoothed n-gram source models of orders 1–5 were fitted on training partitions only. Model order and a source-transfer threshold were selected on development data subject to a development false-positive rate no greater than 10%. The frozen choice was then evaluated once on 48 positive test chunks and 480 controls.

## Primary results

| Representation | Selected order | Source rank-1 | Sensitivity | Control FPR | Maximum family FPR | Eligible for inferred mapping |
|---|---:|---:|---:|---:|---:|---:|
| `word12` | 2 | 95.83% | 2.08% | 10.83% | 47.92% | No |
| `initial` | 1 | 97.92% | 0.00% | 8.54% | 27.08% | No |
| `char` | 5 | 97.92% | 0.00% | 8.13% | 31.25% | No |
| `syllable` | 4 | 95.83% | 0.00% | 7.08% | 27.08% | No |
| `bpe` | 5 | 93.75% | 27.08% | 5.83% | 18.75% | No |
| `word` | 4 | 47.92% | 2.08% | 6.88% | 25.00% | No |

Global gate:

```json
{
  "any_message_specific": false,
  "any_source_transfer": true,
  "decision": "STOP_BEFORE_INFERRED_MAPPING",
  "eligible_representations": []
}
```

## Interpretation

Source-family attribution was often very high, reaching 94–98% for five representations. This did not translate into message specificity.

Once a threshold was frozen to constrain control false positives, genuine held-out source chunks were largely rejected. The best representation, BPE, detected only 13 of 48 positive chunks. The original 12-class projection detected 1 of 48 while accepting 52 of 480 controls, including 23 of 48 ordered-HMM controls.

Therefore source-family transfer, as implemented here, is not a valid arbiter of message-bearing cipher versus structured generation. It primarily identifies distributional compatibility with a source corpus. Structured and source-conditioned generators can possess the same compatibility without encoding an independently selected source message.

The v0.4.0 result rules out:

- inferred-mapping escalation under this comparator;
- crossing this comparator with more cipher renderers or keys;
- neural-decoder or larger GPU optimisation intended to rescue the same target;
- application to the Voynich Manuscript.

## Provenance limitation

Both jobs completed the scientific computation, but attempts to upload the complete JSON and CSV artifacts failed with Hugging Face permission errors:

1. the token could not create a new dataset repository;
2. the token could not obtain an Xet write token for an existing private model repository.

The complete summaries and family-level results remain in immutable Hugging Face job logs, but the full row-level result artifact was not durably uploaded. This is a provenance defect and prevents treatment of this developmental run as a formal locked validation. It does not alter the frozen stop decision because every representation failed by a large margin.

## Required methodological pivot

The next programme must not attempt to infer generic messagehood from source-family likelihood. The defensible target is one of:

1. exact or approximate recovery of independently specified plaintext information in synthetic calibration, measured against known ground truth;
2. bounded Bayesian or MDL comparison between explicit cipher and generator families, with an abstain/non-identifiable outcome;
3. external-anchor tests using candidate plaintexts, cribs, parallel texts, illustrations, or independently predicted lexical/semantic correspondences.

A universal binary cipher-versus-generator arbiter remains non-identifiable without restrictions or external anchors.
