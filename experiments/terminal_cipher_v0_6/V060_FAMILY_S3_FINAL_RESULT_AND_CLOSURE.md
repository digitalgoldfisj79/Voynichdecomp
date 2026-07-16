# v0.6 Family S3 — final neural ensemble result and Family S closure

Date: 2026-07-16

Verdict: **FINAL DEVELOPMENT GATE FAILED. FAMILY S IS CLOSED.**

No locked test, Voynich text or Davis hand labels were opened.

## Registered final amendment

The sole final S3 amendment used:

- two independently trained Transformer encoder-decoder models;
- seeds `1731` and `1732`;
- 30,000 updates per model;
- effective batch 32 per model;
- 960,000 fresh synthetic train examples per model;
- arithmetic ensemble of plaintext and boundary posteriors;
- beam width four;
- eight exact boundary-posterior segmentations under the frozen 1/2/3-symbol code-length prior;
- unchanged S2 mapping at `700,000 × 50` per legal lattice path and `700,000 × 200` final refinement;
- train-only calibrated selection between direct and lattice hypotheses.

Both model checkpoints were reconstructed and verified byte-for-byte before evaluation.

| Seed | Final checkpoint SHA-256 |
|---|---|
| 1731 | `bf31c7ad18c65170d4525f834bb3b32d1a283dc1a693c4356f8c17eb4db6d206` |
| 1732 | `9e77411dd192726132380d7481db0661d3030b83a497c7abba01380899a8ff97` |

## Execution chain

| Phase | Hugging Face job | Scientific artefact |
|---|---|---|
| Ensemble inference, train-only calibration and top-eight lattices | `Digitalgoldfish79/6a592a5e85d9643ce16d6a29` | `phase1.json`, SHA-256 `dd3d2382b1eada742d51dc01f37dd33fea0fc3a6b092eba5fe560dc90b441e08` |
| Bounded lattice mapping | `Digitalgoldfish79/6a592dbd85d9643ce16d6a4c` | `phase2.json`, SHA-256 `525263039243353f6c2b179a78f6de30cfb6bd69b35ecabb74749c4d7ec4e63a` |
| Final truth scoring | `Digitalgoldfish79/6a592ebc85d9643ce16d6a57` | `final.json`, SHA-256 `6290934da2bd5020897a7db61f8bb4b3f80b1eaea0d491d4ec3c5a529b76c44f` |

The full artefacts remain in the private Supabase bucket under `v060/s3/evaluation/dev/`.

## Structural lattice result

Every one of the 16 development trials produced eight complete boundary-posterior paths. However, every path in every trial contained more distinct visible code groups than the frozen 63-unit S2 inventory. Consequently:

- legal lattice candidates: **0/128**;
- trials with a legal lattice arm: **0/16**;
- lattice abstentions: **16/16**;
- no vocabulary was truncated, merged or expanded;
- no annealing or final refinement was run on an illegal path;
- the direct neural beam was the only legal hypothesis in every trial.

This is not merely a mapping-search failure. The boundary model generated an over-fragmented code vocabulary that was structurally incompatible with the registered plaintext-unit inventory.

## Final development results

### Plaintext recovery

| Statistic | Result | Required |
|---|---:|---:|
| Mean | **22.3796%** | ≥75% |
| Median | **22.2656%** | ≥85% |
| Minimum | **20.3125%** | ≥40% |
| Trials ≥75% | **0/16** | ≥13/16 |
| Exact plaintexts | **0/16** | — |

### Boundary recovery

| Statistic | Result | Required |
|---|---:|---:|
| Mean F1 | **48.4133%** | ≥85% |
| Median F1 | **48.4231%** | — |
| Minimum F1 | **27.7680%** | — |

All five registered conditions failed where applicable. The failure is not marginal: plaintext recovery remained tightly concentrated near 20–25%, and the best individual boundary F1 was only 71.17%.

## Interpretation

The models trained stably and achieved low synthetic training loss, but that did not transfer to the frozen development construction. The failure has two linked components:

1. **Plaintext generalisation failure.** The ensemble produced outputs that the train-only likelihood calibration regarded as plausible, but they were not aligned with the true fresh codebooks. Surface fluency or model confidence therefore did not imply decipherment.
2. **Segmentation failure.** Boundary predictions were insufficiently accurate and systematically generated too many distinct code groups for the bounded S2 inventory, eliminating the entire lattice-refinement arm.

The second independent model did not rescue the first: the ensemble remained close to chance-like character recovery for this alphabet size and far below every gate.

## Protocol consequence

The final S3 development gate failed after the sole permitted amendment. Therefore:

- Family S receives no locked-test run;
- no Family S solver may be applied to Voynich;
- no further S3 architecture, inventory, segmentation or calibration amendment is permitted under v0.6;
- Family S — syllabic, polygraphic and unsegmented substitution — is closed with a negative recoverability result.

Machine summary: `v060_family_s3_final_result.json`.
