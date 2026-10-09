# LANG2 CI 24-pair confirmation: execution manifest v2

2026-10-09. This is an **execution record**, not a change to frozen confirmatory statistical preregistration.

## Infrastructure correction
All sixteen first submissions in the initial 24-pair panel failed before model training. The frozen four-seed wrapper limited argparse `--seed` and `--scramble-seed` to four pilot seeds; all 24 new seed choices were rejected. This was an infrastructure/CLI error, not negative scientific evidence. An initial smoke retry erroneously included an extra file-length check and failed before training; the corrected smoke completed, job [6ac94573fee2c9007017cad2](https://huggingface.co/jobs/Digitalgoldfish79/6ac94573fee2c9007017cad2).

Confirmed new-seed smoke: **PASS**; stdout `VGER2_SMOKE.pass=true`, `scores_shape=[735,4103]`, same frozen physical-fold SHA `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`, same solver/runtime/source vocab. Reproducible runtime-patched wrapper SHA256 **`0421631aad33221a484388caeab4efa0080f30fe73d1a2d4ce47c562b0f53cbf`**. This patch modifies precisely the two seed-choice lists (adding exactly the SHA-frozen 24 pair manifest entries) and the per-mode/seed cache file name to prevent packed replica write collisions; the original NeuroCipher scientific code, 10x150 epochs, fold structure, source vocabulary, model, E step and scoring do not change. Patch audit: [LANG2_CI24_CLI_EXTENSION_AUDIT_20261009.md](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/4345b6f81aa9be43f450a60a24cffcbb5a0f6a0c/research/LANG2_CI24_CLI_EXTENSION_AUDIT_20261009.md).

## Successful submissions (sixteen replacement batch jobs)

Each contains three independent model replicas and directly streams their per-seed `VGER2_RESULT_JSON` records. 2-hour max timeout, A10G-large $1.50/hour (billing by run minute). Batch ordering is unrelated to inference and **not sequential testing**.

| Pairs | Real job | Scrambled job |
|---|---|---|
| 1-3 | [6ac945d7fee2c9007017cb42](https://huggingface.co/jobs/Digitalgoldfish79/6ac945d7fee2c9007017cb42) | [6ac945db095c5780893098b8](https://huggingface.co/jobs/Digitalgoldfish79/6ac945db095c5780893098b8) |
| 4-6 | [6ac945df095c5780893098ba](https://huggingface.co/jobs/Digitalgoldfish79/6ac945df095c5780893098ba) | [6ac945e4fee2c9007017cb4b](https://huggingface.co/jobs/Digitalgoldfish79/6ac945e4fee2c9007017cb4b) |
| 7-9 | [6ac945f6fee2c9007017cb63](https://huggingface.co/jobs/Digitalgoldfish79/6ac945f6fee2c9007017cb63) | [6ac945f8fee2c9007017cb65](https://huggingface.co/jobs/Digitalgoldfish79/6ac945f8fee2c9007017cb65) |
| 10-12 | [6ac945fafee2c9007017cb67](https://huggingface.co/jobs/Digitalgoldfish79/6ac945fafee2c9007017cb67) | [6ac945fd095c5780893098cf](https://huggingface.co/jobs/Digitalgoldfish79/6ac945fd095c5780893098cf) |
| 13-15 | [6ac945fffee2c9007017cb6b](https://huggingface.co/jobs/Digitalgoldfish79/6ac945fffee2c9007017cb6b) | [6ac94602fee2c9007017cb6d](https://huggingface.co/jobs/Digitalgoldfish79/6ac94602fee2c9007017cb6d) |
| 16-18 | [6ac94614fee2c9007017cb77](https://huggingface.co/jobs/Digitalgoldfish79/6ac94614fee2c9007017cb77) | [6ac94616095c5780893098e6](https://huggingface.co/jobs/Digitalgoldfish79/6ac94616095c5780893098e6) |
| 19-21 | [6ac94619fee2c9007017cb7e](https://huggingface.co/jobs/Digitalgoldfish79/6ac94619fee2c9007017cb7e) | [6ac9461cfee2c9007017cb82](https://huggingface.co/jobs/Digitalgoldfish79/6ac9461cfee2c9007017cb82) |
| 22-24 | [6ac94621fee2c9007017cb86](https://huggingface.co/jobs/Digitalgoldfish79/6ac94621fee2c9007017cb86) | [6ac94624fee2c9007017cb89](https://huggingface.co/jobs/Digitalgoldfish79/6ac94624fee2c9007017cb89) |

Smoke and initial first GPU job inspected; pack1 REAL stdout contained `VGER2_TRAIN` at epochs 10 and 20 for **all three** seeds plus expected GER2 folds, demonstrating that patched CLI admitted new seeds and actual full training had started. The 48 terminal results and preregistered 24-pair adjudication have **not** yet completed. No inferential claims from interim loss values.

Frozen analysis protocol: https://github.com/digitalgoldfisj79/Voynichdecomp/blob/26e6379668900b89b68a2c393fb06b747f1ce654/research/LANG2_CI_24PAIR_CONFIRM_PREREG_20261009.md
Frozen seeds: https://github.com/digitalgoldfisj79/Voynichdecomp/blob/74b889b1d1973c998d8938ef96ed7a6743462eb0/research/data/LANG2_CI_24PAIR_SEEDS_20261009.json
Original 4-pair Latin GER2 result remains FAIL: https://github.com/digitalgoldfisj79/Voynichdecomp/blob/f766600e0233671e6c891368116b0817ba052697/research/LANG2_LATIN_CI_GER2_FINAL_20261009.md
