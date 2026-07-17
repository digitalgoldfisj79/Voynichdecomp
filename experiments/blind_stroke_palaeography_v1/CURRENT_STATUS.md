# Current status — blind stroke palaeography programme

**Canonical continuation entry point. Read this file first in every new chat.**  
Updated: 2026-07-17

## Do not repeat completed work

Do **not** re-audit Hugging Face job `6a5a1540d216bd6f3a1fb177`.

The complete v1.5.1 result has already been recovered from 873 log chunks, its bundle checksum verified, its result files inspected, and its scientific decision frozen.

Canonical audit:

- `results/V1_5_FULL_RESULT_AUDIT.md`

Prospectively frozen next protocol:

- `AMENDMENT_005_V1_6_CONTINUOUS_STYLE_SIMILARITY.md`

## Frozen scientific conclusion

The v1.5.1 full run is **Category B**:

> Writer retrieval is statistically significant and robust, but the acquisition-nuisance ratio and discrete-K recovery gates fail.

Therefore:

- no validated writer-identification claim;
- no independent hand recovery;
- no valid K or five-scribe claim;
- no opening of Voynich labels or expected boundaries;
- proceed only with prospective external validation of continuous handwriting-style similarity.

Decisive values:

- selected checkpoint: step 500;
- selected representation: `resid_combined`;
- validation mAP: `0.3841019402`;
- terminal-test mAP: `0.4107341042`;
- acquisition nuisance mAP: `0.3053187056`;
- absolute margin over acquisition: `0.1054153987` — pass;
- ratio over acquisition: `1.34554` — fail against frozen 1.5× gate;
- permutation: `p=0.005`, `n=199` — pass;
- all five v1.5 perturbation-retention gates — pass;
- exact-K recovery: `0.1333333333` — fail;
- K within ±1: `0.2222222222` — fail.

Verified bundle SHA-256:

`7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa`

## Voynich seal

Still sealed:

- Davis hand labels and five-hand assignments;
- f115r line boundary;
- Voynich sections and Currier labels for model selection;
- any forced K=5 analysis.

The full result records:

- `davis_labels_loaded=false`;
- `f115r_loaded=false`;
- `voynich_opened=false`.

## Immediate operational blocker

The recovery bundle contains:

- `result.json`;
- `writer_split.json`;
- `exact_features.npz`.

It does **not** contain the trained checkpoint bytes. The completed job reported:

- checkpoint: `saghog_v15_best.pt`;
- bytes: `115400460`;
- SHA-256: `9b543dbc13600a8a32ca04e7794541d8e184c0cbcd793e1b9492d37e03445d09`.

The transient container has exited, so new-corpus v1.6 inference requires either exact checkpoint recovery or one documented deterministic reproduction run that persists the checkpoint and all artifacts. The acceptance tolerances are frozen in Amendment 005.

## Next action

1. Persist the recovered v1.5 result bundle as a durable Hub artifact rather than log-only base64.
2. Implement a v1.5.1 checkpoint-persistence reproduction launcher with pinned dependencies, explicit random-state controls, explicit KMeans `n_init`, separate SHA manifest, and Hub upload.
3. Run the reproduction once.
4. Compare every representation, nuisance metric, selected checkpoint and gate decision with the frozen tolerances.
5. Only if reproduction is accepted, execute the v1.6 external continuous-similarity validation.

Do not return to result interpretation unless a newly discovered checksum or implementation fact directly contradicts the canonical audit.