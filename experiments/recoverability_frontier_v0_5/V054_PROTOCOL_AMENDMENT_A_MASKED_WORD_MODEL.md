# Recoverability frontier v0.5.4 — protocol amendment A: masked word model

Date: 2026-07-16

Status: fixed after completion of the train-only word n-gram frontier, before neural execution.

## Evidence

The interpolated train-only word n-gram does not recover nomenclator code identities reliably.

Across the completed frontier:

- 384 characters, pool 32: 44.24% mapping accuracy;
- 768 characters, pool 32: 44.40%;
- 1536 characters, pool 32: 55.61%;
- 1536 characters, pool 96: 35.06%.

Longer text and denser code support improve occurrence-level recovery, but exact observed-code mapping remains well below the frozen 80% gate. Expanded character recovery is not an adequate substitute because wrong code words occupy a small fraction of the plaintext.

## Added A1 model

A train-only masked word Transformer is added as a distinct codeword-inference architecture.

- vocabulary: top 6000 train words plus explicit unknown, mask and sentence-boundary tokens;
- input: decoded literal words and mask tokens at oracle-known nomenclator positions;
- training: dynamic random word masking on corpus `train` only;
- inference: aggregate contextual log probabilities across every occurrence of each recurrent code symbol;
- assignment: Hungarian one-to-one matching between observed code symbols and the frozen candidate codeword pool;
- refinement: up to three deterministic fill-and-rescore iterations;
- no pretrained embeddings or external language model.

The true code words, test text and test codebook are not supplied.

## Development evaluation

The model is evaluated on the same nine frozen A1 frontier cells:

- lengths 384, 768 and 1536;
- candidate pools 32, 64 and 96;
- codebook size 16 for pool 32 and 24 for pools 64/96;
- eight English development chunks per cell.

## Gate

A condition passes A1 only if:

- mean observed-code mapping accuracy is at least 80%;
- mean code-occurrence accuracy is at least 80%;
- mean expanded character recovery is at least 90%;
- at least eight code symbols are observed on average.

The minimum-length passing condition becomes the nomenclator Stage B development environment. If no condition passes, Stage B remains prohibited and nomenclator identity is recorded as underdetermined under the tested source model.
