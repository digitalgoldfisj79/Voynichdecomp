# Recoverability frontier v0.5.3 — neural/classical homophonic bake-off

Date frozen: 2026-07-16

Status: **development protocol; no Voynich inference permitted**

## Purpose

v0.5.2 established that bounded fresh-key homophonic ciphertexts are often almost completely recoverable, but the selected classical solver reaches the correct basin unreliably. A direct `MESSAGE/NO_MESSAGE` classifier remains prohibited.

v0.5.3 compares materially different decipherment architectures on exactly the same English corpus partitions and synthetic cipher family before further large-scale restart expenditure.

## Scientific boundaries

- No Voynich text is scored.
- Plaintext language and cipher family are known in this stage.
- Every fresh-key example receives an independently sampled homophonic key.
- Raw synthetic cipher-symbol integers are never supplied. Inputs use first-occurrence recurrence canonicalisation.
- Corpus `train`, `dev` and `test` partitions remain hash-pinned and disjoint.
- Development selects architecture and schedules. Untouched test blocks are opened only after selection.
- Shared-pool results are a positive control and may not be reported as fresh-key recovery.

## Fixed development environment

Primary hard cell:

- language: English;
- cipher family: bounded frequency-adaptive homophonic substitution;
- normalized plaintext length: 384 characters;
- channel noise: 0%;
- development chunks: corpus `dev`, replicates 0–7;
- initial inferred homophone inventory: unchanged from v0.5.2.

Secondary checks may use 96 and 192 characters after the primary 384-character gate.

## Arm A — classical independent-restart curve

Architecture:

- strict fixed-inventory CrypTool-style exhaustive pair sweeps;
- train-only quadgram objective;
- calibrated target initial acceptance `0.05`;
- `3,000,000` pair proposals per independent restart;
- no inventory mutation.

One maximum run records prefix-best results after:

`12, 24, 48, 96, 192` restarts.

This prevents recomputing the same trajectories for separate restart grids.

## Arm B — neural-language-model beam search

Architecture:

- train-only character LSTM language model;
- cipher symbols assigned in first-occurrence order;
- the inferred plaintext-label multiset is consumed exactly once;
- beam states preserve the LSTM hidden state and cumulative prefix log-likelihood;
- when a new cipher symbol is assigned, the newly decipherable maximal prefix is scored;
- an admissible length-normalised rest-cost term prevents systematic preference for short resolved prefixes;
- final candidates are rescored over the complete plaintext.

Development grid:

- beam widths `128, 512, 2048`;
- one- and two-layer LSTM language models;
- neural LM trained only on corpus `train`.

## Arm C — fresh-key recurrence decoder

Architecture:

- Transformer encoder-decoder trained from scratch;
- input: first-occurrence recurrence IDs plus position embeddings;
- target: plaintext character IDs;
- every synthetic training example receives a fresh key;
- online generation from corpus `train` chunks;
- development uses corpus `dev` chunks and unseen keys;
- no shared symbol pool and no pretrained language model.

The model reports greedy recovery and constrained key-consistent recovery. The constrained output estimates one plaintext label per recurring cipher symbol from decoder posteriors, then reapplies that key to the whole ciphertext.

Development grid:

- `d_model` 256 and 384;
- 6 and 8 Transformer layers;
- at least 1,000,000 fresh-key training examples for the selected full run.

## Arm D — shared-pool LSTM positive control

Architecture:

- fixed homophone code pool shared across training and evaluation;
- individual examples activate random non-empty subsets of each plaintext character's pool;
- attention-augmented bidirectional LSTM sequence labeller;
- stable symbol identities are intentionally available.

This arm answers only whether the implementation reproduces the easier shared-pool setting described in recent literature. It is not eligible to win the fresh-key bake-off.

Positive-control gate:

- at least 95% mean character recovery on unseen texts and unseen active subsets drawn from the same pool.

## Arm E — neural-seeded classical hybrid

The fresh-key recurrence decoder supplies per-cipher-symbol plaintext posterior scores. Up to 32 high-probability key seeds are constructed subject to the fixed inferred inventory. Each seed receives a short strict CrypTool-style refinement under the unchanged quadgram objective.

The hybrid is evaluated in the same job and on the same development examples as Arm C. It must report both pre-refinement and post-refinement recovery.

## Primary development metrics

For each fresh-key arm:

- mean and median normalized character recovery;
- exact recovery rate;
- fraction of trials with recovery at least 70%, 90% and 95%;
- mean wall-clock and accelerator time per ciphertext;
- inferred-inventory overlap where applicable;
- objective score of the selected candidate;
- per-trial result rows.

## Development selection rule

An arm is eligible for untouched testing only if, on the eight hard development ciphertexts:

- mean recovery is at least 70%;
- median recovery is at least 90%;
- at least 7/8 trials recover at least 70%;
- there is no use of test data, true keys or true inventories.

Among eligible arms, select the Pareto frontier by reliability and compute. Do not select solely by mean recovery.

## Untouched test blocks

Fresh-key arms that pass development are evaluated once on English test replicates 128–147. The earlier blocks 0–115 remain excluded because they were used in previous diagnostics or locked tests.

Shared-pool positive control uses a separately generated deterministic test seed bank and does not share its symbols or keys with fresh-key arms.

## Locked English test gate

- mean recovery at least 70%;
- median recovery at least 90%;
- at least 16/20 trials recover at least 70%;
- no post-test architecture or threshold changes.

## Advancement

If at least one fresh-key arm passes the locked English test, rerun that arm across six languages and lengths 96, 192 and 384 using untouched source blocks.

If no fresh-key arm passes, record bounded homophonic substitution as **recoverable in favourable basins but not reliably identifiable under the tested compute and model class**, and proceed to a different cipher family rather than continuing ad hoc search modifications.
