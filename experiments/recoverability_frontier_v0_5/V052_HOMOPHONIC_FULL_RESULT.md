# Recoverability frontier v0.5.2 — six-language homophonic result

Date: 2026-07-15

Verdict: **FAIL FROZEN GENERALISATION GATE; DIAGNOSE LONG-TEXT INSTABILITY BEFORE NULLS**

No Voynich text was scored.

## Frozen architecture

- first-occurrence recurrence canonicalisation;
- bounded variable homophone inventory;
- train-only character quadgram model, alpha `0.05`;
- unigram score weight `0.12`;
- deterministic global annealing, reheating and greedy polishing;
- fixed schedule `700,000 × 50`;
- no nulls and no channel noise;
- test source-chunk block 64–83, untouched before execution.

## Aggregate result

Across 360 unseen-key trials:

- mean character recovery: **83.8462%**;
- frequency-slot baseline: **26.1798%**;
- exact plaintext recovery: **26.6667%**;
- mean initial homophone-inventory overlap: **91.6286%**;
- mean final inventory overlap: **94.6984%**.

| Length | Mean recovery |
|---:|---:|
| 96 | 79.3490% |
| 192 | 92.9557% |
| 384 | 79.2339% |

Aggregate summary SHA-256: `ba8f60423ec5c4f28829ddd5e0e0a3923192f529a35b96838a259d87734f0a11`

## Results by language

| Language | Mean recovery | 96 chars | 192 chars | 384 chars | Exact | Baseline | Shard SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| English | 57.1224% | 88.2292% | 73.9063% | **9.2318%** | 18.33% | 23.8845% | `787618e7bc3d04011cbb5268181d08a4016cd756244b08f8f870262cbb54adcc` |
| German | 97.1181% | 92.7083% | 99.1146% | 99.5313% | 30.00% | 26.5061% | `86c3d871b82474d53d8c66106e0d581a411dd7329012cb4dbc2c7c8a71501589` |
| Finnish | 98.4115% | 97.3438% | 98.4635% | 99.4271% | 46.67% | 23.1684% | `13999ee802c8034ea67cdb27f924b4cfd40a38ac207d6ef27de95028ef5c654f` |
| Turkish | 85.9332% | 59.8438% | 98.1771% | 99.7786% | 31.67% | 23.1076% | `448dacc3ab298235d7e5d36409d30bb247f851bf3e7fced05352de79a4c7ee09` |
| Hebrew | 70.7769% | 52.8125% | 91.6406% | 67.8776% | 18.33% | 28.7326% | `fec391e7d0eaf33f42cedf73b6a73a76495024086cc137dcd1412c2b7d5fc724` |
| Arabic | 93.7153% | 85.1563% | 96.4323% | 99.5573% | 15.00% | 31.6797% | `019bc1967d96689fa4f22d49d0ea38f26c6de0314aaf278932998f0f1969dd86` |

## Job provenance

- English: `Digitalgoldfish79/6a58075085d9643ce16d58b6`
- German: `Digitalgoldfish79/6a58075cb1669a49bf076352`
- Finnish: `Digitalgoldfish79/6a58076585d9643ce16d58bc`
- Turkish: `Digitalgoldfish79/6a58076d85d9643ce16d58be`
- Hebrew: `Digitalgoldfish79/6a58077485d9643ce16d58c0`
- Arabic: `Digitalgoldfish79/6a58077bb1669a49bf076354`

Each immutable job log contains its complete gzip/base64 row-level JSON artifact.

## Frozen gate

- overall mean at least 70%: **pass**;
- every language at least 50%: **pass**;
- aggregate 96-character recovery at least 60%: **pass**;
- longer texts do not materially collapse: **fail**.

The failure is not a general long-text limitation. German, Finnish, Turkish and Arabic approach complete recovery at 384 characters. It is concentrated in English, with a secondary Hebrew weakness.

## Interpretation

The family-specific homophonic architecture is viable in principle and substantially outperforms the frequency baseline. However, the fixed language-model/search combination is not stable across all corpus and length environments. In particular, more text can drive the English search toward a high-scoring but incorrect basin.

This result prohibits:

- proceeding directly to null-bearing homophonic substitution;
- reporting a general homophonic solver pass;
- hiding the English failure inside the strong aggregate;
- applying this architecture to the Voynich Manuscript.

## Required next diagnostic

For English 384-character trials, and secondarily Hebrew 384-character trials, compare:

1. score of the true observed key;
2. score of the recovered key;
3. recovery with the true observed homophone inventory;
4. recovery under alternative train-only language-model orders or interpolated objectives;
5. results by source genre/domain and unseen-character inventory.

If the recovered key outscores the true key, the source model is misspecified for that corpus environment. If the true key outscores the recovered key, search remains the bottleneck. No solver modification may be treated as validation without a new untouched test block.
