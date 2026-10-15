# VBM v142b surface-seam diagnostic — 14 October 2026

## Correction and bounds

- Earlier v142/v143 B-reference trigram-support failure remains unchanged: 60,454 / 68,396 development words and 11,745 / 13,232 Hartlieb words contain at least one unattested raw 3-gram.
- The failures **all cross a join between generated components**. This is unsurprising in part because the code cells were drawn from the same B corpus used to establish the support inventory. It is a deterministic engineering diagnosis, not evidence of historical encipherment.
- An intermediate one-escape capacity calculation falsely treated every second cell as though it had to be independently compatible immediately after a bridge. That extra restriction is not part of the architecture. **Retract** the resulting conditional `o`-vowel capacity claim (195). Corrected optimistic maximum is **236** using a second cell without this extra restriction. All capacities are necessary/optimistic counts, not demonstrations of fully admissible output words.

## Source and audit

Input: original `vbm_v143_crosscarry/vbm_v142_inband/04_pair_bridge_manifest.json`, the two original `*.pair.vbm` ciphertexts and `vbm_v143_crosscarry/VMS_B_REVA_reference.txt`. The source hashes are recorded in `RESULTS.json`.

The audit independently parses every v142b emitted word into **bridge** and **cell** components, checks roundtrip of the parsed component string to the ciphertext word, then locates every character trigram absent from the same supplied Currier B reference. Each absent trigram is classified according to whether it lies entirely inside a component or crosses a component boundary. It reproduces the previously reported counts exactly.

| Corpus | Generated words with unsupported trigram | Bad trigram occurrences crossing bridge-related join | Bad trigram occurrences crossing cell–cell join | Bad occurrences wholly inside components |
|---|---:|---:|---:|---:|
| Development | 60,454 / 68,396 (88.388%) | 129,272 (93.94% of failures) | 8,344 (6.06%) | 0 |
| Hartlieb | 11,745 / 13,232 (88.762%) | 22,821 (92.75%) | 1,784 (7.25%) | 0 |

These are deterministic counts, with no meaningful matched-null standard deviation for the intra/reference-support identity.

## One-cell and one-escape capacity screen

Per vowel, the current v142b prefix-free inventory is first filtered for whether the actual `bridge + first-cell` string has all its character bigrams and trigrams attested in B. This yields 207 (`a`), 198 (`e`), 117 (`i`), 98 (`o`) and 187 (`u`) directly join-compatible cells, below the frozen 222 source-unit count.

A separate *optimistic* one-escape screen reserves one compatible first cell and permits each other original state cell as a continuation **without falsely requiring the continuation itself to follow a bridge**. The maximum number of available distinct encodings becomes 410, 394, 255, 236 and 392 respectively. These values exceed 222, so no simple cardinality impossibility is established for this frozen v142b inventory. They do **not** ensure downstream cell-to-cell seams, longer words, global code assignment or manuscript distribution can all be satisfied.

## Decision

The immediate engineering failure is at **compositional joins**, especially bridge-to-cell joins. There is no reason to rerun the synthetic cross-state association tests to establish that. The next legitimate *feasibility* question is whether a **single globally reversible assignment** of the 222 units to seam-compatible cells can obey frozen native surface constraints on source-held-out words, tested separately against A and B. Even if feasible, the v144 autonomous transducer is observationally equivalent and **would still prevent a cipher attribution from ciphertext alone**.

## Files

- `audit.py`, `RESULTS.json`: deterministic component seam decomposition.
- `capacity.py`, `CAPACITY.json`: corrected optimistic seam-compatible capacity bound.
- This bundle depends on the previous v142/v143 frozen archive for the source ciphertext, inventory and B-reference transcription.
