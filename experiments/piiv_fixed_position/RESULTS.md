# PIII-FIXED-POSITION results

> **Historical-label correction:** the original branch/path says `piiv`, but source inspection shows that the fixed-second-character coverword table is in *Polygraphia* **III**. See `ERRATUM.md`. This correction changes no analysis or result.

**Formal verdict: `FAIL_EXACT_FIXED_POSITION_PAYLOAD`.**

## Test

The literal *Polygraphia* III mechanism predicts that one fixed character inside every coverword—historically the second character—forms the plaintext stream. The Voynich test therefore extracted nine frozen positional streams (`F1`–`F6`, `L1`–`L3`) and attempted to recover each as a globally consistent monoalphabetic substitution of external Latin or Old Italian.

Mappings were learned on complete approximate-quire training groups and evaluated on held-out quires. Language models were trained only on external *Secreta Secretorum*, *Picatrix*, and *Rettorica* corpora. Davis hand labels and Voynich-derived language models were not used.

## Primary result

| Quantity | Result |
|---|---:|
| F2 held-out language gain | **−1.45795 bits/character** |
| F2 family-wise permutation p | **1.0000** |
| F2 nominal permutation p | **0.7228** |
| F2 fold mapping agreement | **0.3261** |
| Best position | **F1** |
| Best position gain | **−1.25272 bits/character** |

A positive gain means that the recovered sequence is better predicted by an external trigram language model than by its corresponding unigram model. The F2 value was strongly negative: the fitted trigram model made held-out prediction worse, not better.

The family-wise null repeated the complete nine-position search 100 times after independently permuting extracted characters within each approximate quire while preserving symbol frequencies, line lengths, folios, sections, and position eligibility. Every null produced a maximum positional gain at least as large as the observed F2 gain.

## Positive calibration

| Planted control | Held-out gain | Planted mapping recovered |
|---|---:|---:|
| Latin | +0.27408 | 72.73% |
| Old Italian | +0.93975 | 90.48% |

The same pipeline therefore recovered the historical mechanism when a monoalphabetically disguised natural-language character stream was actually planted in the F2 line/quire template.

## Transcription sensitivities

| Transcription | F2 gain | Nominal p | Best position |
|---|---:|---:|---|
| ZLZI | −1.45795 | 0.7228 | F1 |
| ZLZB | −1.36475 | 0.1000 | F1 |
| TTIA | −1.40105 | 0.4250 | F1 |

No transcription selected F2, no transcription produced positive F2 language gain, and none produced a significant F2 result.

## Decision ledger

- Positive calibration: **pass**
- F2 best or tied-best: **fail**
- Family-wise p ≤0.05: **fail**
- Mapping agreement ≥0.70: **fail**
- Alternate-transcription replication: **fail**
- Gain within 0.50 bits/character of the weaker positive control: **fail**

## Interpretation

This rejects the literal fixed-position formulation:

> one Voynich token represents one plaintext character, recovered from the same glyph position in every token, under one global monoalphabetic mapping.

It therefore directly rejects the historical second-character mechanism as a manuscript-wide explanation under ordinary EVA word segmentation.

The result does not reject changing payload positions, polyalphabetic or scribe-specific mappings, multi-glyph payloads, insertions/deletions/nulls, sparse encrypted passages, or a separate post-encryption surface realiser. Those are different and substantially more flexible hypotheses.

## Historical source recovery

The exact table was independently located in the 1620 French *Polygraphie* on Gallica. The translated “Troisième livre” begins at frame 179; its table occupies the pages before the epilogue at frame 214. “Quatriesme livre” begins at frame 215 and concerns transposition tables. The table entries visibly follow the second-letter rule—for example the columns around frame 185 progress through coverwords whose second characters encode the alphabetic row. A separate OCR/transcription quality-gated harvester is required before using that table for an exact forward-surface test; the fixed-position payload result above does not depend on OCR of the historical table.
