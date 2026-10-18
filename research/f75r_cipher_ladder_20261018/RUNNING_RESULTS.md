# RETRACTED FINDINGS (TOP OF RUNNING RESULTS)

**R1 / 2026-10-18 / RETRACTED:** The automatically emitted `candidate_promotion=true` on a naive within-position glyph-permutation null **does not establish a decipherment, cipher mechanism, source language, or anything language-specific**. All six distinct language models clear this null; the null destroys readily detectable graphotactic structure. Replace the gate with planted recovery **plus** language-specific hard negative controls, chronology-matched witnesses, and genuinely held-out coherent plaintext before promotion. This retraction remains at the top on all updates.

# F75R PROGRESSIVE BRUTE-FORCE LADDER: RUNNING RESULTS

Source: TTLI f75r, 352 words and 20 atomic EVA glyph types; training page f75r, heldout same-ecology page f76r (5 unknown glyph instances' **words**, not glyphs, excluded). Caesar conventional glyph-order sweep then 26-letter injection monoalphabetic GPU parallel population annealer. All corpus transcriptions and source SHAs pinned in [results/rung1_gpu_results.json](results/rung1_gpu_results.json). Calibration uses independently heldout UD text; source corpora are not fifteenth-century witnesses.

| Language | Planted character recovery | Voynich training dictionary hits | Naive score advantage | Naive null SD | SD units | GPU job |
|---|---:|---:|---:|---:|---:|---|
| la | 93.2% | 3.4% | 0.8622 | 0.0307 | 28.1 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a58c6628814522785f1) |
| it | 95.8% | 21.9% | 0.9127 | 0.0328 | 27.8 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a635ccb47ba0f7c78b2) |
| de | 95.5% | 27.8% | 0.8390 | 0.0223 | 37.6 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a6f5ccb47ba0f7c78b8) |
| cs | 95.0% | 25.0% | 1.0347 | 0.0299 | 34.6 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a7bc66288145227860f) |
| he | 100.0% | 30.4% | 0.4405 | 0.0131 | 33.6 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a87c662881452278629) |
| yi | 93.0% | 24.4% | 0.5057 | 0.0176 | 28.7 | [job](https://huggingface.co/jobs/Digitalgoldfish79/6ad52a92c66288145227862d) |

## Qualified statement

The six planted monochrome substitution tests passed the prewritten loose recovery gate for instrument operation. **This establishes that the solver can recover some genuine monoalphabetic substitutions**, not that Voynich contains one. Correspondingly, the Voynich outputs are largely nonlexical; word-hit percentages cannot be compared directly between different reference dictionaries or corpora. Naive null z scores are non-specific. 

## Active audit

One additional programme tests fixed-key word-order shuffles, re-optimised whole-word shuffles and Markov(2) pseudo-word replacement with preserved token recurrence and lengths. No language promotion or cipher adjudication before these tests complete and their own positive-control qualification.

## Compute budget

Six T4 GPU jobs, each capped at 35 minutes at the current public T4 rate $0.40/h, gives a combined time-limit ceiling of $1.40 excluding short smoke/probe jobs. The added bounding jobs have independent separate caps. This is a cap estimate, not verified billing data or an assertion of account balance.

## Scope limitations

- Simple substitution was fitted to one folio, not deciphered.
- UD reference corpora vary widely in period/genre and tokenize Hebrew/Yiddish via Unidecode, which is merely a lossy proxy, not a fifteenth-century transliteration model.
- The page-internal permutation nulls were not adequate to identify language.
- No plaintext translation, semantic claim or historical cipher inference is licensed.
