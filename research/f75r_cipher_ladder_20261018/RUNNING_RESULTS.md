# RETRACTED FINDINGS (TOP OF RUNNING RESULTS)

**R4 / 2026-10-19 / RETRACTED (GENERALISATION):** The 20/20 synthetic inserted-null recovery rate in Italian UD samples is a genuine result for **those samples** but does **not** qualify a one-null decipherment instrument at Voynich word length/type recurrence. In shorter length-/recurrence-matched tests, Italian selected the true null 18/20 times but falsely favoured one-null on 9/20 genuine no-null cases. Latin selected the planted null 15/20 with 4/20 no-null positive. Thus any claim that Italian p=0.05 planted success licensed a Voynich null search is withdrawn; the instrument fails the required false-positive control.

**R3 / 2026-10-19 / RETRACTED:** The 2026-10-18 Italian one-null improvement `+0.2207300` log units per original cipher glyph was obtained with **20 separate one-null candidate searches versus one monoalphabetic search**. It is not an equal-search-budget mechanism effect. Equalising to 20 independent mono restarts overturns stable promotion: Italian f76r heldout gains changed sign across seeds, with selected nulls changing; Latin did not show a replicable advantage. The earlier gain is withdrawn, not silently replaced. See `results/null3_parity_pilot_20261019.json`.

**R1 / 2026-10-18 / RETRACTED:** The automatically emitted `candidate_promotion=true` on a naive within-position glyph-permutation null **does not establish a decipherment, cipher mechanism, source language, or anything language-specific**. All six distinct language models clear this null; the null destroys readily detectable graphotactic structure. Replace the gate with planted recovery **plus** language-specific hard negative controls, chronology-matched witnesses, and genuinely held-out coherent plaintext before promotion. This retraction remains at the top on all updates.

**R2 / 2026-10-18 / BOUNDING CORRECTION:** Initial huge deviations against within-word glyph permutation were non-specific. Under matched whole-word permutation with full refitting, none of six languages exceeds 2 null SD; the corresponding heldout word-order contrast also fails to resolve. This is **not** an exclusion of monoalphabetic substitution because these controls preserve lexical word identities and may not distinguish correct decipherments; positive language-discrimination controls remain to be qualified.

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

## Harder bounding controls (all six GPU jobs complete)

Three sets per language: refit to whole-word-order shuffles, reoptimise on recurrence-preserving Markov(2) generated pseudo-word types, and fixed mapping tested on untouched f76r against line-local word-order shuffles. Cells give **effect / null SD / ratio**. The word-order comparison retains every actual Voynich word so it specifically tests between-word order, NOT dictionary alignment or lexical substitution. Positive control of that semantic discrimination is still outstanding; avoid an exclusion claim.

| Language | Train: refit word-order null | Train: recurrence-preserving Markov(2) null | Heldout: word-order null |
|---|---:|---:|---:|
| la | -0.0034 / 0.0510 / -0.07 | 0.3025 / 0.1051 / 2.88 | 0.0047 / 0.0025 / 1.88 |
| it | 0.0209 / 0.0293 / 0.71 | 0.1488 / 0.0832 / 1.79 | 0.0041 / 0.0046 / 0.90 |
| de | -0.0385 / 0.0372 / -1.04 | 0.2075 / 0.0705 / 2.94 | 0.0035 / 0.0020 / 1.80 |
| cs | -0.0375 / 0.0220 / -1.71 | 0.1516 / 0.0739 / 2.05 | -0.0020 / 0.0015 / -1.36 |
| he | 0.0180 / 0.0175 / 1.03 | 0.1358 / 0.0702 / 1.94 | 0.0047 / 0.0030 / 1.56 |
| yi | -0.0109 / 0.0540 / -0.20 | 0.2152 / 0.0979 / 2.20 | 0.0016 / 0.0031 / 0.53 |

**Interpretation:** All six word-order refit scores are below 2 null SD; all six heldout word-order comparisons are below 2 null SD. The Markov(2) score is higher in several languages, but those controls reproduce only a subset of Voynich graphotactics and do not distinguish lexical decipherment from generic word-formation structure. An automated `candidate_promotion=true` from the prior run is **still retracted**.

Rung 2 one-null search: new exhaustive twenty-way glyph-null branch added with a synthetic injected-null qualification gate; it will not score f75r if that gate fails. Prior calibration never licensed historical language selection.

## Rung 2 completed: one-null glyph, Latin and Italian (2026-10-18)

Job logs (direct primary outputs):
- Latin: https://huggingface.co/jobs/Digitalgoldfish79/6ad52f32c662881452278b01
- Italian: https://huggingface.co/jobs/Digitalgoldfish79/6ad52f3dc662881452278b11
- Initial Latin smoke failed the planted instrument, and was excluded: https://huggingface.co/jobs/Digitalgoldfish79/6ad52dee5ccb47ba0f7c7c35

Full 900-step, 256-way GPU search with exactly one candidate null position enumerated over every glyph; synthetic planted null identification passed independently for both languages before target scoring. Fixed pnull = 0.05, with log-probability scoring per original cipher glyph.

| Language | Planted character accuracy | Correct planted null | Best selected f75r null | f75r deletion frequency | f76r one-null minus mono (per source glyph) | Matched null SD |
|---|---:|:---:|---|---:|---:|---|
| Latin | 93.85% | yes | EVA e | 13.77% | -0.0055784 | not measured |
| Italian | 98.20% | yes | EVA d | 10.65% | +0.2207300 | not measured |

**Inference firewall:** Both real-text effect sizes await a properly matched null standard deviation and controls distinguishing the one-null family from flexible graphotactic overfitting; no historical-language, literal-null or cipher promotion. The pnull=0.05 fixed encoding differs substantially from the selected 10–14% empirical deletion frequencies and must receive prior-sensitivity checks. The comparison uses different optimization runs and equal-search-budget assessment should be confirmed. Do not infer a language rank from raw Latin vs Italian score deltas. Positive no-null synthetic false-selection rate must also be calibrated before promotion.

**Status:** NO CLAIM. Hard family-matched null controls, false-positive rate, and historical corpus fitness remain unqualified. Additional complexity is blocked until those tests can discriminate the alternatives.

## Rung 2B: search-parity correction and full-strength calibration (2026-10-19)

**Primary data:** [results/null3_parity_pilot_20261019.json](results/null3_parity_pilot_20261019.json). Frozen job code commit `cd7c25cd0c84ef87724af3a0deb064a36e33d8ca`; Tesla T4, 650 annealing steps and population 192 per candidate. Each null candidate has 20 trials, and mono has 20 independent restarts; all keys chosen by f75r train score only and evaluated unchanged on f76r. 4 independent *synthetic* draws at each deletion probability 0, 0.05, 0.10, 0.15 for Latin and Italian; four independent search seeds for the *same* Voynich passage are not four independent Voynich observations.

| Model | Real f76r mean minus no-null synthetic mean | Synthetic no-null SD | Effect/SD | Synthetic 5% null symbol found | Synthetic no-null gains >0 | Real gain >0 across seeds |
|---|---:|---:|---:|---:|---:|---:|
| it | 0.189464 | 0.129299 | 1.47 | 4/4 | 0/4 | 2/4 |
| la | -0.071590 | 0.236564 | -0.30 | 4/4 | 1/4 | 1/4 |

**Inference:** metric does not resolve the Voynich one-null mechanism (neither effect magnitude reaches 2 matched-null SD). The 5% planted synthetic null is recovered 4/4 for both languages at this setting, but small no-null sample sizes preclude any calibrated 5% false-positive claim. Source-language adjudication invalid because UD witness distributions, glyph statistics, spelling period and normalisation are not matched to fifteenth-century running text. The common f76r holdout does not create independent sample size across seed reruns. Synthetic no-null experiments were completely independent of Voynich aside from the frozen architecture; their matching quality to Voynich graphotactics is NOT certified.

**Independent-control extension underway:** additional 60 zero-null and 20 five-percent-null draws per language at the same compute settings, fresh seeds, GPU job logs to be appended with positive recovery and false-positive bounds. The prior program-level homophonic and terminal cipher studies were consulted before proposing any new cipher family; no duplicate homophone search authorised.

## Rung 2C: large controls and VMS-length/recurrence matched negatives — FINAL (2026-10-19)

Primary data: [60+20 controls](results/null3_expanded_controls_20261019.json), [length/type-matched controls](results/null3_lengthmatched_controls_20261019.json), [equal-search-budget Voynich keys](results/null3_parity_pilot_20261019.json). Codes frozen at `67ae96aaede567fd43e79ff2ba3ec461eae58962` (expanded controls) and `1c27be4245c08da2831bf97adbb7a00db29d093e` (matched controls). Four folds of *search-seed replication on the same heldout folio* do not constitute four physical test documents. Each synthetic trial independent corpus randomization.

| Language | Null cohort | Real f76r mean minus synthetic no-null mean | Synthetic null SD | Effect/SD | Synthetic no-null false positives | Correct planted 5% null symbols |
|---|---|---:|---:|---:|---:|---:|
| it | standard | 0.124790 | 0.104888 | 1.19 | 11/60 | 20/20 |
| it | matched | 0.032115 | 0.076928 | 0.42 | 9/20 | 18/20 |
| la | standard | -0.071450 | 0.167003 | -0.43 | 22/60 | 13/20 |
| la | matched | -0.068254 | 0.121240 | -0.56 | 4/20 | 15/20 |

**Decisive falsification:** For both languages, the observed null advantage is **less than 2 matched-null SD** in both negative-control cohorts. The preregistered false-positive gate is <=5%, yet Italian is 11/60 (18.3%) standard and 9/20 (45.0%) length/recurrence matched, while Latin is 22/60 (36.7%) standard and 4/20 (20.0%) matched. The 90%-correct-null recovery gate fails for Latin, with 13/20 standard and 15/20 matched; Italian reaches 20/20 standard and 18/20 matched. Counting both correct-symbol recovery AND positive heldout gain further limits power: Italian 20/20 standard, 15/20 matched; Latin 7/20 standard, 12/20 matched. The simple criterion gain>0 alone is a *conservative* false positive diagnostic; no post-hoc threshold tuning is authorised.

**Representation and validity audit:** True f75r TTLI word lengths are 1–7 EVA glyphs (352 tokens, 163 distinct word types); f76r has 473 tokens. Original planted controls were not length/type matched and had inflated recovery power. Length-matched synthetics map genuine heldout UD word forms independently into f75r/f76r's word-length and repeated-type templates. Italian matched no-null set has zero type collisions; Latin has four unavoidable one-letter lexical collisions among 308 template types. Null insertion changes positive-control ciphertext lengths slightly, and plain source corpora are historical-period mismatched; both caveats remain. No synonym or semantic evidence, no morphology validation, no identifiable source language, no qualified mechanism. The original R3 correction remains retracted at file top.

**Disposition: FAILED (no promotion).** Do not treat the Italian `d` or later `l/e` choices as a null or a decipherment. The next rung would require a newly instrument-qualified model and independent fifteenth-century source corpus; do not increase flexibility solely to fit f75r. Our completed July v0.5/v0.6 cipher programmes already evaluated homophonic and polyalphabetic families, with their own qualification failures; avoid duplicating their unsuccessful mechanisms.
