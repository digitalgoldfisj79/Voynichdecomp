# VBM Joachim Exact v9 — Q1 fresh-fit specificity / codebook-cost protocol

Date: 2026-09-01
Branch: `experiment/vbm-joachim-exact-v9-20260901`
Status: **FROZEN BEFORE Q1 OUTPUT**
Depends on: Q0 PASS and source-faithful Q0b PASS.

## Question

Joachim's published f115v.26 example is explicitly a feasibility construction, not a claimed decipherment. Q1 asks the narrower question that such a construction can answer quantitatively:

> Given the source-faithful v9 parser, how selective is an exact proposed plaintext fit before any reusable codebook has been learned?

A VBM line with `B` inter-token bridges has `B+1` nucleus positions. After spaces are removed from a proposed plaintext, its vowel/consonant decomposition is therefore forced into:

`C0 V0 C1 V1 ... V(B-1) C(B)`

where each `V` is one of `a,e,i,o,u`, each non-empty `C` is a consonant string of length 1–5, and empty nuclei require empty consonant runs.

Under the v9 global-value rule an exact fit additionally requires:

1. repeated occurrences of the same bridge surface type map to the same vowel;
2. repeated occurrences of the same non-empty nucleus surface type map to the same consonant run;
3. different surface types may be homophones;
4. line-edge halves emit nothing in this feasibility audit;
5. no contextual, folio-specific, or hand-specific values are permitted.

This is an exact constraint check. There is no optimiser and no language score in Q1.

## Parser

Binding parser is Q0b:

- longest initial atom from `ckh, cth, cph, cfh, ch, sh, qo` when followed by at least one further character;
- otherwise first character is `L`;
- final character is `R`;
- substring between `L` and `R` is nucleus `N`;
- one-character tokens use `SINGLE_SHARED` in the primary analysis;
- bridge between adjacent tokens is `R_i|L_(i+1)`;
- no bridge crosses a line.

The published f115v.26 fixture must parse and exact-fit the supplied plaintext `tizsichtrageundegetnichtsnelle`.

## Target firewall

The consumed diagnostic folios remain excluded:

`f28v f31v f88r f5r f34r f81v`

The sealed C1 folios remain excluded:

`f85r1 f53v f33r f10r f23r f111r`

Q1 uses only the Q0 TRAIN + INTERNAL_HOLDOUT pool. It performs no decoding and does not inspect H1/C1.

## Line sample

Eligible lines:

- all tokens valid `[a-z]+` under the source-faithful parser;
- 5–15 bridges inclusive;
- at least 6 tokens;
- no target-firewall folio.

Sampling is deterministic and stratified by bridge count. At most 80 lines are taken from each bridge-count stratum, ordered by SHA256 of `VBMJOACHIMEXACTV9Q1::<folio>::<line>`.

The f115v.26 fixture is audited separately and is not inserted into the random line sample.

## Plaintext candidate distributions

Primary natural-language candidate banks are generated independently of Voynich from the `wordfreq` frequency lists using fixed seed `90117`:

- `DE` — German;
- `IT` — Italian;
- `EN` — English.

Words are ASCII-normalised, punctuation/spaces removed after deterministic sampling, and concatenated. These banks are **not historical-language evidence**; they are only natural-orthography capacity controls.

`SHUFFLED_DE` is formed by deterministically shuffling the characters of the German bank and is a non-language marginal-character control.

For a line with `B` bridges, a candidate is a sequence of `B` consecutive vowels and the `B+1` complete consonant runs surrounding them. Primary candidates start/end on consonant-run boundaries. Edge-truncated runs are not allowed in Q1; this is conservative because allowing them would increase VBM flexibility.

At most 5,000 candidate windows per line per bank are sampled deterministically without reference to fit status.

## Headline outputs

For every line and candidate bank report:

- candidate windows inspected;
- exact fits;
- exact-fit fraction;
- smoothed fit probability `(fits+1)/(candidates+1)`;
- fit surprisal `-log2(p_smoothed)`.

Aggregate by bank:

- median exact-fit fraction;
- median fit surprisal;
- fraction of lines with >=1 exact fit;
- fraction with >=10 exact fits;
- bridge-count-stratified results.

No p-value is attached to Joachim's chosen Breslau sentence because the sentence was not selected under a preregistered sampling process.

## Codebook charge

For every exact fit the incremental fresh-key description cost is:

`K = K_bridge * log2(5) + sum_unique_nucleus [log2(5) + len(value)*log2(21)]`

where `K_bridge` is the number of unique bridge surface types on that line and the nucleus sum runs over unique non-empty nucleus surface types.

For each line/bank report the minimum `K` among fitting candidates. Also report:

`NET_SINGLE_LINE = fit_surprisal - min_key_bits`

This is a diagnostic single-line MDL balance, not a corpus likelihood ratio. A positive value would mean the structural rarity of the fit exceeds the newly introduced dictionary description length; a negative value means the fresh dictionary has more descriptive capacity than the fit supplies.

The published f115v.26 assignment is charged directly using Joachim's stated mappings.

## Structural shuffle null

For a deterministic subset of up to 120 sampled lines, generate 20 null variants each by independently shuffling the nucleus-position sequence and bridge-position sequence while preserving their multisets. This preserves:

- token/bridge count;
- number of empty nuclei;
- surface-type frequency multiset;

but destroys where repeated types occur.

Score the shuffled variants against the same DE candidate bank. Compare real-line median DE fit surprisal with the distribution of shuffle medians. This is a topology-specificity diagnostic only.

## Interpretation bands

These bands are frozen before Q1 output and are descriptive gates for whether a *fresh-key example* is selective enough to motivate stronger evidence claims:

- **HIGHLY NONSELECTIVE**: median DE fit fraction >= 0.01, or >=50% of lines have >=10 fits among the sampled candidates.
- **NONSELECTIVE**: median DE fit fraction >= 0.001, or >=50% of lines have at least one exact fit.
- **INTERMEDIATE**: median DE fit fraction in `[1e-4, 1e-3)` and real topology is at least 2 shuffle-SD more selective than shuffled topology.
- **SELECTIVE**: median DE fit fraction < 1e-4, >=75% of lines have no exact fit in 5,000 candidates, and real topology is at least 2 shuffle-SD more selective than shuffled topology.

If conditions conflict, use the less favourable band.

These bands do **not** validate VBM as the Voynich cipher. They only quantify whether fresh post-hoc sentence fitting is easy or hard under the published architecture.

## Decision / next stage

- If HIGHLY NONSELECTIVE or NONSELECTIVE: Joachim-style one-line feasibility readings receive no evidential weight. A future VBM test must use a reusable codebook learned without the target plaintext; no further fresh-fit demonstrations can promote the model.
- If INTERMEDIATE or SELECTIVE: proceed to a separately preregistered global-codebook transfer/identifiability stage.

Regardless of band, a global reusable-codebook experiment may be run only after a synthetic positive/negative qualification protocol is frozen. Q1 itself never opens H1 or C1.

## Stopping rules

1. No parser changes after Q1 output.
2. No vowel-set changes.
3. No nucleus length >5.
4. No contextual values.
5. No per-line dictionaries in any later evidential test.
6. No historical-language inference from the wordfreq controls.
7. Negative and null results are retained permanently.
