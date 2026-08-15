# Historical Operator Pairs v0.1 — preregistered protocol

Date: 2026-08-15

## Question
Do real medieval scribal abbreviation/compression operations move ordinary text toward the Voynich entropy/structure residual, especially the combination of approximately ordinary unigram entropy with unusually low first-order conditional entropy?

## Core design
This experiment uses **paired witnesses of the same historical text** in unexpanded/diplomatic and abbreviation-expanded forms. It measures the empirical transformation vector caused by real scribal abbreviation rather than fitting synthetic abbreviation rules to Voynich.

Discovery corpus: Nuremberg Letterbooks 2–5 (1408–1423), diplomatic transcriptions with and without expanded abbreviations.
Independent replication corpus: ORIFLAMMS Dated and Datable Manuscripts, 101 manuscript transcriptions whose TEI/ALTO releases preserve with/without abbreviation expansion.

Voynich remains sealed until (a) both external schemas are resolved, (b) pairing passes QC, and (c) all metric definitions and decision rules below are frozen.

## External metrics
Computed on paired line/window samples after Unicode NFC normalization and removal of editorial/XML markup while preserving the literal scribal graphemic string:

1. H0 character entropy.
2. H1 first-order conditional character entropy.
3. H2 second-order conditional character entropy where support permits.
4. delta = H0-H1.
5. mean outgoing next-character entropy over observed contexts.
6. fraction of character contexts with one unique follower.
7. character bigram type density.
8. token mean/std length.
9. token type/token ratio and hapax rate.
10. one-edit token-neighbour rate.
11. adjacent repeated-token rate.
12. zlib/bz2/lzma compression ratio.

Primary operator vector is EXPANDED minus ABBREVIATED for every metric, so negative dH1 means historical abbreviation lowers H1.

## Pairing/QC gates
- Nuremberg: >= 5,000 confidently paired diplomatic lines; >= 500 abbreviation-bearing pairs.
- ORIFLAMMS: >= 30 manuscripts with paired with/without-expansion data and >= 1,000 abbreviation-bearing aligned units total.
- For identical no-abbreviation controls, normalized strings must be identical in >= 98% of pairs after removal of expansion markup.
- Samples shorter than 30 alphabetic characters are excluded from entropy inference but retained for schema diagnostics.
- Editorial expansion letters must not be counted in the abbreviated side.
- Literal generic placeholders such as `*`, `<ex>`, or XML tags must not be treated as scribal characters.

If a corpus fails these gates, status is SCHEMA_OR_PAIRING_FAILURE and no Voynich comparison is permitted.

## External inference
Within each corpus, paired bootstrap over documents (Nuremberg: correspondence/page blocks if writer metadata cannot be robustly recovered; ORIFLAMMS: manuscript) with 2,000 replicates. Report median paired shift and 95% percentile CI.

A historical abbreviation effect on the H0/H1 anomaly is externally qualified only if both corpora independently satisfy:
- median dH1 < 0 and its 95% CI excludes 0;
- |median dH0| < |median dH1| OR median d(H0-H1) > 0;
- direction is unchanged after length-matched resampling.

No numerical effect-size threshold is tuned to Voynich.

## Sealed Voynich comparison
Only after external freeze, compute the same metrics on a pinned Voynich transcription already present in the repository. The comparison asks whether the *direction and magnitude* of the empirically observed abbreviation vector could plausibly bridge an ordinary-text baseline toward Voynich.

Primary interpretation:
- HISTORICAL_ABBREVIATION_MECHANISM_SUPPORTED: both corpora externally qualify and the observed H1 shift is in the Voynich-required direction without a contradictory H0 shift, with the Voynich residual lying within the broad external operator distribution on >=2 of H0/H1/delta diagnostics.
- HISTORICAL_ABBREVIATION_DIRECTION_ONLY: both corpora qualify directionally but magnitude is clearly too small or causes incompatible side effects.
- HISTORICAL_ABBREVIATION_NOT_SUPPORTED: one or both corpora show no replicated H1-lowering effect or opposite effect.
- SCHEMA_OR_PAIRING_FAILURE: external data cannot be paired cleanly enough; no target inference.

## Anti-fitting rules
- No changes to metric definitions, thresholds, pairing rules, or corpus inclusion after any Voynich score is computed.
- No lexical matches to Voynich.
- Nuremberg is discovery only; ORIFLAMMS is the independent replication gate.
- Representations/transcriptions of Voynich are sensitivity analyses, not independent replications.

## Sources
- Mayr et al., Scientific Data 12, 811 (2025), DOI 10.1038/s41597-025-05144-z; Zenodo 10.5281/zenodo.13881575.
- Stutzmann et al., Dated and Datable Manuscripts dataset, Zenodo 10.5281/zenodo.6507965.
