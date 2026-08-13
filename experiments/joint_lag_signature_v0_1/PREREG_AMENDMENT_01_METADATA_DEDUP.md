# Preregistration amendment 01 — metadata-only frame deduplication

Date: 2026-08-13. Frozen **before any VMS hypothesis statistic was computed**. The only inspection performed after the original preregistration was corpus size/coverage and SHA-256 hashing of the literal transcription rows.

The >=10,000-token eligibility rule yields 16 frame IDs, but four pairs are byte-identical across all supplied rows:

- `GCGA` = `GCGI`
- `PCCA` = `PCCI`
- `TTIA` = `TTII`
- `FFSG` = `FFSG-1`

Counting both names as independent replications would be pseudoreplication. Therefore the **primary cross-frame decision rules are evaluated on unique transcription contents**, retaining the first ID in each identical pair. Alias-inclusive counts are reported descriptively only.

Primary unique eligible frame IDs: `GCGA`, `VDRB-1`, `TTVE`, `TTIA`, `ZLZB`, `ZLZI`, `TTLI`, `VDRB`, `FFSG`, `FFSG-2`, `RGVN`, `PCCA` (12 unique contents).

No threshold, hypothesis, statistic, or null has been changed.
