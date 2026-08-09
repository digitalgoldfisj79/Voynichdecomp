# Amendment 002 — H20 execution details

Date: 2026-08-09
Status: prospective; frozen after RF surface-only census and before any Paduan-language T20/H20 score.

No scientific threshold is changed.

## RF surface assertions

The source-only preflight must reproduce RF1b SHA-256 `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`, 157,254 parsed full-STA characters, and the already-frozen 99.5%-coverage vocabulary size K=36. The exact 36-symbol vocabulary is frozen by descending RF corpus frequency with lexical tie-break.

A rare/non-retained STA member is a hard scoring break. IVTFF locus/line boundaries, uncertain bracketed readings, and drawing intrusions are also hard breaks. Word spaces are not cipher symbols.

## H20 buckets

For each H20 folio, define

`bucket = SHA256("TRANCHSTA20Hbucket::<folio>")[0] mod 4`,

where `[0]` means the first digest byte. Each folio belongs to exactly one of four buckets.

## Order-shuffle null

For null replicate `r = 0..199`, independently permute the retained full-STA symbols inside each uninterrupted retained segment using deterministic seed

`SHA256("TRANCHSTA20Hshuffle::<r>::<folio>::<segment_index>")`.

Hard breaks and segment lengths are therefore preserved exactly. The frozen T20 key is not refitted for any null replicate.

The overall 99th percentile is NumPy's default linear empirical quantile at `q=0.99`. Bucket medians use the ordinary median over the 200 corresponding null scores.

## Paduan model transport

The exact recovered Paduan training partition is converted to sparse raw counts before remote execution:

- alphabet: `abcdefghilmnopqrstu`;
- additive smoothing: alpha=0.5 as frozen in Amendment 001;
- observed quadgram entries: 24,159;
- transport archive SHA-256: `3c56ab107934c140adcbf70e26dfef373fbf829648ccdd7276ecb5333976229f`.

The transport archive contains sorted uint32 quadgram indices, raw uint16 observed counts, raw uint32 unigram counts and alpha only. The remote runner reconstructs the probability tables from these counts. Thus model transport does not quantise log probabilities or alter the frozen scoring model.

The 12 Stage-A0 fixed-map positive-control scores are reconstructed under the same probability model. The binding 5th-percentile absolute-score floor is `-2.3672276834921244` nats/scored event.
