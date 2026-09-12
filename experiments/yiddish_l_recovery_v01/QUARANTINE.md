# QUARANTINE — L recovery v0.1

## RETRACTIONS / PERMANENTLY INADMISSIBLE STATE

1. Run `34714545374` (head `f9f6358240f8b58770b708c5af926949ae2237b2`) is **PERMANENTLY INADMISSIBLE**. It used the correct seven ReF XML payloads but the wrong measurement unit: individual `tok_dipl` fragments instead of CorA virtual `<token>` units.
2. **Retraction:** the earlier required-action wording below identified commit `0f9e30ee32185cc65591b2da4867fa3bb8b23574` as containing a packaged ReF XML tarball. It did not. That commit configured an Actions workflow intended to create/upload the package as an artifact. The successful Candidate-2 corpus artifact was instead recovered from workflow run `34693265951`.
3. No v0.1 solver output, aggregate, call, score, apparent threshold crossing, or failure is scientific evidence. None may be reused by v0.1b.

## Source-level falsification of v0.1

The successful Candidate-2 corpus artifact was recovered and its exact seven ReF XML files independently checked. Their SHA-256 values match the Candidate-2 freeze byte-for-byte. The corrected extraction rule — concatenate all direct-child `tok_dipl/@utf` fragments within each CorA `<token>`, then apply literal lowercase ASCII `a-z` reduction and drop only empty results — reproduces all seven frozen Candidate-2 word counts exactly, with zero residual discrepancy.

By contrast, v0.1 iterated `tok_dipl` elements independently and produced inflated counts. This establishes a representation-unit defect rather than a corpus-version difference.

| work | correct frozen count | v0.1 fragment count | excess |
|---|---:|---:|---:|
| F014 | 19,097 | 19,544 | +447 |
| F015 | 18,162 | 18,811 | +649 |
| F016 | 8,124 | 8,385 | +261 |
| F018 | 6,270 | 6,589 | +319 |
| F034 | 5,663 | 5,849 | +186 |
| F037 | 15,327 | 15,897 | +570 |
| F148 | 20,282 | 21,556 | +1,274 |

Expected exact source XML SHA-256 values, all recovered exactly:

- F014 `059211745db8c12b96d9ac4538cd94201c758f47cbc756b5fca70eeb59872f76`
- F015 `29b07d566cc8f3ceedbac16a5a0f98a9a4d8059b748c53d336d97aaffcd250f9`
- F016 `b898a98189fe8b3d20867dd347f71639643ee609513b76d5d85f19048bf80284`
- F018 `c694913a336563a5526b5447279359ac822da86e9ecf9ea50d1b30857de8368b`
- F034 `9d4e2da5fe4ac5f0438fd7599234749f4b3fe02333f60f19fed1e27215829135`
- F037 `8f7e855a96c3a6fcf03c4d91102197ddbd9b93fc79ab94393854f35385825f7d`
- F148 `4ca05d5794994980356dd3f9c932f8403a42d964745950dd1966d121d36c2802`

The exact Candidate-2 counts are:

- F014: 19,097
- F015: 18,162
- F016: 8,124
- F018: 6,270
- F034: 5,663
- F037: 15,327
- F148: 20,282

No threshold, solver outcome, aggregate result, or target result caused this quarantine. The defect was found during representation audit before any L outcome was admitted. Voynich/C7 remains sealed.

## Consequence

v0.1 cannot be rehabilitated. A separately frozen v0.1b is required and may repair **only** the ReF extraction unit while inheriting all scientific budgets, thresholds, nuisance controls and decision rules unchanged. v0.1b must create fresh preflight/private artifacts and rerun all solver jobs; `34714545374` is never an input.
