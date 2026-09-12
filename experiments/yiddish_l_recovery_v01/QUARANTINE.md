# QUARANTINE — L recovery v0.1

## RETRACTED / INADMISSIBLE STATE

Run `34714545374` (head `f9f6358240f8b58770b708c5af926949ae2237b2`) is **QUARANTINED BEFORE ANY RESULT INTERPRETATION**.

Reason: after v0.1 was frozen but before any solver outcome was admitted, an extraction-audit found that the new `ref_words()` implementation is not demonstrably equivalent to the previously frozen German representation used in Candidate 2. The prior frozen representation is ReF CorA-XML `tok_dipl/@utf` at the diplomatic/source layer with **diplomatic fragments concatenated per virtual token before literal a-z reduction**. v0.1 currently iterates `tok_dipl` elements independently and may therefore change the measurement unit.

This is a representation-dependence defect, not a negative scientific result. Any v0.1 solver or aggregate outputs are inadmissible unless an independent source-level equivalence test proves that the two extraction procedures produce exactly the same token sequences for all seven registered ReF works (`F014`, `F015`, `F016`, `F018`, `F034`, `F037`, `F148`).

Expected Candidate-2 source-layer token counts, frozen from the prior primary-data audit:

- F014: 19,097
- F015: 18,162
- F016: 8,124
- F018: 6,270
- F034: 5,663
- F037: 15,327
- F148: 20,282

Expected exact source XML SHA-256 values:

- F014 `059211745db8c12b96d9ac4538cd94201c758f47cbc756b5fca70eeb59872f76`
- F015 `29b07d566cc8f3ceedbac16a5a0f98a9a4d8059b748c53d336d97aaffcd250f9`
- F016 `b898a98189fe8b3d20867dd347f71639643ee609513b76d5d85f19048bf80284`
- F018 `c694913a336563a5526b5447279359ac822da86e9ecf9ea50d1b30857de8368b`
- F034 `9d4e2da5fe4ac5f0438fd7599234749f4b3fe02333f60f19fed1e27215829135`
- F037 `8f7e855a96c3a6fcf03c4d91102197ddbd9b93fc79ab94393854f35385825f7d`
- F148 `4ca05d5794994980356dd3f9c932f8403a42d964745950dd1966d121d36c2802`

No threshold, solver outcome, or target result caused this quarantine. Voynich/C7 remains sealed.

Required next action: source-only extraction equivalence probe using the exact packaged official ReF XML from Candidate 2 commit `0f9e30ee32185cc65591b2da4867fa3bb8b23574`. If equivalence fails, v0.1 stays permanently inadmissible and a separately frozen v0.1b repair is required.
