# Supplement S7: EVA Transcription Robustness

## Transcription source

All analyses use the ZLZI consensus transcription (Zandbergen-Landini, voynich.nu),
accessed via the deterministic PGCS decomposition pipeline documented in Paper 1
(Bozzard 2026a). The enriched_records dataset contains 37,465 tokens across 9 sections
and 224 folios.

## Robustness test: ZL vs ZI variants

The ZLZI consensus resolves disagreements between the Zandbergen (ZL) and Landini (ZI)
transcription layers. To test whether PGCS results depend on transcription choice:

**Test A:** Top-20 tokens were compared across ZL and ZI readings. 0/20 tokens differ.

**Test B:** PGCS decomposition was applied to both ZL-only and ZI-only variants for
the 500 most frequent tokens. Slot assignments (prefix, gallows, core, suffix) agree
in all cases where both transcribers provide a reading.

**Test C:** The suffix-family classification (Y, N, L, R, M, BARE) was compared across
transcription variants. No token changes family under ZL-vs-ZI substitution, because
family assignment depends on the final character sequence, which both transcribers
agree on for all high-frequency tokens.

## Top 20 tokens (ZLZI consensus)

| Rank | Token | Count | % | Prefix | Gallows | Core | Suffix | Family |
|------|-------|-------|---|--------|---------|------|--------|--------|
| 1 | daiin | 845 | 2.3% | d | ∅ | ∅ | aiin | N |
| 2 | ol | 542 | 1.4% | o | ∅ | ∅ | l | L |
| 3 | chedy | 503 | 1.3% | ch | ∅ | ∅ | edy | Y |
| 4 | aiin | 497 | 1.3% | ∅ | ∅ | ∅ | aiin | N |
| 5 | shedy | 434 | 1.2% | sh | ∅ | ∅ | edy | Y |
| 6 | chol | 393 | 1.0% | ch | ∅ | ∅ | ol | L |
| 7 | ar | 384 | 1.0% | ∅ | ∅ | ∅ | ar | R |
| 8 | or | 373 | 1.0% | o | ∅ | ∅ | r | R |
| 9 | chey | 351 | 0.9% | ch | ∅ | ∅ | ey | Y |
| 10 | s | 336 | 0.9% | s | ∅ | ∅ | ∅ | BARE |
| 11 | dar | 320 | 0.9% | d | ∅ | ∅ | ar | R |
| 12 | qokeey | 307 | 0.8% | qo | k | ∅ | eey | Y |
| 13 | qokeedy | 306 | 0.8% | qo | k | ∅ | eedy | Y |
| 14 | y | 300 | 0.8% | y | ∅ | ∅ | ∅ | BARE |
| 15 | qokain | 277 | 0.7% | qo | k | ∅ | ain | N |
| 16 | dy | 275 | 0.7% | d | ∅ | ∅ | y | Y |
| 17 | shey | 274 | 0.7% | sh | ∅ | ∅ | ey | Y |
| 18 | qokedy | 273 | 0.7% | qo | k | ∅ | edy | Y |
| 19 | qokaiin | 265 | 0.7% | qo | k | ∅ | aiin | N |
| 20 | al | 257 | 0.7% | ∅ | ∅ | ∅ | al | L |

All 20 are EC tokens (empty core = function words under the cipher model).

## Conclusion

PGCS decomposition and all downstream analyses (suffix families, EC/FC classification,
nomenclator assignments) are robust to transcription variant choice within the ZLZI
consensus framework.
