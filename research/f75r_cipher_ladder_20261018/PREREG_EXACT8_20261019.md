# F75R exact partial-substitution ladder — registered scope (2026-10-19)

Status: GPU execution ongoing. This file registers interpretation constraints **before inspecting full K8 result**. R1–R4 retractions from `RUNNING_RESULTS.md` remain authoritative.

## Question
Does progressive, explicit enumeration of partial one-to-one glyph → letter assignments produce candidate *words* whose matches show language-specific coherence on genuinely unseen types, rather than merely optimising a whole-page character language-model score? This is the user's proposed approach and is **not** the earlier annealing/global monoalphabetic experiment.

## Frozen input and alphabet
- TTLI f75r, all running-text lines, 352 word tokens. EVA compound units `ch, sh, ckh, cth, cph, cfh, ikh, ith, iph, ifh` treated as atomic. Target transcription pinned by SHA in job logs.
- Folio f76r reserved for evaluating unseen **word types**, not just repeated instances of known types. K8 covers 42 unique train word types/129 occurrences and 25 novel f76r types/40 occurrences.
- Whole-word boundaries assumed correct for this finite model; no unknown segmentation, abbreviations, null glyphs, homophony or source mixing as *generating* mechanisms.
- K5: `ch,d,e,sh,y`, 26P5 = 7,893,600. K6: `ch,d,e,k,sh,y`, 26P6 = 165,765,600. K7: `d,e,k,o,q,sh,y`, 26P7 = 3,315,312,000. K8: `d,e,k,l,o,q,sh,y`, 26P8 = 62,990,928,000.
- Glyph subsets selected to maximise number/length of fully decoded actual word occurrences without using any outside dictionary or interpreting candidates. K6 is not a strict subset of K7; these coverage-optimal rungs are separate exhaustive searches, not a single nested path.
- All 26 injective letter assignments enumerated by 26-ary integer ranges and collision rejection; shard prefixes are disjoint and exhaust the full K space.

## Lexicon and scoring
- Available `wordfreq` 3.x top 25k forms in 18 languages (it, de, cs, fr, es, pl, nl, ro, sk, sl, hu, el, ru, uk, pt, ca, en, he), plus train-only UD Latin Perseus and UD Yiddish YiTB. Normalize with `Unidecode` and ASCII a-z. These are **not 15th-century matched witnesses**, particularly Hebrew/Yiddish transliterations and modern borrowing.
- A candidate earns a hit if every glyph in an observed word has been assigned and resulting string is in the **union of languages**, not necessarily one coherent language. This permissive criterion inflates apparent matches, especially consonantal transliterations and 3/4-letter forms.
- Score each distinct word type once with capped occurrence weight `min(3, sqrt(count)) * (1+.16*(glyph_len-3))`; add 25% for 4+ glyph words. Words below 3 glyphs do not contribute. Full code and logs govern exact computation.
- Fixed 2,600 candidate keys retained at K8 (200 per 13 shards) for subsequent beam extension, which is **not exhaustive**.
- K9 onward is pruned beam search, not a combinatorially complete exploration.

## Required tests and limitations
- Exhaustive K8 control 1: per-type glyph-order shuffle seed 119. Control 2: seed 409. These preserve glyph multiset per word, source token lengths, word-type recurrence, and all words' counts but do *not* preserve positional graphotactics or word morphology, so positive target-control separation alone is weak evidence.
- K5 exact tests include real and those two controls, but K8 selection is much larger and must have K8-matched controls; do not conflate null distributions between rungs.
- **No statistical p-value or language identification is licensed by just two matched controls**. They cannot establish an empirical 5% false-positive rate.
- For any shortlisted key, score exact word-form hits on the 25 genuinely **novel** f76r word types separately, alongside language-provenance counts and repeated types. A heldout match to a word also present on f75r is not independent verification.
- Require a single coherent language with contextual morphology and independently appropriate 15th-century witnesses before even discussing a decipherment. No language-model compatibility or union dictionary match is semantic verification. Positive cipher recovery and hard-negative calibration are required for promotion and **have not been carried out in this exact word-first instrument**.
- K9 full exhaustion costs ~1.13 trillion injective mappings (~18x K8), beyond this compute envelope. No claim of having exhaustively searched 20-glyph permutation space.

## Cost and persistence
- User funded ~$20 HF credit. Cost of these GPU jobs must be estimated from actual terminal timestamps and published T4 unit rate, not asserted as billed balance.
- Source code `gpu_exact_partial_enum.py`, K6/K7 audit JSON, K8 per-shard terminal JSON and merged manifest in `results/`, downstream pruned beam results.
- No deletes or changes to existing Supabase, handoffs, or main branch; this is an isolated experiment branch.
