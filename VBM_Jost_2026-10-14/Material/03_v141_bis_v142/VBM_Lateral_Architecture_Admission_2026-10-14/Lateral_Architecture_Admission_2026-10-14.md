# VBM lateral architecture admission (14 October 2026)

## RETRACTED OR NARROWED FINDINGS

- **Narrowed from v141:** The 722-to-443 pigeonhole argument *only* excludes one globally injective cell per complete consonant cluster. It does **not** prohibit a compositional or contextually decoded VBM implementation. The v141 NO-GO remains correct for its frozen scope.
- **No inference from ambiguous segmentation alone:** The input string `bc` may be parsed into `bc` or `b` + `c`. Both decode to the same consonant characters. This is **not** by itself an ambiguity in the recovered plaintext. What remains untested is whether encoded cell strings can be uniquely parsed and vowel/word boundaries recovered.
- **No mechanism claim:** Source-corpus capacity and Hartlieb coverage are engineering prerequisites, not proof of Voynich cryptography or historical scribe feasibility. No new decipherment or manuscript-level prediction is admitted.

## Scope and competing mechanisms

A. Original one-global-cell-per-consonant-cluster v138 design: capacity insufficient and bridge fallbacks already fail.

B. New compositional architecture: chunks of length at most two, with incoming-vowel-conditioned tables and singles as an open-class fallback. Source clusters concatenate into one decoded consonant run. The required vowel state is **not allowed as an external plaintext side channel**: a future full VBM implementation must derive it unambiguously from previous glyphs and bridge pairs, including initial state and EOF.

C. Autonomous section-conditioned slot generator: can reproduce many of the same output statistics without underlying German; all historical claims must be discriminated against this and equally flexible controls.

## Verified deterministic feasibility counts

Source: supplied `bairisch_kurzertext_GROSS.txt`, normalised and eventised according to v138 (independently reconstructed from the raw source; equality checked against the original events). The supplied VMS-B cell inventory and bridge map are unchanged. Counts are deterministic enumerations; null standard deviation does not apply.

- Complete cluster inventory: 722 distinct types, 91,234 nonempty cluster events, 443 available distinct VMS-B cells.
- Max-one-char chunks: 18 types, 153,106 units, **+67.82%** relative to 91,234 events.
- Max-two-char chunks: **271 types**, 97,849 units, **+6,615 (+7.25%)** event units.
- Max-three-char chunks: **585 types**, 92,329 units, **+1.20%**. This exceeds 443 overall cells and also the vowel-specific inventories in several states.
- Max-four-char chunks: 692 types and 91,384 units, exceeds inventory.
- Max-five-char chunks: 720 types and 91,272 units, exceeds inventory.

### State-specific alphabet sizes vs available key-compatible cells

| Previous decoded vowel | Required unique short-unit forms including 18 singletons | Distinct cells with compatible incoming bridge | Capacity margin |
|---|---:|---:|---:|
| a | 232 | 357 | +125 |
| e | 231 | 443 | +212 |
| i | 222 | 236 | +14 |
| o | 186 | 236 | +50 |
| u | 185 | 259 | +74 |

This proves only that **independent injective assignments could exist within every vowel state**, given the supplied cells and observed training primitives. The total of the five required state alphabets is **1,056** entries if implemented independently. A smaller common ordering/derived rule is a design hypothesis, not yet a demonstrated compression of the tables.

### Frozen vocabulary on supplied Hartlieb text

The training per-state dictionaries and all 18 singleton letters were defined using the original source; Hartlieb was not added to the vocabulary. The Hartlieb transcription has 18,070 source events, of which 16,925 have nonempty consonant clusters. Max-two-char chopping yields 18,373 chunks. **56 chunks (0.33% of nonempty source events) required splitting an unrepresented pair into two single consonants**, yielding 18,429 symbolic units. No previously unseen consonant letter appears. This is an *internal author-supplied cross-text check*, not an external independent corpus nor a Voynich hypothesis test; the Hartlieb file is acknowledged to have transcription errors.

### Controls and failures still open

- Positive: independently reconstructed the source's event list and verified exact agreement; verified no same-state cell collisions in a constructive provisional mapping.
- Hard negative: forced a same-vowel cell collision, and the injectivity condition rejected it.
- Not performed: actual glyph emission, seam/trigram admissibility, initial-state signalling, vowel-bridge parsing, word-boundary reconstruction, independent full-text decoder, truly external-text transfer, or matched within-section VMS/slot-model mechanism comparison.
- Normalisation losses remain intrinsic; not all original spellings can be reconstructed. Word-boundary information remains unencoded.

## Falsification-first next-stage specification (NOT RUN)

1. Build a **joint encoder/independent decoder specification**, not another frequency fit. Encode an initial vowel/state and cluster ends via observable ciphertext rules; require codeword parsing without sidecars, and preserve exact normalised character stream and explicit boundaries. No silent exceptions.
2. Precommit a table-complexity ceiling in advance, expressed as manually recordable entries plus rule description length. Compare against the original four-table burden and similarly complex slot models.
3. On separate development simulations, use hard controls: a feasible bridge-state round-trip positive; an almost-identical mismatched-bridge or cell-collision negative; an unmarked-boundary negative; unseen digraph transfer; and a wrong-initial-state failure. Require all relevant controls to discriminate.
4. If fully reversible at acceptable complexity, freeze tables before the real manuscript comparison. Assess within-section, page/quire blocked holdouts and fair grammar/Markov controls. No section mixing, no tuning to validation pages, and no mechanism claim without a distinguishing external anchor.
5. If a same-complexity autonomous slot model can match all specified observables, report *observational non-identifiability* and halt the historical inference, even if the cipher works in synthetic experiments.

**Decision:** GO for architectural *exploration* of max-two-char composition with a stateful bridge; **NO-GO for claiming a qualified v141 cipher or any VBM evidence from Voynich**. Strongest open risk is the encoding seam/boundary grammar; `i`-state capacity has only 14 cells of slack on the training representation.

## Reproduction

Run `python vbm_lateral_feasibility.py` beside the existing `Generator_v138_Pruefpaket_2026-10-14.zip` and the extracted prior `vbm_v141_gate/source/gen138`. Re-run against the original author package, preserving its hashes and never modifying the archive. Atomic stage files are `stage0_source.pkl`, `stage1_training_capacity.pkl`, `stage2_holdout_coverage.pkl` and `stage3_controls.pkl`. Full counts in `results.json`.
