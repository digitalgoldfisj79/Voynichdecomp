# VBM v141 lateral architecture: frozen compositional codec, independent recovery and surface gate

**Date:** 14 October 2026  
**Status:** **ENGINEERING ROUND-TRIP PASS; VMS SURFACE NO-GO; MECHANISM UNIDENTIFIED.**  
**Source:** The user's `Generator_v138_Pruefpaket_2026-10-14.zip`, SHA-256 `e67fe89555f8d8b2e74218077bb186f4f6fec4e5444f0ee9c289750d1efd4033`. The original ZIP was not modified.

## RETRACTED FINDINGS / CORRECTIONS (retain at file top)

1. **Not promoted:** The previous v141 NO-GO remains valid **for the original one-global-cell-per-cluster mechanism**. A compositional codec does not rescue the claim that the frozen v138 table is bijective.
2. **Narrowed:** The earlier lateral result of **271 source primitives** refers to the v138 cross-word vowel-event segmentation. This working design segments individual *normalised words*, giving **222 shared units** (including the empty unit). These inventories use different boundaries and must not be equated.
3. **Leakage corrected before the final run:** An initial implementation mistakenly collected singleton consonants from both the development and Hartlieb texts. The final rebuild computes *every* primitive from the development text only; Hartlieb remains unseen when the frozen manifest is made. The two sources happen to share the same 18 consonant letters, so the correction did not change the success count.
4. **Important downgrade:** Full decoding is now proven for a *framed, artificial alphabetic stream*. It is **not** proof of a plausible Voynich running-text cipher. Indeed, the latter currently fails its separate surface gate.
5. **Previous failures retained:** The v138 collisions, bridge mismatches and sequence violations, plus all prior cipher-identifiability failures, remain in the record. No historical VBM confidence increase is admitted from this run.

## Research question, before measurement

- **A: original VBM**: one global Voynich cell per complete German consonant cluster; blocked by the frozen inventory and bridge constraints.
- **B: lateral VBM**: compositional one- or two-letter consonant units and a source-vowel-conditioned cell mapping, encoded with an independently decodable bridge/word grammar.
- **C: nonplaintext autonomous slot generator**: may match Voynich surface observables without underlying German.

**Identifiability precondition:** B and C can both yield Voynich-like surface sequences. An engineering test of B **cannot** decide B vs C. Historical attribution would require a frozen *differential prediction* with independent linguistic/manuscript anchors, positive and fair matched negative controls, and within-section page/quire holdouts. No such identifiable target was available for this engineering stage, so **no Voynich-section historical inference was run**.

## Precommitted contract

Recorded in `00_frozen_contract.json` and `stage0_contract_provenance.pkl`: at most 235 shared units, 443 cells, 40 bridge pairs, 12 explicit grammar rules; exact full-word decoding from ciphertext and frozen manifest alone on development and Hartlieb; deliberately broken controls must produce parser failure or observable expected-plaintext disagreement. No silent encoder fallbacks beyond literal splitting of an unrepresented digraph into two universally specified singletons. The manuscript surface is a **separate gate**.

### What was actually built

1. Apply the author's source `norm_word`/`words_of` rules, then canonicalise the author's `§` word macro to `und`. This is the **declared recoverable text**, not the original spelling or punctuation.
2. Split each source word into its initial consonant run and repeated `(vowel, consonant-run)` segments. Cut each run into ordered two-character chunks, falling back to two one-character chunks for unknown digraphs.
3. Assign an ordered, **single shared list of 222 source units** (including epsilon) to the ranked 443 Voynich-derived cells filtered by incoming vowel and allowed initial key. This is one rule plus a shared primitive rank table, **not five independent unit-to-cell tables**. The lowest-capacity vowel states have 236 eligible cells, so the 222-unit vocabulary fits.
4. Emit each consonant chunk as a separate actual Voynich-derived `key+state+head` cell string. An explicit `m+bridge` control token denotes a vowel; its value is determined by `(bridge, next cell's key)` in the author's original vowel pair table. A vowel with no following consonant gets an explicit epsilon cell. A standalone `s` marks the end of every source word. Leading consonants use the public initial state `e`.
5. `independent_decoder.py` uses **only** the resulting stream and `01_frozen_manifest.json`; it never loads the original German, the encoder or its event ledger. It derives the current vowel, inverse unit rank and each boundary from the ciphertext.

**Critical qualification:** `m+bridge` and dedicated `s` controls are experimental framing, not historical Voynich syntax. The fact that many cell spellings were selected from Currier B also **leaks the target inventory**, so in-sample glyph attestation is not held-out evidence.

## Reproduction and completeness

| Precommitted check | Measured result | Decision |
|---|---|---|
| Development recovery | 68,396 of 68,396 normalised words, exact order and boundaries | PASS |
| Author-supplied Hartlieb transfer | 13,232 of 13,232 words, byte-for-byte equality | PASS |
| Independent decoder | Separate source implementation; requires manifest and ciphertext only | PASS |
| Five sequential Hartlieb partitions | Five out of five exact concatenated recoveries | PASS (engineering only) |
| Vocabulary budget | 222 <= 235 | PASS |
| Cell inventory budget | 443 <= 443 | PASS |
| Vowel pair budget | 32 <= 40 | PASS |
| Rule budget | 12 <= 12 | PASS |
| Hartlieb digraph fallback | 12 unrepresented pair occurrences out of 7,724 source pair chunks; 12 additional singleton cells; zero unknown letters | PASS |
| Negative: duplicate source primitive in manifest | Decoder rejects | PASS |
| Negative: impossible vowel/next-key bridge | Decoder rejects | PASS |
| Negative: truncated final word delimiter | Decoder rejects at EOF | PASS |
| Negative: wrong initial state | Rejects or changes reference recovery; here rejects | PASS |
| Negative: delete **interior** word delimiter | **Parser accepts** changed text; external reference detects word-boundary loss | **UNDETECTED CORRUPTION**, not exact original recovery |
| Negative: replace bridge by another **valid** bridge | Parser accepts a different plaintext | **MALLEABLE**, no intrinsic corruption detector |

**Interpretation:** Exact reversal of an intact encoded message is demonstrated. No claim of error correction, tamper detection, or recovery of the unknown historical codebook is demonstrated. Wrong-plaintext attacks are expected for a basic substitution cipher and should not be confused with defective inverse code construction.

## Cost of the solution: the surface fails

| Measure | v141 development | v141 Hartlieb |
|---|---:|---:|
| Normalised source words | 68,396 | 13,232 |
| Cell tokens | 156,445 | 29,725 |
| Vowel marker tokens | 101,108 | 18,928 |
| Word marker tokens | 68,396 | 13,232 |
| **Total emitted tokens** | **325,949** | **61,885** |
| Different printed token types | 293 | 205 |
| Tokens containing unattested VMS-B atomic bigrams | 16,175 | 2,980 |
| Tokens containing unattested VMS-B atomic trigrams | 0 | 0 |
| Tokens not attested as whole VMS-B tokens | 16,452 | 3,041 |

The v138 reference output has **101,297 tokens** on the same **raw main source**, versus 325,949 for v141: **+224,652 tokens / 3.218x** (a deterministic comparison: no null SD). Caveat: the normalised representations differ because the author emits special `§` words while the new codec expands that macro to `und`; 3,466 of the normalised main-source words are this macro. Therefore the v138 ratio is a **descriptive scale check**, not an exact source-normalisation-matched performance estimate.

The generated ciphertext is **52.0% explicit `m`-prefix vowel markers or exact `s` word markers**, while the supplied REVA Currier B tokens are **2.892%** in those combined categories: **+49.11 percentage points** (descriptive inventory comparison; no section/quire-matched null SD is available). It has nothing like natural VMS spacing: every consonant unit and explicit control occupies a separate token.

Whole-token attestation is not enough in any case: most of the valid cell output deliberately reuses an inventory fitted on the same reference material. The test does establish that even this extremely permissive fit cannot justify the framing markers. Output trigrams being attested is a weak property here because many controls consist of only one or two atoms.

**Result:** The reversible *framed* v141 design fails the Voynich-surface admission gate. There is no claim that its text was historically plausible or that it predicts an independently withheld Voynich section.

### Exact expansion accounting

The development ciphertext contains 156,445 cell tokens + 101,108 vowel controls + 68,396 word controls = **325,949 tokens**. The overhead is therefore directly accounted for by the representational choices; there is no need to invent a stochastic explanation. The **necessary next change**, if any, would be to represent vowel transitions and word boundaries **inside** parseable VMS-like glyph sequences, eliminating the visible controls while retaining a unique inverse parser. That is a *new architecture*, not a fitted parameter adjustment to this failed one.

## Verification questions (and independent answers)

1. **Does decoder success rely on the original German being available during decoding?** No. The decoder imports only standard libraries and processes a separate frozen JSON manifest and ciphertext. The reference texts are used exclusively by the verifier to establish exact recovery.
2. **Was Hartlieb used when building the dictionary?** No in the final build. Training-only consonant letters and digraphs; Hartlieb introduced 12 previously unrepresented two-character chunks which decomposed without new table entries.
3. **Can a corrupted valid glyph sequence silently change the decoded plaintext?** Yes. One valid bridge mutation produces a different plaintext without an error, and removal of an interior boundary can merge words silently. No authentication or corruption-detection capability is claimed.
4. **Has the 443-cell pigeonhole obstruction been defeated in its original scope?** No. The original scope is still impossible: full complete-cluster injectivity. The replacement avoids the pigeonhole by changing the source representation, and uses a compact context-dependent mapping.
5. **Does exact synthetic recovery or 0 unattested trigrams identify the historical Voynich generator?** No. No independent linguistic/archival anchor or qualified matched section-specific statistical discriminator has been supplied. Fitting the same VMS-B cell list creates measurement leakage.

## Final decisions and next defensible step

- **PASS:** a compact, reproducible, state-dependent compositional cipher is mechanically possible for both supplied texts, with exact recovery of its declared normalised plaintext.
- **FAIL:** the explicit-control emission model resembles neither VMS tokenisation nor a credible scribal surface grammar; no promotion to Voynich generative validation.
- **UNQUALIFIED:** historical codebook recovery, blind decipherment, and mechanism attribution.

The appropriate follow-up is a **new preflight for an in-token, uniquely decodable bridge grammar**. Before another implementation, prove the grammar has a potential observable that can distinguish it from a matched autonomous slot transducer. If this is impossible without an external anchor, stop historical inference. Never relabel the current surface failure as progress towards decipherment.

## Files in this package

- `00_frozen_contract.json`, `01_frozen_manifest.json`: preregistered ceiling/grammar and complete decoder key.
- `build_encoder.py`, `independent_decoder.py`, `verify.py`, `extra_checks.py`: independently runnable generation/decoding/verification scripts.
- `train.vbm`, `hartlieb.vbm`; matched `*.reference.txt` and `*.decoded.txt`.
- `02_verified_results.json`, `03_diagnostic_bounds.json`, atomic `stage*.pkl` files.
- `REPRODUCE.md`: exact commands, environment variables, source ZIP requirement and integrity checks.

All manuscript comparisons in this report are **descriptive**, not section-specific evidence for VBM. Every computed contrast above reports its effect size and explicitly indicates where a null SD does not exist or was not estimated.
