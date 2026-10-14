# VBM v142 admission audit — in-word coding and original bridge semantics

Date: 14 October 2026. Status: **MECHANICAL PASS / CODEBOOK COMPACT PASS / VMS SURFACE NO-GO / HISTORICAL VBM NOT IDENTIFIED.**

## RETRACTED OR NARROWED FINDINGS — retained prominently

1. **Narrowed the earlier capacity obstruction.** The impossibility of assigning 722 complete consonant clusters injectively to 443 cells remains correct **only for v138's global one-cell-per-whole-cluster model**. It is not a general impossibility for VBM-like stateful encodings.
2. **Corrected the v141 bridge interpretation.** The original bridge-pair vocabulary does not, by itself, prevent an internally parseable one-word output. This experiment constructs an exact decoder using **nine actual bridge/next-key pairs** from the author's 32-pair lookup, under a deliberately revised word-internal placement grammar. The earlier concern that 150 `y` cells cannot cover all 222 possible source units applies only to a *single-cell, universal-in-every-state* assignment. One escape cell plus an additional cell for rare units removes that restriction.
3. **v142a is a *relaxed control*, not proof of original VBM.** The simpler first branch used dedicated internal glyphs for vowels; v142b restores `(bridge, following cell key) -> vowel` semantics. Neither branch reproduces v138's cross-token layout.
4. **No promotion from exact decode to surface or history.** Both branches decode German exactly, but both fail frozen atomic-seam admission. No inference of medieval German plaintext, manuscript provenance, or actual Voynich encipherment is admissible.
5. **Earlier failures retained.** The v138 inventory collisions, bridge mismatches, invalid glyph seams; our v141 framed-token surface failure; the prior Q65/V10 negative identification and key-recovery results all remain unrescinded.

## Identifiability-first preflight

Competing mechanisms:

- **P:** a deterministic plaintext-driven state-dependent, compositional VBM transform, with vowels represented by a bridge plus the initial key of the next code cell.
- **S:** an autonomous, section-conditioned slot/Markov generator producing similar surface patterns but no recoverable German plaintext.
- **R:** an unconstrained in-word code with explicit vowel glyph markers, as a positive engineering control demonstrating that word delimitation and prefix-free parsing are possible without dedicated marker *tokens*.

**Distinguishing observable:** Intact synthetic source/cipher pairs with frozen tables and independent decoding distinguish a viable encoder from a faulty encoder; they **do not** distinguish P from S for a single unlabelled historical manuscript. Under a sufficiently flexible S, output distributions are observationally equivalent. The required independent plaintext, bilingual crib, recoverable historical table or external linguistic anchor is absent. Consequently there is **no historical-mechanism experiment** in this stage, and no VMS-section pooled model-selection inference.

A manuscript glyph seam is only a *necessary* surface property. A negative control formed by shuffling intact VMS tokens inside each line has exactly the same within-token bigrams and trigrams; its null standard deviation for these violations is mathematically **0** and the surface gate will incorrectly accept its destroyed word order. Thus the seam gate cannot identify a mechanism, even if passed.

## Frozen contracts and source boundary

The earlier `/00_admission_contract.json` and the narrower `/03_pair_bridge_contract.json` were recorded before their respective emissions. Original author package `Generator_v138_Pruefpaket_2026-10-14.zip` is unchanged. Normalisation is the author's v138 normalization, with the documented `§ -> und` macro expansion; the recoverable object is **normalised plaintext**, not original Middle High German orthography or punctuation. Development = supplied Bavarian source; transfer = author-supplied Hartlieb. The latter is held out from table formation but is not an externally authenticated language sample.

The codebook derives solely from existing author-fitted B cells; thus any positive appearance of native glyph properties would be **in-sample by construction** and insufficient as a held-out observation. The tests use only this supplied Currier B transcription. No comparisons are pooled across Voynich sections to infer a common mechanism.

## Construction, in brief

Both branches use short, shared consonant units, a vowel-dependent table of native `key+state+head` cells, a prefix-free code within each vowel state and **one printed ciphertext word per normalised German word**. Ordinary whitespace acts as the observed word delimiter. One-to-one source/Voynich word-boundary alignment is an **assumption being tested synthetically**, not an established manuscript fact.

- **v142a:** 221 shared consonant units, five dedicated internal vowel-glyph controls and shortest-first prefix-free cells. It is a relaxed reversible engineering control, not the author's vowel-bridge scheme.
- **v142b:** 222 shared source units (including epsilon), real original pairs: `sh|y`/`o|n` for `a`, `d|y`/`ch|n`/`a|r` for `e`, `o|y` for `i`, `p|y` for `o`, `l|y`/`ch|r` for `u`. All nine pair values were verified against the frozen original lookup. All consonant-cell key initials are `y,n,r`, while bridge initials are different, so the parser can distinguish markers and cells. Within the vowel state the cells form a prefix-free code. For `i` and `o`, their 146 prefix-free `y` cells encode 145 frequent source units directly and 77 rarer units via a reserved escape cell plus one more cell. An epsilon cell carries vowels without following consonants. Original bridges are **inside** the word, unlike the author's cross-word scheme.

The compact v142b manifest explicitly records **443 input cells + 222 shared source units + 9 distinct bridge pairs = 674 logical entries** (the JSON also duplicates those nine bridge pairs in the forward/reverse views). A separate decoder *re-derives* all five state tables and reproduces the 958 materialised state-cell entries exactly; no such expanded tables are required in the compact manifest. It fits the prior 697-entry engineering budget as a logical table representation, but the longer parsing procedure and historical copying burden remain unmeasured. The expanded manifest remains in the package for comparison.

## Independent test outcomes

| Gate | v142a internal-vowel control | v142b actual bridge pairs |
|---|---:|---:|
| Bavarian recovered with exact word boundaries | 68,396 / 68,396 | 68,396 / 68,396 |
| Hartlieb recovered with exact word boundaries | 13,232 / 13,232 | 13,232 / 13,232 |
| Separate decoder needs only ciphertext and manifest | Pass | Pass; also pass with compact-derived manifest |
| Printed ciphertext tokens per source word | Exactly 1 | Exactly 1 |
| Mean output word length, characters | 6.299 | 7.106 |
| Development words violating reference atom bigram support | 24,114 (35.26%) | 21,052 (30.78%) |
| Development words violating reference atom trigram support | 60,897 (89.04%) | 61,541 (89.98%) |
| Transfer words violating reference atom bigram support | 4,377 (33.08%) | 3,519 (26.59%) |
| Transfer words violating reference atom trigram support | 11,536 (87.18%) | 12,027 (90.89%) |
| Atomic seam admission (zero unknown B-reference bigrams/trigrams) | **FAIL** | **FAIL** |
| Mechanism identification | NOT TESTED | NOT TESTED |

The Currier B reference supplies **21,679 words** and has zero unobserved within-word atomic bigrams or trigrams relative to its own full inventory *by definition*. For the final pair-preserving branch, **61,541/68,396 generated words (89.98%) violate the trigram constraint**, an **+89.98-percentage-point** difference against the reference, with within-line token-permutation null **SD = 0**. The standardized ratio cannot be meaningfully estimated because this null is degenerate. This is a direct, non-inferential failure of the *frozen surface admission rule*, not proof that all reversible VBM systems are impossible.

**Independent representation bound:** Repeating the constraint with **raw character** trigrams rather than the author’s REVA atoms gives **60,454/68,396 (88.39%)** violating words for the pair-preserving branch, against **0%** for the source reference, a **+88.39-percentage-point** effect with a token-permutation null **SD = 0**. The severe failure therefore survives changing the unit of measurement. Raw character bigrams independently fail in **17,386/68,396 (25.42%)** development words versus **0%** reference, an effect of **+25.42 percentage points** and null **SD = 0**. This is still a self-reference support diagnostic, not calibrated inference about independent manuscripts. See `07_raw_character_bound.json` and `stage9_raw_character_bound.pkl`.

Average pair-branch output word length **7.106** vs the reference **5.061**, a **+2.045 characters/word** descriptive difference; a section/length-matched null SD was *not established*, so this length difference is not qualified as a manuscript-level finding. Source normalization and representation differences further limit direct comparison. These figures are not used to select or revise the encoder after seeing results.

## Positive and adversarial controls

- Positive: Both separate decoders reconstruct each source byte-for-byte after normalization, from manifest and ciphertext only. Compact-derived decoder reproduces every materialised state table exactly. v142a also decodes a composed synthetic test word not present in the data.
- Hard negatives: forcing a same-vowel code collision, making a prefix-overlapping entry, tampering with the escape-cell specification, or adding invalid glyphs produces a parser error. An illegal bridge/next-key pair is rejected.
- Valid mutations: changing a legal bridge to another legal bridge can silently change the decoded plaintext. Removing an ordinary word space may likewise merge words. This is normal un-authenticated cipher malleability; exact intact round trips do **not** demonstrate tamper detection or ciphertext-only key recovery.
- Fairness: An order-shuffled VMS corpus keeps every token valid and automatically passes the fixed atomic seam gate; it is a hard negative for attributive interpretation. No claim is made of a fair, capacity-matched likelihood competition against an autonomous slot generator because no independent discriminating anchor exists.

## Audit checklist (in order)

1. **Circularity:** B inventory previously learnt from all supplied B transcription; permits only necessary surface rejection, not positive evidence. PASS flagged.
2. **Leakage:** Hartlieb never used in codebook construction; author's word normalizer shared openly. PASS for synthetic transfer, source independence limited.
3. **Confounds:** German source word alignment assumed, normalization reduces spelling and boundaries; no historical manuscript anchor. FLAGGED.
4. **Matched nulls:** within-line token-order negative retains atom seams exactly; SD=0; cannot qualify mechanism. FLAGGED.
5. **Control fairness:** parser adversarial controls qualify engineering only, not historical discrimination. FLAGGED.
6. **Measurement degeneracy:** trigram admission rejects these implementations but cannot reject VMS-order shuffles. FLAGGED.
7. **Representation dependence:** reverse-EVA (REVA) atoms, vowel states, consonant chunking, and short word assumption are substantive. FLAGGED.
8. **Decision-rule fragility:** zero-unknown-n-gram gate is a strict *support condition*, not a calibrated generalisation threshold for actual novel manuscript folios. FLAGGED.
9. **Audit completeness:** source, compact manifest, output streams, independent parsers, hard negatives, numerical JSON and stage pickle checkpoints included. PASS.

## Final decision

**PASS: in-word, prefix-free, synthetically reversible compositional cipher, including an original-pair-vowel branch at ≤697 logical mapping entries.**

**NO-GO: current v142 in-word grammar as a Voynich surface generator.** It violates the frozen B atomic-seam condition on almost nine-tenths of output words. No adjustments were made after observing that failure.

**NOT ADMITTED: historical VBM mechanism, plaintext recovery, five-vowel language claims, scribal practicality.** The old v138 and v141 failures remain in force. The next legitimate research move is not another encoder variant: it is to obtain an independent discriminating anchor or a frozen, section-specific prediction that a capacity-matched autonomous generator demonstrably cannot reproduce. Until such evidence exists, more surface fitting would be an engineering pastime, not a mechanism test.