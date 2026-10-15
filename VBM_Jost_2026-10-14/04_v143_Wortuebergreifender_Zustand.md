# VBM v143: cross-word carry from v142b — independent admission

Date: 14 October 2026  
Status: **SYNTHETIC STATE-TRANSFER ENGINEERING PASS; VMS SURFACE NO-GO; HISTORICAL MECHANISM NOT IDENTIFIED.**

## RETRACTED OR NARROWED FINDINGS (keep at top)

1. **The earlier cross-space test was not a test of our v142 architecture.** It used JoJo's original v138 key labels. Its corrected −0.0067-bit matched-null effect (SD 0.0055) remains a negative/undetermined result for *that instrument*, not for v142 or v143.
2. **This v143 prototype is not JoJo's original crossing-space VBM.** It retains v142b's within-word vowel bridges and ordinary plaintext-aligned word spaces. It adds a *cross-word codebook permutation* controlled by the preceding decoded word. No inference about the original author's historical key system follows.
3. **Correction during this stage:** Although the three arms share the same codecell strings, first letters and codecell lengths, a wrong-mode decoder can reject ciphertext because it reads the reserved epsilon cell in the wrong syntactic position. Therefore their **complete formal languages are not necessarily identical**. The early expectation that every wrong decoder would accept every sequence was false; this was corrected before reporting.
4. **No overturning of older failures:** v138 illegal seams/collisions, v141 framing overhead, v142 ~88–90% invalid trigrams, V10 synthetic key identification failure and the qualified negative cross-space bridge-label results remain in force. 
5. **Synthetic detection ≠ historical inference.** The very large paired-source effect below is engineered by construction, then recovered by an independent instrument. It validates the instrument and mechanism; it is not evidence of plaintext in Voynich.

## One-question design

Can v142b's normalised-German code be made **sequentially state-dependent across ciphertext word boundaries**, without adding an encoder-only ledger, control tokens, codebook entries, extra characters or changing the original allowed bridge/next-cell pairs? And can a frozen paired-source instrument discriminate that mechanism from a similarly resourced state-free construction on held-out Hartlieb?

### Frozen hypotheses and identifiability gate

- **A, original v142b:** An independent prefix-free vowel-dependent code for each plaintext word; no information from the previous word is needed.
- **B, v143 carry:** For word `i`, select one of six *bijective* within-(vowel-state, initial-key, glyph-length)-class codecell permutations according to the last vowel of the **preceding decoded plaintext word**; use `-` if absent. No standalone transmission of the state is necessary: the decoder recovers it from word `i-1`.
- **C, matched state-free control:** Exactly the same codecells, permutations, length and delimiter policy. Select the same six options using a publicly reproducible word-index hash; its six-choice probability table is fitted **once on development text only**, not Hartlieb.

The prior v142b manifest and this contract were sealed before emission. The permutation is rotation by the fixed state index in each codecell group. That is a bijection: **glyph length and the initial vowel-bridge key cannot change**, and the parser retains prefix-free segmentation. The 222 source units, nine original bridge pairs and 443 native cells remain exactly as before. Within-word vowel bridges are *not* moved across spaces.

**Identifiability decision:** Known synthetic plaintext/ciphertext pairs can differentiate B from C; unlabeled VMS ciphertext generally cannot distinguish a hidden plaintext generator from an autonomous generator implementing the same observable law. A frozen independent historical codebook or plaintext anchor would be needed for such a claim. We therefore did not run an attribution experiment on VMS sections. We ran only the inherited necessary surface diagnostic.

## Implementation and blind transfer

Two distinct implementations, `encode_v143.py` and `decode_v143_independent.py`, separately construct the permutation/inverse from the published manifest. The decoder obtains the next carry from **its own preceding decoded word**; it does not read source text, prior carry labels, encoder events or external logs. The state-free decoder uses only the public deterministic schedule.

| Metric | Bavarian development | Hartlieb transfer |
|---|---:|---:|
| Source words | 68,396 | 13,232 |
| v142b exact recovered words | 68,396 | 13,232 |
| v143 carry exact recovered words | 68,396 | 13,232 |
| State-free exact recovered words | 68,396 | 13,232 |
| Ciphertext word count and *individual word lengths* across all arms | Identical | Identical |
| Nontrivially remapped carry cells / all cells | 152,546 / 156,511 | 29,117 / 29,734 |
| Nontrivially remapped state-free cells / all cells | 152,265 / 156,511 | 28,788 / 29,734 |

These are deterministic exact checks: a stochastic null SD does not apply. Hartlieb was withheld from codebook construction, group choices and state-free schedule weighting, but it is still an author-supplied German text and not an external historical anchor.

## Paired-source falsification instrument

Only word instances whose plaintext source word recurs and whose first eligible codecell belongs to a rotation group of at least six cells are scored. This lets a *separately written parser* infer the selected permutation directly from a source/ciphertext alignment, with no encoder event trace. Frozen test: compare the inferred state to the last vowel of the preceding known source word, against 300 shuffles of inferred states **within exact current plaintext word types**. Those permutations preserve the current word, its source-unit composition, and the complete inferred-state marginal for that type.

| Hartlieb paired test | Effect over within-word conditional null | Null SD | z | Eligible word instances |
|---|---:|---:|---:|---:|
| B, real carry | **+0.58354 accuracy** | **0.004093** | 142.58 | 10,652 |
| C, word-index state-free | +0.004134 accuracy | 0.003380 | 1.22 | 10,652 |

For C, **the metric does not resolve this**: its ratio is below two. B's enormous effect is expected *by design*. This is a useful positive synthetic control, not a discovery about historical VMS. The conditional null controls the exact current source word but cannot turn an intentionally encoded dependency into independent historical evidence.

The state-free schedule is only *approximately* distribution matched on Hartlieb: Hartlieb has a different last-vowel distribution from development text (e.g., carry `-` occurs 5 times versus 138 selections by the state-free schedule). Hence simple between-arm vocabulary or n-gram comparisons are not qualified as effects of carry. The within-source-word conditional null is the appropriate primary instrument.

## Additional adversarial checks

- Intact B, C and A round-trip exactly with the independent decoder.
- A forced same-state code collision is rejected, as is an invalid leading glyph.
- Decoding B output under the A or C decoding convention does **not** generally work: the parser rejects an epsilon marker appearing outside a valid vowel context. This is a genuine observable signature **conditional on already possessing our frozen codebook/grammar**; it does not constitute an independent historical codebook.
- Changing the code state without changing the admissible glyph set may produce an alternative, wrong plaintext or a parser error. The scheme is not authenticated and this is not a proof of robust recovery under copying errors.

## Surface constraint and diversity bound

The inherited original v142b problem is still present. The source comparison here is the **full author-supplied Currier B REVA reference**, from which the cell inventory was learned. The raw-character trigram screen is a *necessary in-sample-support bound*, not a calibrated test of historical mechanism. Shuffling intact VMS tokens makes its within-token support unchanged (null SD exactly 0), so no usable z or historical p-value is available.

| Raw-character trigram support violation | v142b | v143 carry | State-free |
|---|---:|---:|---:|
| Bavarian, words violating | 60,454 / 68,396 (88.39%) | 61,205 / 68,396 (89.49%) | 61,554 / 68,396 (90.00%) |
| Hartlieb, words violating | 11,745 / 13,232 (88.76%) | 11,810 / 13,232 (89.25%) | 11,911 / 13,232 (90.02%) |

The development v143 carry versus v142b change is **+1.10 percentage points**; the comparable matched historical null SD is **not available**, and the reference's own token-shuffle null SD is zero. The gate still fails. No parameter was changed to improve these results.

Vocabulary diversity also moves away from the inherited model: on Hartlieb, v142b produced 2,276 distinct printed forms and v143 carry 3,461 (an increase of **1,185 forms**, +52.1%, deterministic, null SD not applicable). The word count, characters and per-position lengths are identical. The state-free arm produced 3,847 forms, but its holdout state distribution is imperfectly matched, so this is descriptive, not a causal comparison.

## Independent verification questions

1. **Does this genuinely transfer state *between* words?** Yes: the next permutation is determined from the preceding **decoded plaintext** word, not the source provided to the decoder. No external carry stream is transmitted.
2. **Could the result be driven by word lengths, additional glyphs or vocabulary fitting on Hartlieb?** No: word lengths and cell inventory are invariant by construction, the codebook is frozen and Hartlieb is not fitted. The next-word permutation itself changes the realised glyph sequence.
3. **Does the matched null detect an association introduced by B but not C?** Yes for the paired synthetic source; +0.58354 accuracy (SD 0.004093) for B, whereas C is unresolved at +0.004134 (SD 0.003380). No historical inference.
4. **Does this make Voynich text generable?** No. The inherited surface gate remains failed, and the new sequence also fails. Its prior failure has not been erased by changing the mechanism.
5. **Could an autonomous model produce the same surface data without an externally supplied German source?** Yes, by matching or implementing the ciphertext-output stochastic law. A fixed v143 grammar does define a restricted parser, but without externally grounded cell semantics that does not establish a plaintext interpretation.

## Audit trail in mandated order

1. **Circularity:** Cell inventory was fitted on the complete supplied B transcription. Flagged; no positive VMS model test.
2. **Leakage:** Hartlieb not used in table formation or control weights. Pass for synthetic holdout; it is author supplied.
3. **Confounds:** Carry last-vowel distribution and source language are correlated. The per-exact-current-word permutation corrects current-word confounding, not all source sequencing or interpretation issues.
4. **Matched nulls:** Exact-current-word permutation, 300 precommitted draws, repeat-only eligibility. Pass for paired synthetic association; no VMS matched autonomous alternative has qualified.
5. **Control fairness:** C shares codecells, bridge vocabulary, word lengths and deterministic decoding but Hartlieb state marginals differ; no claims from naive between-arm differences.
6. **Measurement degeneracy:** Reference-inventory support null is exactly zero and token-order insensitive. Never count it as identification.
7. **Representation dependence:** Carry uses preceding plaintext word's last normalised vowel, not a discovered glyph property. This requires knowing the codebook and decoder.
8. **Decision fragility:** Large paired effect is deliberately planted, not independent evidence; preserving frozen stop criteria forces VMS NO-GO.
9. **Audit completeness:** All manifests, source hashes, code, emitted ciphertexts, original v142 comparison data, JSON results, reference input and atomic pickle checkpoints are provided.

## Decision and next legitimate step

**v143 B is a qualified *synthetic* cross-word state-transfer cipher.** It is a concrete extension of v142, not a reprise of JoJo v138. It can be decoded, and a source-aware hard null separates it from a matched word-index state-free control.

**No VMS mechanism promotion.** The new variant fails the necessary manuscript surface condition, and the only discriminating success requires a known source/ciphertext pairing. Do not tune this program and do not begin a fresh, open-ended SELECT programme on its strength. An admissible next scientific step would require an independently grounded relationship between observed glyph families and predicted carry states, together with a capacity-matched autonomous negative that could genuinely falsify the claim.