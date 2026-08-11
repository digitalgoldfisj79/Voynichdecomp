# Amadi Residual Cipher Programme v1 — Source Ledger

Date: 2026-08-11
Branch: `experiment/amadi-residuals-v1-20260811`
Parent closeout: `experiment/cipher-coverage-closeout-v1-20260809` @ `418da5635ffa2b1e86053dfd49fc1022ba15c297`

## Purpose

Separate what the Scheers/Amadi material actually attests from earlier Voynich-facing interpretations, then define the finite mechanisms that are eligible for a new computational stress-test.

This ledger is deliberately conservative. It does **not** treat old Amadi-session claims about Voynich semantics as established facts.

## Source basis

Primary working source supplied by the research programme: D. P. J. A. Scheers edition/transcription of the Giovanni Battista/Agostino Amadi cipher treatise, catalogued in the March 2026 Amadi running-results files.

The cumulative catalogue reached approximately 522 sections across five volumes and about 250 folios. The sections below are cited by the numbering used in that catalogue.

## Evidence-status vocabulary

- `ATTESTED_EXACT`: source description is sufficiently explicit to reproduce an operational transformation.
- `ATTESTED_PRINCIPLE`: source clearly states the mechanism class, but the surviving description is not yet sufficient for one unique executable algorithm.
- `SOURCE_COMPOSITE`: source explicitly combines two mechanisms.
- `VOYNICH_INFERENCE`: an interpretation proposed in the old Amadi analysis, not something established by Amadi.
- `STRUCTURAL_MISMATCH`: exact historical mechanism is already incompatible with the frozen RF surface unless a new latent representation is introduced.
- `NEEDS_SOURCE_RECONSTRUCTION`: must return to the Scheers text/illustration before coding; no invented completion is permitted.

## Mechanism ledger

### A01 — reduced alphabet / phonetic elimination

Sections: 024–026; 390–392; 454–458.

Status: `ATTESTED_EXACT` for several Italian examples.

Attested facts:

- Amadi gives multiple twelve-character reductions rather than one immutable reduction rule.
- Section 454 explicitly uses the twelve-character alphabet `A C E I L M N O R S T V` and supplies replacement/elimination rules.
- Sections 456–458 integrate a twelve-character reduction with a 12×12 polyalphabetic table and wheels.
- Section 026 adds multiple cipher signs for reduced-alphabet vowels, so reduced-alphabet preprocessing and homophony are both present in the repertoire.

Important correction:

A reduced plaintext alphabet is **not** equivalent to the global 19×19 bijective monoalphabetic family closed by M0. Information is merged or deleted before encryption.

Admissible computational forms:

1. exact twelve-symbol Amadi reductions on Italian controls;
2. `R12H`: an explicitly declared stress-test composition in which the twelve latent reduced values may have multiple observable cipher signs, with one document-global mapping.

The second form is broader than a literal twelve-symbol surface but is historically motivated by section 026. It must qualify from fresh controls before target use.

### A02 — vowel/consonant-dependent rearrangement

Sections: 013; 438–440; 462.

Status: section 013 `ATTESTED_EXACT`; later variants partly `NEEDS_SOURCE_RECONSTRUCTION`.

Attested facts:

- Section 013 removes vowels from their internal positions and appends them at the end of each word in original order, leaving a consonant skeleton followed by the vowels.
- Sections 438–440 contain additional consonant/vowel separation and distribution methods.
- Section 462 states a system in which letters/symbols/numbers replace vowels and the consonants follow.

Important computational point:

Section 013 is lossy with respect to original vowel positions. It therefore must be tested by modelling the *forward-transformed language*, not by pretending a unique plaintext inversion exists.

Admissible v1 exact arm: `VC_END` = consonants in original relative order followed by vowels in original relative order, independently within each pre-existing word, then one document-global monoalphabetic relabelling.

No other vowel/consonant rule may be added until its Scheers wording and worked example are reconstructed exactly.

### A03 — word-reset positional polyalphabetic / positional table selection

Sections: 445; 459; 461; 463; 478.

Status: 445/461 `ATTESTED_EXACT` at the schedule level; 459/463 `ATTESTED_PRINCIPLE` for multiple rings/inventories.

Attested facts:

- Section 445 describes word-by-word polyalphabetic use in which successive character positions use successive alphabets and the schedule restarts for the next word.
- Section 461 explicitly alternates two different tables/alphabet rows by position within a word: first position from table 1/alphabet 1, second from table 2/alphabet 1, third from table 1/alphabet 2, fourth from table 2/alphabet 2, and so on.
- Sections 459 and 463 describe multiple rings with varied/different character inventories.
- Section 478 explicitly acknowledges second, third, fourth, penultimate and final positions as cipher variables.

Important correction:

Section 463 by itself does not prove that every Voynich-like token slot corresponds one-to-one to one ring. Section 461 is the cleaner source for position-dependent source selection. The old claim that the PGCS architecture is therefore 'exactly' Amadi is an inference and is not carried forward as fact.

Admissible broad rejection family: `PWA_K`, a word-reset positional cipher with `K` independent document-global substitution maps, `K ∈ {2,3,4,5}`, position class `j mod K`, preserving word boundaries. This superfamily is intentionally broader than any one Amadi table; a negative against it is conservative. A positive cannot be called an Amadi match until a narrower source-exact schedule also passes.

### A04 — houses / multiple substitution states

Sections: 485–490.

Status: `ATTESTED_EXACT` for existence of 12 houses and several schedules; target selector interpretation is `VOYNICH_INFERENCE`.

Attested facts:

- Section 486 gives a 12-house table with paired alphabets.
- Sections 489–490 give multiple polyalphabetic tables and several ways to move between houses, including line/word progression, explicit key selectors, and changes every four or five characters.

Important correction:

The March 2026 statement `gallows = house selector` was **not confirmed** by Amadi or by the two-crib exercise. It is a finite Voynich hypothesis derived from Amadi and must be tested as such.

Admissible target-specific arm: `GHOUSE5`.

- Frozen observable selectors: EVA/RF gallows classes `{k,t,p,f,NONE}` only.
- Selector extraction rule must be frozen before target scoring.
- Each selector class chooses one document-global substitution map.
- The selector glyph is treated as control information, not silently reinterpreted as plaintext payload.
- A matched selector-label permutation test is mandatory before any positive interpretation.

`GHOUSE5` may test the old hypothesis but cannot by itself establish that Amadi's houses were used in the Voynich Manuscript.

### A05 — plaintext-driven autokey

Section: 490.

Status: `ATTESTED_PRINCIPLE`, `NEEDS_SOURCE_RECONSTRUCTION` before implementation.

Attested wording in the catalogue states that the first alphabet is used for the first word; a plaintext letter then indicates the alphabet for the following word, and a letter of that word points to the alphabet for the third word.

The exact state transition must be reconstructed from Scheers before coding. No generic Vigenère/autokey substitute is permitted. If the source cannot determine one executable transition rule, this arm is classified `UNDERDETERMINED_SOURCE` and does not reach Voynich.

### A06 — NTR/NTRC/DBAC coordinate codes

Sections: 477–484, especially 479–482.

Status: `ATTESTED_EXACT`; direct form `STRUCTURAL_MISMATCH` candidate.

Attested facts:

- all two-, three- and four-character combinations over `{N,T,R,C}` are explicitly enumerated;
- the NTR box uses a tiny fixed alphabet to encode the ordinary alphabet by coordinate-like groups;
- the DBAC extension likewise uses a very small base alphabet.

The exact literal mechanism emits only a few base characters. The frozen RF representation requires nineteen core surface symbols to retain >=99.5% of alphabetic positions. Therefore exact literal NTRC is subject first to a structural-feasibility gate.

Collapsing nineteen Voynich glyphs post hoc into three or four latent NTRC classes is **not** the historical mechanism; it is a new homophonic/representation model and may not be introduced as a rescue after seeing target results.

### A07 — walking/two-stream cipher

Sections: 369; 373–376.

Status: `ATTESTED_EXACT` at the source level, but target applicability `NEEDS_SOURCE_RECONSTRUCTION`.

Attested facts:

- the walking cipher uses two numerical keys to distribute/extract plaintext into two interlocked streams;
- the Glorioso construction uses a two-stream split as one layer.

Before a target arm is admitted, the programme must reconstruct the exact linear ciphertext representation and prove on a worked example that the inverse is unique given the key. If the output depends on cover-text placement or missing physical cues, it remains outside the direct Voynich-text test.

### A08 — multi-layer Glorioso

Sections: 374–376.

Status: `SOURCE_COMPOSITE`.

Attested sequence in the catalogue:

1. columnar transposition;
2. two-key stream split;
3. half-alphabet substitution;
4. ordinal-number conversion;
5. steganographic embedding in cover text.

This is evidence that later Venetian cryptographers combined mechanisms, but it does **not** license arbitrary composition in the Voynich search. v1 does not target Glorioso as a whole because the final representation is cover-text steganography and the Voynich text is openly cipher-like. It may be revisited only if independent manuscript evidence supplies the missing carrier/route constraints.

### A09 — syllable mutation

Section: 397.

Status: `ATTESTED_PRINCIPLE` / partly exact, but previous reuse claim corrected.

The source states that a syllable can be changed by one letter into another syllable.

Important correction:

This does **not**, on the evidence presently extracted, establish a rule of 'when the same token is reused, mutate one component'. The March 2026 phrase `token mutation on reuse` is stronger than the quoted source. A reuse-conditioned Voynich rule is therefore not admitted on section 397 alone.

An exact source-derived mutation arm may be added only if the full section yields a unique executable encryption rule independent of Voynich behaviour.

### A10 — modulo-105 / three-position numeric code

Sections: 498–500.

Status: `ATTESTED_EXACT`; direct form likely `STRUCTURAL_MISMATCH`.

Each plaintext letter is represented by three digits drawn from different small ranges and decoded by the modulo-105 construction. Literal output therefore has a tiny digit alphabet and approximately three output units per plaintext unit. Direct compatibility with the RF surface must be evaluated structurally before any latent-class reinterpretation.

### A11 — dual-meaning / two-text cipher

Sections: 491–492.

Status: `ATTESTED_PRINCIPLE`, but generally `NON_IDENTIFIABLE` for Voynich without an externally fixed second reading/key.

The source attests ciphers that can yield a false/idle text and a true text, and alternating use of two texts/rings. Without independent constraints on the false text, true text, key schedule or selector, this family has too many degrees of freedom for a legitimate target search.

## Source-attested compositions admitted prospectively

Only the following compositions may enter v1 without a new protocol version:

1. `R12H`: twelve-letter reduction + globally fixed homophonic surface mapping, motivated by sections 024–026.
2. `R12_PWA`: twelve-letter reduction + word-reset polyalphabetic/table operation, explicitly motivated by sections 456–458.

`R12_PWA` may be scored on Voynich only if both the reduced-alphabet instrument and the positional-polyalphabetic instrument qualify independently on fresh controls.

No other composition may be created in response to target performance.

## Historical-horizon classification

The computational programme is intentionally broader than the 1400–1450 closeout. Every admitted family receives a separate historical grade:

- `H0`: secure primary/manuscript attestation dated <=1450;
- `H1`: secure near-primary attestation <=1450 or a securely dated <=1450 witness preserving the operation;
- `H2`: later source explicitly attributes the operation to an earlier tradition/person, but no <=1450 operational witness has yet been verified;
- `H3`: securely attested only after 1450 in the present evidence;
- `HX`: chronology unresolved.

Only `H0/H1` can reopen the earlier statement about the circa-1400–1450 active cipher space. `H2/H3/HX` experiments are deliberate later-Renaissance/anachronism stress-tests.

Historical research must use non-Voynich sources and should prioritize manuscripts, critical editions, library catalogues and specialist history-of-cryptography scholarship. Voynich forums/blogs may not establish admission or chronology.

## Binding interpretive rules

1. A source mechanism can be real and historically interesting while still being anachronistic for Voynich.
2. Failure of a solver to recover fresh positive controls is `CALIBRATION_BLOCKED`, not evidence against the mechanism.
3. A target negative is valid only when the family qualified prospectively and the best converged held-out target fit misses an absolute positive-control floor.
4. A target score above a floor is not evidence unless family recognition/specificity also qualifies.
5. No historical claim is upgraded because a Voynich score looks good.
6. No Voynich representation, language, selector, key schedule, family member or threshold may be changed after target scoring to rescue a result.
