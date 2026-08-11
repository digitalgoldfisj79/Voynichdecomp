# Amadi Residuals v1 — Source Reconstruction

Date: 2026-08-11
Status: **FROZEN BEFORE QUALIFICATION / VOYNICH H2 SCORING**

Source basis: Scheers transcription/edition as preserved in the supplied Amadi extraction and cumulative running analysis. This document distinguishes exact executable operations from source principles that remain underdetermined.

## A01 / R12H — twelve-letter reduction

### R12_V1_024 — admitted exact reduction

Section 024 explicitly avoids `b,d,f,g,h,p,q,u` when the last is consonantal. The operational examples give:

- `b -> u`
- `d -> t`
- `f -> deletion`
- `g -> i`
- `h -> deletion`
- `p -> deletion`
- `q -> c`
- consonantal `u/v -> o`

Remaining reduced alphabet:

`a c e i l m n o r s t u`

Source examples include `labro -> lauro`, `grande -> grante`, `felice -> elice`, `gioue -> ioue`, `pietro -> ietro`, `quando -> cuando`, `mouendo -> mooendo`.

Implementation rule for modern UD controls is frozen as follows: preserve modern `v` long enough to apply the source consonantal-u/v rule (`v -> o`) before the inherited `j/v/w/y/x/z` historical normalization. This avoids destroying the only observable distinction needed to execute the historical rule. Other normalization remains unchanged after reduction.

Q0 requirement: the runner contains an exact regression set for the quoted word examples and must pass all examples before Q1.

### R12_V3_390 — NOT admitted as an executable rule

Sections 390–392 show a wheel using only the twelve values `casteimlonru`, with `d,f,g,p` absent, but the supplied source extraction does not provide a complete deterministic transformation from arbitrary plaintext into that twelve-symbol surface. Missing letters alone do not specify whether each is deleted or replaced.

Status: `SOURCE_UNDERDETERMINED`. It is not a Q1/Q2 rule in v1.

### R12_V4_454 — NOT admitted as a complete rule in v1

The extracted section gives an explicit twelve-letter target alphabet `A,C,E,I,L,M,N,O,R,S,T,V` and states B→V, D→T, F→I, G→P, Q→C, consonantal V→O, while B,D,F,G,H,P,Q,V(consonant) are all said to be avoided. The worked example supports B→V, D→T, F→I and H deletion.

However the extracted rule `G -> P` points to `P`, which is simultaneously excluded, and the supplied extraction does not give a unique subsequent P rule. Completing `G -> P -> deletion` would be an invented inference. Therefore full arbitrary-text execution is not licensed in v1.

Status: `SOURCE_UNDERDETERMINED` for general-purpose controls. No target rule is created from it.

### R12H stress-test surface

The single source-exact reduction rule admitted to qualification is `R12_V1_024`.

After reduction, the broad `R12H` stress-test maps the 19 observable RF/control symbols surjectively onto the 12 latent reduced letters. Each observed symbol has one document-global latent value; every latent value has >=1 observed symbol. This 19→12 surface is broader than section 026's exact vowel-homophone table and is labelled a conservative superfamily.

## A02 / VC_END — exact section 013

Exact operation, independently within every existing word:

1. copy consonants in their original relative order;
2. append vowels in their original relative order.

No letter is inserted, deleted or substituted by this step.

Regression examples from the source:

- `non -> nno`
- `staro -> strao`
- `discorere -> dscrioee`
- `differentia -> dffrntieia`
- `riputatione -> rpttnuaioe`
- `competitori -> cmpttroeioi`
- `et -> te`
- `il -> li`
- `splendore -> splndreoe`
- `uestitto -> stttueio`

Vowels for the normalized Latin-script panel are frozen as `{a,e,i,o,u}`. The transformed language model is trained on the forward-transformed corpus. No inverse vowel-position recovery is attempted.

Status: `ATTESTED_EXACT`; Q0 requires every regression pair above to pass.

## A03 / PWA_K — source schedule + broad rejection superfamily

Section 445 states that within a word the first letter uses the first alphabet, the second the second alphabet, subsequent positions continue through successive alphabets, and the sequence returns to the first alphabet for the next word.

Section 461 states a two-table alternation:

- position 1: table 1 / alphabet 1
- position 2: table 2 / alphabet 1
- position 3: table 1 / alphabet 2
- position 4: table 2 / alphabet 2
- and so on.

The target rejection family remains the protocol-defined broad superfamily `PWA_K`, `K in {2,3,4,5}`, one independent global bijection per modulo-word-position state, phase reset at each word. It does not claim to reproduce either historical table exactly.

Q0 for the broad family is synthetic self-reproduction: encrypt/decrypt generated controls under each K and verify exact oracle recovery. Exact 445/461 narrowing is deferred unless H2 survives.

## A04 / GHOUSE5 — selector rule frozen

Amadi attests multiple houses; he does **not** attest gallows as the selector. `GHOUSE5` is therefore target-specific architecture testing.

The target extractor is now frozen:

- candidate marker set `{k,t,p,f}`;
- scan each retained RF word from left to right;
- the **leftmost** occurrence of any marker is the selector;
- remove exactly that occurrence from the payload before cipher scoring;
- selector class is its glyph (`k/t/p/f`);
- if no marker occurs, class is `NONE` and payload is unchanged;
- if more than one marker occurs, record the word as `multi_gallows_ambiguous`, but still use the leftmost marker and retain all later markers as payload;
- words whose payload becomes empty are retained for selector census but contribute zero payload likelihood.

This extractor uses no section, semantic or illustration information and assigns at most one state per word.

Synthetic GHOUSE controls carry the state labels as explicit control metadata rather than attempting to infer them from a randomly generated cipher glyph. This qualifies recovery of the state-conditioned maps. On Voynich, selector validity is separately tested by the preregistered 256 within-folio state-label permutations.

## A05 / plaintext autokey — stopped at source reconstruction

The extracted section 490 wording establishes the principle: first alphabet for the first word; a plaintext letter selects the following word's alphabet; another plaintext letter then selects the next. The supplied material does not preserve a worked example and complete enough table/state description to remove ambiguity over the exact letter index/state transition in all cases.

Status: `SOURCE_UNDERDETERMINED`. Conditional arm C1 does not enter qualification or target testing in v1.

## A06 / NTRC — exact reconstruction, structurally screened

Sections 479–482 enumerate groups over `{N,T,R,C}`. The NTR box uses a two-character column header plus one row character; literal output uses only the tiny N/T/R/C surface. The exact family therefore has output alphabet cardinality <=4.

No 19→4 RF glyph collapse is licensed.

Status: exact algorithmic principle reconstructed sufficiently for the structural gate; no statistical solver is built because the direct surface gate fails (see `STRUCTURAL_GATE_V1.json`).

## A07 / walking/two-stream — stopped for direct RF application

The supplied extraction establishes keys `8.4.2.1` and `7.9.3.1` and two interlocked streams, while the related 373–376 examples use cover-text placement. The material presently available does not yield one unique direct linear RF ciphertext representation plus a complete worked inverse without relying on the cover carrier.

Status: `SOURCE_UNDERDETERMINED_FOR_DIRECT_RF`. Conditional arm C2 stops.

## A08 / Glorioso

Source composite is reconstructed conceptually as transposition -> two-stream split -> reciprocal substitution -> numericization -> cover-text steganography. The final carrier is not the openly segmented RF surface.

Status: not a v1 target arm.

## A09 / syllable mutation

Section 397 attests that changing one letter changes one syllable into another. It does not provide evidence in the supplied extraction for a reuse-triggered mutation schedule.

Status: no independent v1 arm.

## A10 / modulo-105 — exact principle, structurally screened

Sections 498–500 encode each plaintext letter by three digits from row ranges 0–2, 0–4 and 0–6 and decode with weighted contributions modulo 105. Literal output alphabet is only digits `0..6`, cardinality 7.

No 19→7 RF glyph collapse is licensed.

Status: direct surface gate fails; no statistical solver is built.

## A11 / dual meaning

A false/idle and true reading is attested, but without an independently fixed second text/key it is non-identifiable for a direct target search.

Status: `NON_IDENTIFIABLE`.

## Frozen primary computational inventory after Stage S

Survive to qualification:

1. `R12H / R12_V1_024` only
2. `VC_END`
3. `PWA_K` for K=2,3,4,5
4. `GHOUSE5`

Stopped before qualification:

- R12_V3_390: source underdetermined
- R12_V4_454: source underdetermined at G/P branch
- plaintext autokey: source underdetermined
- walking/two-stream direct RF: source underdetermined / carrier-dependent
- NTRC exact: surface incompatible
- modulo-105 exact: surface incompatible
- Glorioso: not direct RF family
- syllable mutation: no unique independent schedule
- dual meaning: non-identifiable

No further family may be added in response to qualification or target results.