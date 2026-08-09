# Cipher Coverage Closeout Programme v1 — final closeout

Date: 2026-08-09
Repository: `digitalgoldfisj79/Voynichdecomp`
Branch: `experiment/cipher-coverage-closeout-v1-20260809`
Gate-1 audit commit: `4420a84b2385189edf5a10bb8dc7fa91dd165e85`
Protocol freeze: `394adef38ed7d724680c6ee722e2a5ad4d3b44ac`
Implementation: `ea5c5cccdf79d18db66097d181e4535ffcf03cda`
Qualification archive: `cb792a8b933803a5cc02950ef32c9bdc3ada93f0`
H1 target archive: `6c993fe4e1be3a9f9eeab0332f2408bf0cf6657c`

## Final programme verdict

**BROAD IDENTIFIABLE HISTORICAL CIPHER HYPOTHESIS CLOSED UNDER COVERAGE v1 AS AN ACTIVE RESEARCH PROGRAMME.**

This is stopping-point **B** from the handover.

The claim is deliberately bounded. It does **not** mean that every imaginable encoding is mathematically impossible, nor that every historical cipher family has received a successful blind Voynich decryption attempt. It means:

1. the repository-wide audit found no important, finite, historically justified circa-1400–1450 mechanism that had simply been overlooked after the terminal additions were made;
2. the three genuine residual gaps admitted by Gate 1 were addressed under a frozen protocol;
3. all three terminal families qualified strongly on controls;
4. all three then failed the first legitimate held-out Voynich absolute compatibility test;
5. the remaining unresolved cipher possibilities are either already blocked by prospective recoverability/recognition failure, historically outside the primary period, or non-identifiable without new external information.

Accordingly, further cipher-family invention or optimizer tuning is not scientifically justified by the current evidence. Cipher remains a residual logical possibility requiring **new external evidence or a materially new validated inference method**, not an active search programme.

## 1. Terminal experiment result

All three terminal families passed Q1 exact-rule recovery and Q2 absolute calibration before Voynich H1 was opened.

| family | mechanism | Q1 | Q2 rank | Q2 median recovery | best H1 score − positive floor | final |
|---|---|---:|---:|---:|---:|---|
| M0 | global monoalphabetic substitution | PASS | 24/24 | 1.0000 | **-0.34168** | CLOSED NEGATIVE |
| TQ | medieval within-word transposition + one global substitution key | PASS | 24/24 | 1.0000 | **-0.21345** | CLOSED NEGATIVE |
| NQ | deterministic one-null-per-word insertion + one global substitution key | PASS | 24/24 | 1.0000 | **-0.36448** | CLOSED NEGATIVE |

H1 representation coverage was 99.5679%. The winning fit in every family converged with independent-ensemble map agreement 1.0.

No family reached its own frozen 5th-percentile positive-control floor. Therefore:

- Q3 specificity was not run;
- C1 remained sealed;
- no plaintext was emitted or inspected;
- there was no opportunity for readable-output selection.

H1 scientific SHA-256:
`52794ec9f0d583d064f4a5860253bffe20f74ec70802807607da5abb19fea6a7`

## 2. Final mechanism coverage ledger

The classifications below distinguish a mechanism-level target negative from a failed inference instrument.

### A. Mechanisms with legitimate negative target/structural results

**CLOSED**

- simple global monoalphabetic substitution under the frozen RF core-19 representation — M0 v1;
- fixed historical Tranchedino homophonic alphabet — H21 negative against absolute historical-control floor;
- M19/STA hierarchy — qualified controls, held-out target failed preregistered language margin;
- coarse BnF numerical key partitions at Currier/section/section×Currier scales — qualified but no resolving language separation;
- BnF deterministic unmarked numerical schedules — all 25 frozen models structurally rejected;
- medieval within-word/local transposition rules — TQ v1;
- historically attested local transposition composed with a global substitution — TQ v1;
- bounded deterministic scheduled-null insertion — NQ v1;
- tested Tranchedino historical templates as specified in v2.0–v2.4 — programme closed without qualified signal.

### B. Historically meaningful families that were addressed but blocked before target because the instrument did not qualify

**PARTIAL / CALIBRATION-BLOCKED — not evidence for cipher**

- fresh-key generic homophonic substitution: locked blind reliability failed despite near-complete recovery in favourable basins;
- nomenclators / fresh opaque whole-word codebooks: component identifiability/reliability failed;
- substitution + fixed repeated block/columnar transposition: oracle components recoverable, blind joint solver unreliable;
- terminal Family S polygraphic/variable visible code groups: development recovery failed;
- Tranchedino one-sign mixed alphabet/geminate/null/nomenclator and one-sign syllabary variants: prospective component recovery failed before target;
- bounded carrier steganography Family G: oracle carrier recovery strong, blind family-wide detection/recovery failed;
- marked M57 BnF code assignment: first control failed catastrophically;
- quire-level BnF coarse keys: positive-control gate failed;
- source-transfer MDL, compression-transfer, morpholocal and CoReMA: recognition/transfer instruments failed calibration and cannot be promoted into cipher-family negatives.

These rows remain logically possible, but the current programme has no validated way to turn a Voynich fit or failure into evidence. Re-running them with more restarts, different thresholds or favourable subsets would not solve that epistemic problem.

### C. Proposed gaps removed by overlap or historical screening

**NOT A LIVE GAP**

- generic multiple-visible-glyphs -> one plaintext unit: already central to Family S;
- one visible sign -> syllable/mixed plaintext unit: specifically tested by Tranchedino v2.4;
- fixed-width vertical/columnar transposition: already represented in v0.5.5 / Family T;
- irregular/signalled changing alphabets as a primary 1400–1450 mechanism: computational umbrella scope exceeded executed Family P, but the firm Alberti changing-alphabet anchor is 1466–1467 and no concrete pre-1450 observable-state rule was established for v1 admission;
- generic route/grille transposition: no sufficiently specific primary 1400–1450 anchor was established in Gate 1, and arbitrary path freedom is not admissible.

### D. Residual classes that cannot be empirically searched without external constraints

**NON-IDENTIFIABLE**

- arbitrary content-dependent null placement;
- arbitrary per-word/per-line/per-section/per-folio key changes;
- arbitrary geometric or semantic routes;
- folio-specific steganographic extraction rules selected after inspection;
- unconstrained codebooks or plaintext unitisations;
- lossy deletion/omission where missing plaintext is not uniquely recoverable;
- arbitrary combinations of independently flexible mechanisms.

These are not scientific alternatives merely because some parameter choice can be made to produce a readable string.

## 3. What the closeout establishes

The programme now has a much stronger basis for deprioritising cipher than a collection of failed decipherment attempts.

The evidence consists of four different types:

1. **qualified target negatives** where known controls establish that the mechanism is recoverable and the held-out Voynich target fails an absolute compatibility floor;
2. **structural contradictions** where a frozen historical mechanism cannot even represent required control behaviour;
3. **prospective calibration failures** showing that some attractive families cannot currently be tested blindly without conflating search failure with mechanism failure;
4. **identifiability exclusions** preventing arbitrary model flexibility from being counted as evidence.

This distinction is essential. The final conclusion is not "all ciphers failed." It is:

> after repository-wide coverage audit and one final historically anchored terminal programme, no important finite and testable historical cipher mechanism remains as an unexamined active lead; the genuine terminal gaps fail calibrated held-out Voynich testing, while the residual possibilities require new information rather than more search.

## 4. Scientific boundary

### Supported

- broad cipher should leave the active hypothesis set under the current evidence programme;
- simple monoalphabetic substitution now has a legitimate calibrated Voynich negative under v1;
- medieval local transposition, including the historically licensed transposition+substitution composite, now has a legitimate calibrated negative under v1;
- bounded deterministic inserted-null schedules now have a legitimate calibrated negative under v1;
- the earlier Family S/P/T/G and Tranchedino results should be interpreted according to their actual executed scopes, not broad umbrella labels;
- further optimizer-only or representation-shopping extensions are not justified.

### Not supported

- proof that Voynich cannot be a cipher in any logical sense;
- rejection of a family whose positive-control/recovery instrument never qualified;
- rejection of unknown historical mechanisms not represented in surviving sources;
- rejection of a mechanism that depends on an externally unknown crib, key, table, physical cue or reading order;
- any plaintext, language identification or semantic decryption claim from this programme.

## 5. Conditions for reopening cipher research

The broad cipher programme should be reopened only if at least one of the following occurs:

1. **new primary-source evidence** identifies a materially distinct operation or binding usage rule not present in the coverage ledger;
2. **a real historical ciphertext/key or ciphertext/plaintext pairing** makes a previously calibration-blocked family reliably recoverable under fresh locked controls;
3. **new manuscript evidence** independently fixes a key schedule, route, unitisation, carrier or state/reset rule before plaintext inspection;
4. **a materially new inference method** passes fresh prospective controls for one of the calibration-blocked families without using Voynich for development.

Absent one of these triggers, additional cipher search is post-hoc model proliferation.

## 6. Compute closeout

The terminal protocol required a Hugging Face running-job check before every paid launch. The final H1 job completed, and a post-run check found no running Hugging Face jobs. No paid compute was left orphaned.

## Binding final statement

**CIPHER COVERAGE CLOSEOUT v1: STOPPING POINT B REACHED.**

**Broad identifiable historical cipher research is formally closed as an active Voynich programme under the present evidence.**

Residual cipher possibilities are retained only as bounded epistemic uncertainty and require new external evidence or a newly qualified method before they may be reopened.
