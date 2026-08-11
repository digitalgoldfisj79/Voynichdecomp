# VBM v1 — Vowel-Bridge Model Protocol

Date: 2026-08-11
Namespace: `VBMV1`

## Historical model being tested

This is the previously defined VBM, not a new post-Babuini construction. Visible spaces are hypothesised false boundaries cutting a vowel/linking bridge. A line is represented as repeated bridge/core material with line-start and line-end operators. The recovered prior architecture was approximately `VR | C1 | C2 | VL`; cross-space `VL.VR` is the vowel/linking bridge. `C1` is predominantly substitutional; `C2` may be composite/polyphonic; `e/ee/eee` chains are composite. Previously observed couplings included `ed -> y`, `eed -> y`, and the frequent `y.qo` bridge.

## Frozen executable representation

1. Work on the frozen RF core19 transcription.
2. Preserve transcription lines and visible word boundaries.
3. A word beginning `qo` uses `VR=qo`; otherwise `VR` is its first retained character.
4. `VL` is its final retained character.
5. At a line start the first VR is an LSM operator and is not plaintext payload. At line end the final VL is an LEM operator and is not plaintext payload.
6. For every internal visible boundary emit exactly one bridge event `VL.left + '.' + VR.right`. Every bridge event is constrained to decode to a vowel.
7. The material between VR and VL is core material. Greedily at the right edge recognise `eed` then `ed` as single composite C2 events; in the remainder collapse every maximal `e+` run to one composite event; all other retained characters are singleton core events. Every core event is constrained to decode to a consonant.
8. No other multigraph is introduced in v1.

## Target split

The six-folio H1 and six-folio C1 are prospectively split from the 12 folios that remained sealed after Amadi Core Babuini v1.

VBM_H1: `f28v f31v f88r f5r f34r f81v`

VBM_C1: `f85r1 f53v f33r f10r f23r f111r`

FIT remains the original 181-folio FIT-A; no previously opened Amadi H2 material is added.

## Language panel

Primary hypothesis: Bavarian (`bar`), motivated independently by the user's prior VBM hypothesis and current judgement. Nearest rival: Standard German. Source-native comparator: Italian.

Bavarian model corpus: Bavarian Wikipedia dump `bavarian-nlp/barwiki-20250720`; deterministic article split, residues 0–5 train, 6–9 controls. German and Italian use UD GSD / ISDT train for modelling and dev+test for controls.

All text is transliterated into frozen 19-letter Latin normalisation; umlauts collapse through Unicode transliteration; j->i, v/w->u, y->i, x/z->s.

## Stage S0 — binding topology gate

Before any surface-to-letter substitution optimisation, compare the VBM event-type sequence against natural-language C/V topology.

For each language fit an add-0.25 order-4 model over C/V sequences with explicit line/sentence boundary. From untouched language controls create 16 deterministic spans at the VBM_H1 event scale. Freeze the 5th-percentile held-out log score (nats/event) as that language's absolute topology floor.

VBM_H1 passes S0 only if at least one panel language reaches its own absolute floor. Bavarian is a language candidate only if it passes its floor and ranks first with >=0.02 nats/event margin over both rivals.

If no language passes S0, verdict: `VBM V1 TOPOLOGY INCOMPATIBLE`; no substitution solver and no VBM_C1 are authorised.

## Stage S1 — typed substitution (conditional)

Only if S0 passes. Select bridge vocabulary on FIT only at >=0.995 bridge-event coverage. Core events and bridge events are separate typed homophonic alphabets. Core surfaces map only to consonants; bridge surfaces map only to vowels. Fit independent A/B convergence-controlled annealing ensembles for each language on FIT and apply without refitting to H1.

Synthetic qualification must precede target S1: >=0.95 median plaintext recovery, >=0.85 minimum recovery, >=0.90 minimum A/B map agreement, correct language rank1 >=90%, and <=1/50 structured-negative false positives.

Target candidate requires convergence, H1 score >= qualified language-specific floor, and language margin >=0.02. Only then open VBM_C1.

## Stop rules

No changing VR after S0, no adding bridge prefixes other than `qo`, no extra composite core units, no larger language panel, and no optimizer-budget increase after H1. A failed S0 is a binding rejection of this executable VBM v1, not of all imaginable false-boundary models.
