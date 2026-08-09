# BnF M19 German/Bavarian Mechanism Diagnosis v1.1 — Result

Date: 2026-08-09
Branch: `experiment/bnf-m19-why-german-v1.1-20260809`
Parent result: v1.0 `CONFIRMED FRESH-PANEL GERMAN M19 SIGNAL`

## Bottom line

The v1.0 German ranking is a real, highly reproducible property of the **literal-character EVA-family representation under the frozen BnF M19 map**, but v1.1 finds that it should **not be interpreted as evidence of German or Bavarian plaintext**.

The dominant explanation is representation sensitivity. EVA connected sequences (`ch`, `sh`, `cfh`, `ckh`, `cph`, `cth`) were split into their ASCII component characters in v1.0. Those within-connected-sequence edges alone contribute `+0.050970` nats per literal character to German over French, slightly more than the entire v1.0 within-word transition advantage (`+0.049480`). The BnF singleton anchors, especially the value-23 plaintext `n` forced onto a Voynich character that is almost always word-final, supply a second large contribution.

When connected EVA sequences are tokenized atomically, the positively controlled 31-unit M19 instrument remains fully capable of recovering synthetic ciphers, but the German result disappears: German ranks fourth on the atomic H09 panel and its independent fitted keys agree only 73.1%. A stable Greek fit appears on H09 but does not transfer to the untouched C10 panel, where French ranks first and Greek fourth. There is therefore no stable replacement plaintext language under the atomic representation.

The period-specific Bavarian test also rejects the proposed localization at the resolution the qualified instrument can support. A document-disjoint 5th-order numerical classifier distinguishes ReF Bavarian from non-Bavarian 1350–1500 controls with 0.90 balanced confirmation accuracy, but the frozen Voynich number stream prefers the **non-Bavarian** macro-class by 0.0540 nats/symbol, with the same sign in all four buckets.

Overall classification: **EVA REPRESENTATION SENSITIVE; GERMAN SIGNAL EXPLAINED LARGELY BY TRANSLITERATION-EDGE + BnF-ANCHOR INTERACTION. NO BAVARIAN SIGNAL. NO PLAINTEXT CLAIM.**

## 1. Frozen v1.0 phenomenon being explained

The literal-character fixed map was:

`a=5 b=22 c=6 d=4 e=1 f=16 g=22 h=3 i=10 j=20 k=2 l=12 m=9 n=23 o=1 p=7 q=4 r=24 s=30 t=8 u=0 v=28 x=28 y=5 z=20`

On the 122-folio C10 panel the German model ranked first on ZLZI, TTLI and VDRB with margins 0.157735, 0.144796 and 0.164746 nats/letter. All four C10 buckets also ranked German first. These observations remain valid as properties of that representation and model.

## 2. BnF numerical architecture is relevant but not sufficient

A 20,000-key uniform legal-M19 induced-score screen found German ranked first for 20.045% of random legal maps. Only 4.10% of random maps reached or exceeded the frozen map's induced German margin. Thus M19 geometry can generate German preference for some keys; the frozen key is unusual but not uniquely so.

A separate 10,000-permutation null shuffled the 23 actual BnF five-value profiles among plaintext letters while preserving the profile multiset and all numerical values. Only 1.46% of randomized incidences produced a German advantage at least as large as the real BnF letter/value incidence (`p=(146+1)/(10000+1)=0.01470`). The actual BnF assignment therefore matters; the effect is not caused only by having 19 numerical classes.

A 64-map exact-forward sample was deliberately stratified across the induced-score distribution, not sampled uniformly. Four stratified maps exceeded the frozen exact German margin. This demonstrates that strong German exact-forward fits exist elsewhere in the legal key family, but `4/64` is not an admissible random-null p-value.

## 3. Hard BnF anchors explain a large part of the effect

The BnF unmarked channel has five singleton plaintext values: `0→y`, `22→o`, `23→n`, `28→f`, `30→s`.

Under the frozen v1.0 map, on C10 ZLZI:

- cipher `n → 23 → plaintext n`: 3,201 occurrences, of which 3,166 are word-final;
- cipher `s → 30 → plaintext s`: 3,886 occurrences, of which 2,486 are word-initial;
- the other singleton anchors are rare in this transcription.

Exact ablation of the value-23 `n` anchor reduces the German margin from 0.157735 to 0.086753. Removing `s` alone reduces it to 0.130653. Removing both leaves 0.055980. Removing all singleton-anchor surface symbols leaves 96,771 literal characters (93.09% of C10) and a German margin of 0.055477.

Thus the hard positional `n/s` anchors, especially the almost-always-final `n`, are substantial drivers rather than incidental details.

## 4. Direction and transitions, not unigram frequencies, drive the remainder

Under the induced numerical model the German-vs-French advantage is 0.079484 nats/literal-character. Its decomposition is approximately:

- internal within-word transitions: +0.049480;
- word starts: +0.009267;
- word ends: +0.020737.

A unigram-only channel gives a much smaller German advantage (~0.0223). Reversing characters within words destroys the German result: Latin ranks first and German third. Concatenating words within folios while removing normal word-boundary effects still leaves German first, showing that ordered internal structure is important.

## 5. Decisive EVA representation diagnosis

The v1.0 model treated every literal lowercase transliteration character as an independent ciphertext symbol. However, standard EVA marks `ch`, `sh`, `cfh`, `ckh`, `cph`, and `cth` as connected sequences.

Greedy longest-first collapsing of those connected sequences on C10 changes the representation from:

- 103,954 literal characters / 25 literal character labels

to:

- 93,485 connected/basic units / 31 unit labels.

The connected forms are common: `ch` occurs 5,738 times, `sh` 2,513, `ckh` 512, `cth` 455, `cph` 102, `cfh` 40.

The literal edges internal to those connected forms contribute the following German-vs-French induced advantage per original literal character:

- `sh`: +0.025280;
- `ch`: +0.022447;
- `ct`: +0.001917;
- `th`: +0.000910;
- `ck`: +0.000770;
- remaining connected edges net approximately -0.000384.

Total connected-sequence internal-edge contribution: **+0.050970**.

That is slightly greater than the entire observed within-word German transition advantage (+0.049480). Ordinary non-connected within-word transitions therefore net slightly against German in this decomposition.

If connected-form internal edges are removed from the literal induced score while keeping all other literal-character information, German remains first but the margin falls to 0.056319. If the singleton anchors are also removed, the German margin falls further to **0.019784**.

This explains roughly three quarters of the induced German advantage using two identifiable design features: EVA connected-form decomposition and BnF singleton-anchor placement.

A particularly transparent example is EVA `sh`. In the literal v1.0 map `s→30`, and 30 is the BnF singleton plaintext `s`; `h→3`, whose ambiguity set includes several letters. Common following characters then allow the German model to construct high-probability `sch...`-like paths. This is a property of the interaction between EVA's ASCII decomposition and the BnF equivalence classes, not a recovered German word.

## 6. Atomic-EVA falsification

A prospective atomic test was frozen before fitting an atomic Voynich key. Connected sequences were tokenized longest-first as `cfh`, `ckh`, `cph`, `cth`, `ch`, `sh`; all other characters remained singleton units. The 31 surface units were modelled as 19 numerical values with 12 duplicated homophones.

The atomic instrument passed its synthetic qualification gate decisively:

- 6/6 languages correctly identified (Latin, Italian, German, French, Arabic, Spanish);
- minimum correct-language margin 0.084155;
- median numerical-map recovery 1.000;
- minimum numerical-map recovery 1.000;
- minimum independent-fit agreement 1.000.

Atomic Voynich H09 ranking:

1. Greek: -3.119597, fit agreement 0.9066;
2. French: -3.421231, fit agreement 0.5719;
3. Italian: -3.474250, fit agreement 0.9516;
4. German: -3.537436, fit agreement **0.7308**;
5. Arabic: -3.627420, fit agreement 0.9619;
6. Hebrew: -3.643792;
7. Latin: -3.798330;
8. Spanish: -3.888235.

German therefore fails the frozen stability gate under atomic EVA.

Greek meets the H09 atomic primary rank/stability rule, but its T09-fitted atomic key does not transfer to C10. C10 fixed-map ranking is French first (-2.612432), German second (-2.661970), Spanish third (-2.689218), Greek fourth (-2.705727). The H09 Greek result is therefore not a corpus-wide atomic plaintext candidate.

Conclusion of atomic test: **the v1.0 German signal is EVA-representation sensitive; no stable atomic replacement language is established.**

Atomic positive-control/Voynich job: `6a78306fda2af92a634eff42` (completed).

## 7. Bavarian / historical German tests

The official ReF 1350–1500 window provides substantial dialect material, including:

- Middle Bavarian: 8 documents / 583,898 normalized letters;
- North Bavarian: 8 / 311,374;
- South Bavarian: 4 / 202,133;
- plus Alemannic, Swabian, East Franconian, Hessian, Ripuarian, Thuringian, Upper Saxon and other groups.

The simple dialect-level bigram/HMM instrument does not reliably identify its own encrypted document-disjoint dialect controls: Middle Bavarian ranks fifth, North Bavarian eleventh, South Bavarian ninth. Fine dialect localization is therefore **underpowered** and those raw regional Voynich rankings are inadmissible.

A separately frozen coarser numerical n-gram test asked only Bavarian vs non-Bavarian. Fifth-order numerical n-grams were selected on development data (balanced accuracy 0.925). On fresh document-disjoint confirmation controls the classifier achieved:

- balanced accuracy: 0.900;
- Bavarian accuracy: 0.950;
- non-Bavarian accuracy: 0.850;
- positive median true-class margin: 0.02446.

This passed the frozen qualification gate.

On the unchanged C10 Voynich numerical stream:

- non-Bavarian score: -2.506745;
- Bavarian score: -2.560779;
- Bavarian-minus-non-Bavarian margin: **-0.054034**.

All four deterministic C10 buckets have the same non-Bavarian sign (-0.0394, -0.0625, -0.0527, -0.0608).

Verdict of the qualified macro test: **NON-BAVARIAN MACRO SIGNAL**. This is not a positive identification of any specific non-Bavarian dialect; it only rejects a Bavarian preference under this qualified binary instrument.

Bavarian numerical job: `6a782e1eda2af92a634eff26` (completed).

## 8. Transcription transfer

The literal-character map reproduces on additional EVA-compatible or near-identical transcription surfaces after a train-only character crosswalk:

- JSLI: German rank 1, margin 0.150245;
- JGLI: rank 1, 0.129518;
- ZLZB: rank 1, 0.157779;
- VDRB-1: rank 1, 0.157065;
- TTVE: rank 1, 0.157551;
- TTIA/TTII: rank 1, 0.157614.

Characterwise crosswalks to GCGA/GCGI, FFSG, RGVN and PCCA/PCCI fail the pre-set agreement threshold (held-out agreement roughly 0.53–0.69). Their transcription conventions are not safely reducible to a one-character EVA crosswalk, so they are neither positive nor negative evidence for the literal-character key.

The successful additional surfaces largely preserve EVA-style decomposition, so they confirm transcription reproducibility but do not answer the atomic-glyph representation objection.

## 9. Section / Currier diagnostics

With the frozen literal key, German ranks first in every sufficiently populated C10 content section and in Currier A, B and C. The effect is therefore not localized to one manuscript section. This robustness is compatible with a global transliteration/positional-structure interaction and does not rescue a plaintext interpretation.

## Final assessment

The sequence of experiments changes the interpretation of v1.0 substantially.

What remains robust:

- BnF's actual numerical incidence is unusually compatible with Voynich's literal EVA-family structure;
- the v1.0 fixed map produces an out-of-sample, cross-transcription statistical German preference;
- the interaction is directional and highly structured, not a trivial unigram coincidence.

What v1.1 explains away or fails to support:

- most of the German advantage is attributable to EVA connected forms being split into ASCII component transitions plus hard BnF singleton anchors;
- German does not survive an independently qualified atomic-EVA representation;
- the apparent replacement Greek signal fails fresh C10 transfer;
- a qualified Bavarian-vs-non-Bavarian instrument prefers non-Bavarian, not Bavarian;
- decoded text remains non-readable and cannot be treated as plaintext.

### Programme verdict

**EVA REPRESENTATION SENSITIVE / TRANSCRIPTION-EDGE DOMINATED.**

The BnF 7342 apparatus remains a historically interesting numerical transformation mechanism, and its interaction with Voynich structure is non-random enough to merit documentation. But the current evidence does **not** support German, Bavarian, or any other language as recovered Voynich plaintext under M19. The proper next cryptanalytic unit is a manuscript-glyph/connected-unit representation that is independent of EVA's ASCII labels and decomposition choices.
