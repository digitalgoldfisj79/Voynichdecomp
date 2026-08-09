# Tranchedino × STA v2.4 — B1 D1 development note

Date: 2026-08-09
Status: **DEVELOPMENT ONLY — NOT QUALIFICATION EVIDENCE**

Before the B1 oracle qualification protocol was frozen, a simple syllabary-permutation prototype was developed against four sequential held-out Paduan chunks of approximately 12,000 normalised letters each.

The prototype supplied the true non-syllabic semantic mapping and the true syllable/non-syllable class partition, then attempted only to recover the permutation between observed opaque syllable signs and the fixed 64-entry historical f.134v–135r syllabary.

A source-trained semantic-unit trigram model plus frequency initialisation and pair-swap simulated annealing was sufficient to show that this component is computationally promising. One representative development run per syllable-use stratum at `p_null=0.03` gave occurrence-weighted syllable recovery approximately:

- `p_syll=.25`: 0.9567;
- `p_syll=.50`: 0.9969;
- `p_syll=.75`: 1.0000;
- `p_syll=1.00`: 1.0000.

These values were inspected during solver development and therefore **cannot qualify the instrument**.

The four development chunks consumed the first 1,248 held-out line records. All of those records are permanently D1-contaminated for v2.4 recovery testing.

The untouched held-out tail contains:

- 175 line records;
- 6,619 normalised 19-letter characters;
- source pages 243–251.

No recovery metric or decoded text from this tail had been generated at the time the B1-O1 qualification protocol was frozen. It is designated the prospective Q1-O1 source below.

The small size and single-plaintext nature of this tail mean that even a Q1-O1 pass is only a **component-oracle qualification**. It cannot by itself qualify a full blind historical-cipher detector or authorise a Voynich fit.
