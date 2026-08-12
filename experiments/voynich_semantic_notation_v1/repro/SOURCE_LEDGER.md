# VSN-B2 Source Provenance Ledger

Date: 2026-08-12
Purpose: make every source-dependent claim traceable and distinguish primary/catalogue evidence, specialist scholarship, computational inputs, and our inference.

## A. Historical mechanism sources

### A1. Valentina Cacopardo, *Memory and Imagination in the Ars Memorativa in Fifteenth-Century Italy* (2021)

Institutional repository:
https://sas-space.sas.ac.uk/9846/

Repository metadata identifies this as a doctoral thesis, School of Advanced Study, University of London, completed 30 April 2021.

What it supports in VSN-B2:
- an early-fifteenth-century cluster of ars memorativa texts connected to Padua, Mantua and Venice;
- Matteo da Verona, *De Arte Memorandi* (1420), as one of the works treated;
- Bartolomeo da Mantova, *Liber memoriae artificialis* (1429);
- Jacopo Ragona, *Artificialis memoriae regulae* (1434);
- detailed discussion of compositional mnemonic constructions, structured loci and multi-field encoding used in the deep-research phase.

Evidence class: **specialist scholarly analysis / institutional thesis**.

Caveat: this is not itself the medieval manuscript witness. Claims about exact medieval wording should be checked against the edition/manuscript before publication.

### A2. Matteo da Verona modern critical-edition material

Work consulted during discovery: *Documenta Mnemonica*, Band III, containing a modern edition/transcription of late-medieval mnemonic texts including Matteo.

The online-access copy used during the exploratory phase was hosted by a third-party book mirror rather than the publisher/library. It was sufficient to locate and inspect the relevant rules but is **not treated as publication-quality provenance**.

What it supports in VSN-B2:
- the explicit first-syllable artificial-word operation;
- the orthographic examples used as sanity checks (`tripode`, `pepo`, `corvus`, `vetula`);
- contextual/state encoding passages discussed in the historical report.

Evidence class: **modern critical-edition transcription, accessed through non-ideal mirror**.

Required before publication:
- verify the relevant passages against a publisher/library copy of the edition or a manuscript witness;
- record exact page/folio references and edition bibliographic details;
- do not quote the mirror as the authority.

### A3. Ptolemaeus Arabus et Latinus / Jordanus — Vat. lat. 10488

Permanent manuscript record:
https://ptolemaeus.badw.de/jordanus/ms/10273

What the catalogue record directly supports:
- shelfmark: Città del Vaticano, Biblioteca Apostolica Vaticana, Vat.lat.10488;
- century: 15th;
- precise date: 1424;
- place of origin: Venice;
- language: Italian;
- subject: mathematics.

Evidence class: **specialist manuscript catalogue**.

Use in VSN-B2:
- historical comparator for operational mathematical notation in the Veneto;
- not used to generate or tune Matteo K2 strings.

### A4. Bartolomeo / Ragona manuscript claims

Current direct scholarly anchor is Cacopardo 2021 above.

Evidence class: **specialist scholarship**.

Current limitation:
- direct image-level transcription of all relevant manuscript folios was not completed in the first DR pass;
- therefore VSN-B2 does not make new visual claims about unseen folios.

## B. Computational sources

### B1. PyWORDS / Whitaker-derived Latin vocabulary

Repository:
https://github.com/sjgallagher2/PyWORDS

File used:
`pywords/data/lingualatina_voclist.txt`

Git blob at freeze:
`5dc8e924f253ef18cc72d72daa15ec49a805b8f8`

Raw-byte SHA-256 observed during the main HF run:
`5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c`

Observed source size:
- 1,902 lines;
- 1,846 eligible unique normalized source words after the frozen filters;
- 429 unique derived first syllables.

Evidence class: **computational convenience vocabulary**.

Critical caveat:
This vocabulary is not claimed to be Matteo's lexicon, medieval Paduan vocabulary, or the plaintext vocabulary of the Voynich manuscript. Its role is only to instantiate the independently attested *operation* on an external Latin lexicon.

### B2. Voynich Reference Transliteration

Source:
https://voynich.nu/data/sta/RF1b.txt

Frozen SHA-256:
`81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`

Used through the existing VSN-v1 occurrence/type tables and prior frozen STA pipeline.

Evidence class: **transliteration dataset**.

Representation caveat:
RF/EVA segmentation is not assumed to equal scribal grapheme segmentation. For that reason VSN-B2 also ran STA-family/full-STA/AAA robustness checks. Exact one-unit edit topology changes with representation granularity; conclusions are reported accordingly.

## C. Internal data tables

Supabase project: `Voynich Comparator` (`ymaqlcfjmdwncdbjprmw`).

Primary tables used:
- `voynich_semantic_notation_v1.rf_occurrences`
- `voynich_semantic_notation_v1.rf_token_types`
- `voynich_semantic_notation_v1.rf_edit1_pairs`
- `voynich_semantic_notation_v1.line_edit_metrics_v1`

Important edit-pair audit:
- `rf_edit1_pairs` rows: 28,435;
- distinct unordered token pairs: 27,307;
- duplicate edit paths: 1,128.

The section/line hierarchy uses distinct unordered pairs, not raw edit-path rows.

## D. Source vs implementation vs inference

### Source-derived
- Matteo uses an artificial-word construction based on first syllables.
- Matteo belongs to the early-fifteenth-century Paduan ars-memorativa context.
- Bartolomeo and Ragona provide related structured encoding precedents.
- Vat. lat. 10488 is a 1424 Venetian mathematical manuscript.

### Our implementation choices
- choice of PyWORDS vocabulary;
- Latin normalization rules;
- first-syllable orthographic parser;
- exact K2/K3/K5/KMIX sampling procedures;
- unique-type target sizes;
- edit-distance algorithm and ambiguity handling;
- entropy metrics;
- section-matched simulation sizes;
- line-shuffle null construction.

### Our statistical findings
- K2 produces an aggregate edit-location profile close to whole-corpus RF and to some sections;
- matched section vocabularies are all more one-edit-dense than K2;
- one-edit neighbours are concentrated within ordinary running-text lines in multiple sections after section-conditioned line shuffling;
- Voynich retains stronger right-edge and transition constraints than K2.

### Inference only
The working interpretation that Voynich may involve a productive compositional topology plus domain/local gating is an **inference from the statistical pattern**, not something stated by the historical sources.

## E. Publication threshold

Before any paper or public claim that relies materially on Matteo/Bartolomeo/Ragona:

1. replace exploratory mirror access with stable critical-edition/manuscript citations;
2. give exact page/folio references for every historical encoding rule;
3. retain the distinction between source wording and VSN abstraction;
4. rerun the public reproduction scripts from a clean environment;
5. archive raw JSON/CSV outputs with hashes;
6. obtain an independent code review of the edit-pair and line-shuffle logic.