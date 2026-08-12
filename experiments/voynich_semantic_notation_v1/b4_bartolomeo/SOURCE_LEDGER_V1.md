# VSN-B4-v1 Source / Evidence Ledger

Frozen source phase started 2026-08-12.

## P1 — Paris primary witness

- Shelfmark: Paris, Bibliothèque nationale de France, Latin 8684.
- Gallica ARK: `https://gallica.bnf.fr/ark:/12148/btv1b520004421`
- Catalogue identification independently found as `Bartholomaeus Minorita, Liber de memoria artificiali – S. XV`.
- Analytical folios supplied directly by the user from Gallica: 7r, 7v, 8r, 8v.
- f.9r reported blank by direct inspection.
- The four local images are frozen by SHA256 in `PRIMARY_SOURCE_MANIFEST.csv`.

### Primary-witness observations currently secure enough to use

`OBS-P1-01` f.8v contains repeated small bracketed groups linking sets of source names to compact written forms. **Source-derived visual observation.**

`OBS-P1-02` the first group is consistent with the Tripode/Pepo/Corvus/Vetula example reported by Cacopardo. Exact diplomatic readings beyond clearly legible words are not yet frozen. **Primary observation cross-supported by secondary source.**

`OBS-P1-03` f.8v contains larger/nested braces in addition to the immediate four-name brackets. Their graphical existence is source-derived; their semantic interpretation is UNRESOLVED. **Do not call these recursive codewords without further textual evidence.**

`OBS-P1-04` f.7v–8r contains a section whose heading is provisionally read `De numeris ficticiis`; the pages visibly organise numerical forms/grades and bodily/spatial instructions. Exact diplomatic transcription is pending. **Primary visual observation, heading currently B-confidence until independently checked.**

`OBS-P1-05` f.8v is the final nonblank textual folio in the supplied sequence because f.9r is reported blank. **User inspection; not independently re-fetched by the programme.**

## S1 — Cacopardo 2021 thesis

Valentina Cacopardo, *Memory and Imagination in the Ars Memorativa in Fifteenth-Century Italy*, PhD thesis, School of Advanced Study, 2021.

Landing page: `https://sas-space.sas.ac.uk/9846/`
PDF: `https://sas-space.sas.ac.uk/9846/1/VCacopardoMemoryandImaginationPhDthesis.pdf`

Verified from the repository metadata:
- thesis completed 30 April 2021;
- Part Two examines Bartolomeo da Mantova's *Liber memoriae artificialis* (1429);
- the thesis describes the work as including one hundred illuminations.

### S1-A — architectural inventory and directional rule

Verified from parsed PDF pp. 131–135:
- Bartolomeo supplies ten architectural loci;
- each architectural locus has ten loci-objects, yielding one hundred loci-objects overall;
- three imagines are placed on each locus-object, yielding three hundred imagines;
- Cacopardo states that these three imagines belong to flora, fauna and human-figure categories;
- the images are placed horizontally left-to-right as **human figure – fauna – flora**;
- Bartolomeo's instruction is quoted as requiring recitation in the opposite direction, **right-to-left**.

The last point is a source-relevant positional constraint and is frozen before any new Voynich comparison. It is not interpreted as a Voynich analogue at this stage.

The thesis also supplies Latin lists of the ten loci-objects for each architectural locus. For example, the first architectural locus begins `Tripode, Mensa, Mantile, Phyala, Ciphus aureus...`; the second begins `Lectum extensum, Cervical vergatum, Linteamina munda...`. These secondary transcriptions may be used as reading aids but may not overwrite a conflicting primary-witness reading.

### S1-B — syllabic codewords

Verified from parsed PDF pp. 135–136:
- at the end of Bartolomeo's text, before the illuminations, the list is written again with shorter code words;
- for the four figures on each table, first syllables of the four names are strung together to form a four-syllable artificial name;
- the first example is Tripode + Pepo + Corvus + Vetula mancina -> `TRI PE COR VE`;
- the thesis states that `four hundred words` could be reduced to `twenty codewords`, each of four syllables;
- the thesis says this instruction is present in both the Paris and Vatican illuminated manuscripts and the compound-syllable groups are highlighted by squared brackets.

### Unresolved secondary-source arithmetic

Cacopardo's `400 words -> 20 codewords` statement is recorded exactly as a secondary claim. It is not reconciled by assumption. The primary witness must determine how many immediate four-name groups are actually present and whether the displayed page represents a subset, a hierarchical compression, a mnemonic architectural unit, or something else.

## S2 — BnF/Gallica infrastructure

The BnF describes Gallica as its digital library and documents an IIIF image-retrieval API for high-definition image access. A direct automated HD fetch was attempted during stage 0 but container DNS access to Gallica was unavailable. The user-supplied 1024x1387 images therefore remain the frozen stage-1 images.

## PDF inspection incident

The programme obeyed the requirement to attempt visual PDF inspection of the Cacopardo source. `web.run` screenshot calls for PDF pages 134 and 135 failed with an internal/cache-miss error. Parsed PDF text was available and is used only for the explicitly logged secondary claims above. No visual feature of the thesis PDF is treated as evidence.

## Evidence classes

- `PRIMARY`: directly visible in BnF Latin 8684 images.
- `SECONDARY`: stated by Cacopardo or catalogue scholarship.
- `INFERENCE`: programme interpretation from primary/secondary facts; never silently promoted to source fact.
- `UNRESOLVED`: unread or conflicting; excluded from primary metric corpus until resolved.

## Explicitly withdrawn/prevented claims

1. Larger f.8v braces are **not** currently classified as recursive codeword generation.
2. `20` is **not** currently accepted as the total Bartolomeo codeword inventory size.
3. `De numeris ficticiis` is **not** merged with the syllabic codeword system into a single historical grammar.
4. The attested right-to-left recitation rule is **not** treated as evidence of Voynich right-edge morphology until the target gate is legitimately opened.
