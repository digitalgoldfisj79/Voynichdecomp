# Tranchedino × STA v2.4 — historical template freeze: Cod.2398 f.134v–135r

Date frozen: 2026-08-09
Status: **PRIMARY-SOURCE TEMPLATE EXTRACTION — NO VOYNICH FIT AUTHORISED**
Parent programme: v2.3 closure `f5a8b40f26fee17f7d6aff9e17254b8f40528a79`

## 1. Source and system boundary

Primary source: ÖNB Cod.2398, local facsimile `Tranchedino Cipher Ledger (1).pdf`.

Local facsimile SHA-256:
`4e16dace306ba251239e634d840f0ba6b46c1479aa5affdc979d68caf83ae031`.

The template is the single cipher system beginning on **f.134v** (PDF page 285) and continuing on **f.135r** (PDF page 286). PDF page 287 / f.135v begins a visibly new system with a new heading and is excluded.

Rendered-page SHA-256 values used for this manual visual extraction:

- p285 / f.134v: `40f1551f1c710559a0d15523b1bb2ab3367b9968653b52f278f5a6e2fab891bd`;
- p286 / f.135r: `da6795ca8c6f86826a9f4d8d3744d15f95297a1850023d0bd7d4ef4501782f34`;
- p287 / new-system boundary check: `a930a8e074568cfb3a935d6a2871886a6e2ec4717a2c9a886ba49ab6a7eed8f4`.

The machine-assisted catalogue was used only to locate the candidate pages. Counts and inventories below were read directly from the facsimile. Cipher-glyph identities are deliberately not transcribed because the planned calibration requires class geometry and plaintext-unit inventory, not graphic sign equivalence.

## 2. Alphabetic block

The f.134v header gives 23 plaintext columns:

`a b c d e f g h i l m n o p q r s t u x y z &`

Direct row geometry is regular:

- every column has two cipher signs;
- the five vowel columns `a e i o u` have a third sign.

Therefore the complete key-sheet alphabet block contains:

- 23 plaintext columns;
- **51 cipher signs** = `23×2 + 5`.

Under the already frozen 19-letter Paduan normalisation
`abcdefghilmnopqrstu`, the columns `x y z &` are outside the literal alphabetic source model. The compatible alphabetic subset is therefore:

- 19 plaintext letters;
- **43 cipher signs** = `19×2 + 5`.

This is a historical-template count, not a claim that all 43 signs must be observed in a finite ciphertext.

## 3. Null block

The explicit `Nulle` row on f.134v contains **7 visually distinct cipher signs**.

No semantics beyond `null` are assigned to them.

## 4. Geminate block

The explicit doubled-letter row on f.134v contains **8 one-sign plaintext units**:

`bb cc ff mm nn rr tt ss`

Each is paired with one cipher sign.

## 5. Syllabary

The syllabary begins at the lower right of f.134v and continues on f.135r. The plaintext-unit inventory is visually unambiguous and contains **64 one-sign CV units**:

- f.134v: `ba be bi bo bu`; `ca ce ci co cu`; `da de di do du`;
- f.135r: `fa fe fi fo fu`; `ga ge gi go gu`; `la le li lo lu`; `ma me mi mo mu`; `na ne ni no nu`; `pa pe pi po pu`;
- f.135r: `qua que qui quo`;
- f.135r: `ra re ri ro ru`; `sa se si so su`; `ta te ti to tu`.

Count check: `12×5 + 4 = 64`.

Every listed plaintext unit is paired with exactly one visible cipher sign. This is critical: **the historical key does not require discovering boundaries between one-, two-, and three-symbol ciphertext code groups. One visible historical sign is one cipher event; its plaintext expansion may be one or more letters.**

## 6. Lexical / nomenclator block

f.134v contains **44 one-sign lexical or nomenclator rows** before the syllabary:

- **32** rows in the left diplomatic/nomenclator list, ending with `Fanti`;
- **12** rows in the right list above `ba`.

The right list visibly includes high-confidence entries such as `Dinari`, `Galee`, `Nave`, `Grippo`, `che`, `Perche`, `como`, `unde`, and `quando`; several intervening abbreviated phrases are palaeographically less secure.

Binding rule for v2.4: the **count 44 is frozen**, but uncertain lexical strings are not guessed. A later generator may use only separately transcribed high-confidence lexical entries or may treat the 44 slots as latent/unobserved nomenclator capacity. Any semantic transcription beyond this record requires a prospective amendment before calibration.

## 7. Complete slot counts

Direct key-sheet count:

| class | surface signs |
|---|---:|
| alphabetic | 51 |
| null | 7 |
| geminate | 8 |
| syllabic | 64 |
| lexical/nomenclator | 44 |
| **total** | **174** |

If the four literal alphabet columns `x y z &` and their two signs each are removed under the frozen 19-letter Paduan alphabet, the compatible slot count is:

`43 + 7 + 8 + 64 + 44 = 166`.

This exact **166** is recorded descriptively because RF1b also has 166 observed full-STA member types. It receives **zero evidential weight**: the equality was noticed only after both quantities existed, and finite-cipher occupancy can be far below key-sheet capacity.

## 8. Relation to the old terminal Family S

This historical template is **not a rerun of terminal Family S**.

Terminal Family S represented variable-length **one-, two-, and three-symbol visible ciphertext groups** with a fixed 63-unit plaintext inventory. Its core difficulty was discovering the visible-group segmentation; the final lattice overfragmented every trial and the family closed at development.

The f.134v–135r mechanism has the opposite geometry at the relevant interface:

- each historical cipher sign is already one visible event;
- a sign may expand to a plaintext letter, geminate, CV syllable, null, or lexical item;
- there is no hidden 1–3-sign ciphertext grouping boundary in the historical key-sheet mechanism.

Accordingly v2.4 may test a **one-sign / variable-plaintext-expansion historical syllabary** without reopening the closed variable-visible-group Family S search.

## 9. What is and is not frozen

Frozen from primary source:

- f.134v–135r system boundary;
- 51/43 alphabetic geometry;
- 7 null signs;
- 8 geminate units and their labels;
- exact 64-unit syllabary;
- 44 lexical/nomenclator slots;
- total K174 key sheet / K166 19-letter-compatible slot count.

Not yet frozen:

- how often a historical correspondent chose a syllabic sign instead of spelling the same letters alphabetically;
- how often nulls were inserted;
- semantic identities of palaeographically uncertain lexical rows;
- active-sign occupancy at a specified plaintext length;
- any mapping to RF/full-STA or connected-aaa signs;
- any Voynich score or plaintext.

Those are Stage B0 calibration questions, not facts to be inferred from this key sheet.
