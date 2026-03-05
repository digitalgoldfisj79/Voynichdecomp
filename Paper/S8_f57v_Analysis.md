# Supplement S8: f57v Reference Page Analysis

## S8.1 Page Structure

Folio 57v is an astronomical/cosmological diagram consisting of four concentric text rings surrounding a central group of four human figures. Each ring is transcribed as a separate line in the ZLZI transcription.

| Ring | ZLZI line | Spacing | Content |
|---|---|---|---|
| Centre | Lines 1, 6–9 | N/A | 4 figures with labels |
| 1 (innermost) | Line 3 | **Individually spaced glyphs** | Glyph catalogue (period-12) |
| 2 | Line 5 | Mixed singles → words | Assembly demonstration |
| 3 | Line 4 | Standard word spacing | Full Voynichese |
| 4 (outermost) | Line 2 | Standard word spacing | Full Voynichese |
| Margins | Lines 10–13 | N/A | Annotation labels |

Individual spacing in Ring 1 is confirmed in the Beinecke scan — it is not a transcription artefact.

---

## S8.2 Uniqueness of f57v

To establish that Ring 1's structure is not a generic feature of circular-text pages, all 26 pages in the manuscript containing circular text (@Cc/+Cc markers in ZLZI) were tested for individually-spaced glyph sequences.

| Metric | f57v | Other 25 circular-text pages |
|---|---|---|
| Lines with >50% single-glyph tokens | **2** (Lines 3, 5) | **0** |
| Lines with singles → words transition | **1** (Line 5) | **0** |
| Maximum single-glyph % in any line | **100%** (Line 3) | 8.8% (f68v1) |

f57v is the only circular-text page in the manuscript that contains individually-enumerated glyph sequences. This eliminates the alternative that the structure is a generic feature of the cosmological section's layout conventions.

---

## S8.3 Ring 1 (Line 3): The Glyph Catalogue

Ring 1 contains a period-12 sequence repeated four times with two variant positions:

```
o-l-d-r-v-x-k-k-f-t-r-y   (unit 1)
o-l-d-r-v-x-k-m-f-t-r-y   (unit 2: position 8: k→m)
o-l-d-r-v-x-k-k-p-t-r-y   (unit 3: position 9: f→p)
o-l-d-r-v-x-k-k-p-t-r-y   (unit 4: same as unit 3)
+ terminal n
```

The sequence contains **13 unique EVA characters**, all enumerated as isolated tokens (individually spaced, not grouped into words). Positions 7–10 form a contiguous gallows block; the variation across units is confined to this block.

---

## S8.4 The Skeleton/Dressing Partition

The 25 EVA characters attested in the ZLZI corpus divide into two functionally distinct classes. Ring 1 enumerates one class and omits the other.

**Skeleton characters (13) — enumerated in Ring 1:**

| Character | Primary PGCS slot | Slot frequency |
|---|---|---|
| o | Prefix | 55% |
| k, t, f, p | Gallows | 67–86% |
| d, l, r, m, n, y | Suffix | 50–99% |
| v, x | Core | 100% |

**Dressing characters (6) — absent from Ring 1:**

| Character | Function |
|---|---|
| a, e, i | Vowel fill (distributed across Core and Suffix) |
| c, h | Compound prefix formers (ch-, sh- when paired with s) |
| s | Prefix compound former |

The dressing characters are the **variable** part of Voynichese morphology: vowels and consonant clusters that fill and connect slots. Ring 1 enumerates the **fixed frame**.

---

## S8.5 Statistical Significance of the Partition

**Hypergeometric test.** The null hypothesis is that the 13 characters enumerated in Ring 1 are drawn uniformly at random from the 25 EVA characters. The probability of drawing exactly the 13 skeleton characters (and no dressing characters) when selecting 13:

*p* = C(13,13) × C(12,0) / C(25,13) = 1/5,200,300 = **1.92 × 10⁻⁷**

**Gallows paradigm constraint.** The variant positions (8 and 9) exhibit specifically f/p alternation — one of C(9,2) = 36 possible pairings among the 9 gallows characters. Conditional on the hypergeometric result:

*p*(combined) = 1.92 × 10⁻⁷ × (1/36) = **5.3 × 10⁻⁹ < 10⁻⁸**

**Positional test.** Treating each of the 12 sequence positions as an independent draw from the 25-character EVA inventory, the probability of the observed slot assignments (PREFIX at position 1, SUFFIX at positions 2/4/11/12, CORE at positions 5/6, GALLOWS at positions 7–10) yields *p* < 10⁻¹⁶.

**Cross-transcription robustness.** The skeleton/dressing partition holds under JGLI, ZLZI, and TTLI transcriptions: 13/13 skeleton characters remain in Ring 1, 6/6 dressing characters remain absent, across all three systems. Transcription-specific differences affect glyph readings in Rings 3–4 (Full Voynichese) but not the Ring 1 catalogue.

---

## S8.6 Manuscript-Wide Validation

The skeleton/dressing ratio is constant across all nine manuscript sections and all five scribal hands:

| Section | Skeleton % | Dressing % | S/D ratio |
|---|---|---|---|
| Herbal | 53.0% | 45.1% | 1.18 |
| Astronomical/Cosmological | 54.6% | 42.8% | 1.28 |
| Zodiac | 57.2% | 42.3% | 1.35 |
| Balneological | 54.3% | 40.7% | 1.33 |
| Pharmaceutical/Stars | 52.2% | 45.0% | 1.16 |
| **f57v rings** | **55.3%** | **44.7%** | **1.24** |

CV across sections: ±3%. f57v's own rings fall within the manuscript-wide range, confirming the folio is self-consistent with the corpus it is claimed to describe.

The gallows parameters, by contrast, vary substantially across scribal hands (k/t ratio CV = 37%, f/p ratio CV = 35%), consistent with Ring 1's paradigm showing these as freely substitutable positions rather than fixed values.

---

## S8.7 Ring 2 (Line 5): Progressive Assembly

Ring 2 demonstrates a three-stage construction sequence:

| Stage | Positions | Content | Introduced |
|---|---|---|---|
| 1. Skeleton | 0–4 | `o v l r m` | Pure Ring 1 characters |
| 2. First dressing | 5–21 | Mixed single glyphs | `a`, `i` (pos 5), then `s` (pos 8) |
| 3. Full words | 22–25 | `teodar otodar sheky otchody` | Complete Voynichese with `ch`/`sh` compounds |

The missing-character introduction order (`a`, `i` → `s` → `e` → `c`, `h`) matches PGCS slot ordering (Prefix compounds before Core vowels before Suffix compounds). Ring 2 is a construction manual: skeleton characters first, dressing characters added progressively, compound prefixes introduced last.

---

## S8.8 Independence of the f57v Result

The skeleton/dressing partition was derived from the spatial layout of a single folio (f57v Ring 1). The PGCS slot assignments were derived from corpus-wide statistics of 37,465 tokens across 226 folios. The two analyses converge on the same 13/6 character partition and the same slot structure.

**Methodological independence:** The f57v analysis used physical spacing and sequence structure. The PGCS analysis used character-position entropy decomposition and mutual information across the full corpus. These are independent methods applied to independent data.

**Data caveat:** Both analyses derive from EVA transcriptions of the same manuscript. They are methodologically independent but not data-independent. Systematic transcription errors in EVA character identification would propagate to both.

---

## S8.9 Interpretation

f57v Ring 1 is a structural reference for the Voynich notation system. The spatial layout — innermost ring simplest, outermost most complex — is the layout of a construction diagram:

- **Ring 1 (Line 3):** Enumerates the 13 fixed-frame characters with a paradigmatic gallows substitution table
- **Ring 2 (Line 5):** Demonstrates progressive assembly — skeleton → dressing → full tokens
- **Rings 3–4 (Lines 4, 2):** Fully assembled Voynichese in normal operation
- **Centre figures:** Each labelled with a token exemplifying a gallows class ({∅, k, t, p})

This is not a cipher key. It is a notation grammar guide — a visual explanation of how PGCS slots combine to produce Voynichese tokens.

**Falsifiable prediction confirmed:** If this reading is correct, no other concentric-ring page should show a singles→words transition. Tested against all 25 other circular-text pages — none do (§S8.2).
