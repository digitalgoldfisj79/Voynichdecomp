## S6a.2: Middle High German pharmaceutical text

### Rationale

MHG pharmaceutical text represents the strongest non-Latin alternative within the syllabic cipher class. German medical recipe literature shares the same formulaic structure as Latin pharmaceutical prose (verb-noun-conjunction-noun-preposition-noun), and MHG pharmaceutical corpora are available for cross-validation.

### Sonorant concentration

The 93% sonorant-final concentration in VMS tokens is a property of the cipher architecture, not a language discriminant. Under the grid, suffix slots carry the first-vowel column, and all suffix families end in sonorant characters by construction (y, aiin, ol, or, am, ∅). Any language enciphered through this grid would produce similar sonorant concentrations. The sonorant test constrains the cipher class (syllabic, not character-level) and confirms the grid architecture.

### Bigram cross-validation

The function-word bigram optimiser achieves comparable cross-validation on both Latin (r = 0.89, CI 52,004 words) and MHG (r = 0.88, Admonter Arzneibuch) pharmaceutical corpora. This indicates the bigram test identifies the genre (pharmaceutical recipe text with formulaic function-word sequencing) rather than the specific language. The bigram test validates the cipher architecture and constrains the source register to pharmaceutical recipe prose. It does not by itself discriminate between Latin and MHG.

### Consonant row distribution

The consonant test discriminates where the bigram test cannot. Under the grid architecture, FC tokens encode the source word's initial consonant in the core slot. The m_core first character maps to one of seven consonant rows, and this mapping has no parameters adjusted at the language-discrimination stage: the row assignments are fixed by the PGCS architecture (Bozzard 2026a) before any language testing.

**Table S6a.2a.** FC consonant row distribution: VMS Herbal-A, CI pharmaceutical Latin, and MHG pharmaceutical text (combined).

| Row | Consonants | VMS (n=1,621) | CI Latin (n=24,300) | MHG (n=168,266) |
|-----|-----------|---------------|---------------------|-----------------|
| o | c,s,p | 35.4% | 28.1% | 14.7% |
| c | ∅,v | 28.9% | 26.4% | 31.9% |
| e | f,d | 15.2% | 11.1% | 17.3% |
| a | m,l | 9.3% | 11.9% | 8.4% |
| d | r,q,h,n,g | 6.0% | 15.9% | 13.0% |
| l | t | 3.9% | 4.2% | 3.2% |
| r | b,z,x,j,k,w,y | 1.2% | 2.3% | 11.6% |

χ² (VMS vs CI Latin) = 0.025.
χ² (VMS vs MHG) = 1.108.

**Table S6a.2b.** MHG results by individual corpus.

| Corpus | Words | Row r% | χ² | Shape distance |
|--------|------:|-------:|---:|-----------:|
| Admonter Bartholomäus | 27,128 | 10.6% | 0.944 | 0.131 |
| Ortloff | 46,350 | 12.4% | 1.278 | 0.173 |
| Breslauer Arzneibuch | 92,076 | 11.4% | 1.095 | 0.133 |
| Kochrezeptsammlung | 2,712 | 13.5% | 1.491 | 0.140 |
| **Combined** | **168,266** | **11.6%** | **1.108** | **0.147** |

All four corpora produce row 'r' above 10%, confirming that the result is not driven by a single text.

### The row 'r' discriminant

Row 'r' contains consonants b, z, x, j, k, w, y. The VMS places 1.2% of FC tokens in this row (20/1,621). Latin pharmaceutical vocabulary rarely begins with these consonants: b-initial words are limited to balneum, borago; z-, w-, x-, j-, k-initial words are essentially absent from pharmaceutical Latin.

MHG pharmaceutical vocabulary is heavily loaded with row-'r' consonants:

- **w-initial** (5.3%): wasser, wurz, wyn, wermut, wegerich, weich, warm, waschen
- **z-initial** (2.0%): ze, zucker, zitwar, zimt, zerstoßen, zerreiben
- **b-initial** (2.4%): blat, bluot, bein, blüte, balsam, brennen, bitter, brot
- **k-initial** (1.5%): krut, kalt, kraft, kochen, kern, krank

These are core pharmaceutical vocabulary, appearing in virtually every MHG recipe text. w-initial alone (5.3%) exceeds the entire VMS row 'r' (1.2%).

### What drives the mismatch

Three rows account for the structural incompatibility:

1. **Row 'r' is 10× too large in MHG.** MHG's common b/w/k/z-initial words have no parallel in the VMS's rare-consonant row.
2. **Row 'o' is 2.4× too small in MHG.** MHG lacks Latin's c/s-initial word dominance (cum, contra, cortice, semen, sal, si, sed).
3. **Row 'd' is 2.2× too large in MHG.** g (5.0%) and h (3.7%) are frequent MHG initials (gut, ger, her, hin) but rare in the VMS's row 'd'.

### Bootstrap significance test

10,000 bootstrap iterations resampling the VMS FC tokens (n=1,621 with replacement) and computing χ² against both CI Latin and MHG combined:

- CI wins 10,000 out of 10,000 iterations
- CI worst case: χ² = 0.045
- MHG best case: χ² = 0.364
- The 95% confidence intervals do not overlap
- p < 0.0001

### Consonant extraction method

All MHG results use first-letter extraction: the initial character of each word determines the consonant row. An earlier version of the extraction script (row_r_test.py) used last-consonant-before-first-vowel, which biases against cluster-heavy languages. The fixed first-letter method was verified against all four corpora. The corrected script is archived at GitHub.

### Verification protocol

The results are reproducible by any researcher with access to MHG pharmaceutical corpora:

1. Load enriched_records.pkl (archived at Zenodo).
2. Extract FC tokens from Herbal-A (section == 'Herbal-A', empty_core == False).
3. Count m_core first characters to obtain the 7-row distribution (Table S6a.2a).
4. Parse the MHG corpus. Map each word's first letter to the 7-row system using CONSONANT_TO_ROW.
5. Compare distributions. Compute χ².

The row mapping is fixed by the architecture:

```python
CONSONANT_TO_ROW = {
    'c': 'o', 's': 'o', 'p': 'o',
    '': 'c', 'v': 'c',
    'f': 'e', 'd': 'e',
    'm': 'a', 'l': 'a',
    'r': 'd', 'q': 'd', 'h': 'd', 'n': 'd', 'g': 'd',
    't': 'l',
    'b': 'r', 'z': 'r', 'x': 'r', 'j': 'r', 'k': 'r', 'w': 'r', 'y': 'r',
}
```

Alternatively, use `row_r_test.py` (archived at GitHub) with any word list as input.

### Summary

| Metric | Latin (CI) | MHG (combined) | Ratio |
|--------|-----------|----------------|-------|
| Row 'r' | 2.3% | 11.6% | 5.0× worse |
| χ² vs VMS | 0.025 | 1.108 | 44× worse |
| Shape distance | 0.009 | 0.147 | 16× worse |
| Bootstrap | wins 10,000/10,000 | wins 0/10,000 | p < 0.0001 |

MHG pharmaceutical text is decisively excluded as the FC source language under the grid architecture. The consonant row distribution discriminates where the bigram cross-validation cannot. Source-language identification rests on the consonant distribution, not on the bigram cross-validation or the sonorant concentration.

### Source data

- VMS: enriched_records.pkl (37,465 tokens), Herbal-A FC subset (n=1,621)
- Latin: ci_corpus_parsed.pkl (52,004 tokens), FC subset (n=24,300)
- MHG: 168,266 words across four pharmaceutical corpora (Ortloff, Admonter Bartholomäus, Breslauer Arzneibuch, Kochrezeptsammlung), provided by J. Berger. First-letter extraction, all words included (Latin passages retained per source provider's decision).
