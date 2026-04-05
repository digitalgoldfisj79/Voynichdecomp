# Voynich Manuscript: Two-Table Cipher Architecture

**A function-word nomenclator paired with a consonant-vowel syllabic grid, tested against pharmaceutical Latin**

Edward Bozzard · ORCID: [0009-0002-4052-0994](https://orcid.org/0009-0002-4052-0994)

---

## What this is

This repository contains the data, code, and supplements for a testable hypothesis about the Voynich Manuscript (Beinecke MS 408). The hypothesis: the manuscript was produced by a two-table cipher — a nomenclator encoding function words paired with a syllabic grid encoding content words — enciphering pharmaceutical recipe text, most likely in Latin.

The architecture is the standard cipher class of the fifteenth-century Northern Italian diplomatic tradition, documented in 225 keys from the Sforza chancery register (1450–1485).

**What the model does:**
- Reproduces 63/84 distributional metrics across all nine manuscript sections
- Cross-validates function-word assignments on 52,000 words of unseen pharmaceutical Latin (r = 0.89)
- Identifies pharmaceutical recipe genre via bigram correlation
- Discriminates Latin from Middle High German via consonant distribution (χ² = 0.025 vs 1.108, p < 0.0001)

**What the model does not do:**
- Recover the cipher key
- Produce readings
- Identify the source manuscript
- Make plant-level predictions on individual folios

---

## Core data files

| File | Description | Tokens |
|------|-------------|--------|
| `enriched_records.pkl` | 37,465 VMS tokens, PGCS-decomposed, with per-token metadata (section, folio, prefix, gallows, core, suffix, m_core, suffix family, EC/FC class) | 37,465 |
| `p70c_full_layer.pkl` | Zodiac folio analysis layer | — |
| `ci_corpus_parsed.pkl` | Circa Instans pharmaceutical Latin (Wellcome MS 624), parsed with EC/FC classification | 52,004 |

**Transliteration source:** Zandbergen–Landini (ZL) file, version 2b (IVTFF 1.7), obtained via [OrcusLabs/voynich.science](https://github.com/OrcusLabs/voynich.science). SHA256: `c7ffff9e1f3ecbec174e234c04f056b2bec14f8d722726c456f108e2c7060db5`. Original at [voynich.nu/data/](https://www.voynich.nu/data/).

---

## Key fields in enriched_records.pkl

```python
import pickle
with open('enriched_records.pkl', 'rb') as f:
    records = pickle.load(f)

# Each record is a dict:
# {
#   'token':     'daiin',          # raw EVA token
#   'section':   'Herbal-A',       # full section name (9 sections)
#   'folio':     'f2r',            # folio identifier
#   'prefix':    'd',              # PGCS prefix slot
#   'gallows':   '∅',              # PGCS gallows slot (use ∅, never NULL)
#   'core':      '',               # PGCS core slot (empty for EC tokens)
#   'm_core':    '',               # modified core (use for grid row)
#   'suffix':    'aiin',           # PGCS suffix slot
#   'sfx_fam':   'N',              # suffix family (Y, N, L, R, BARE, M)
#   'empty_core': True,            # EC (True) or FC (False)
# }
```

**Section names (always use full names):**
Herbal-A, Herbal-B, Pharmaceutical, Balneological, Astronomical, Cosmological, Biological, Zodiac, Stars

**Critical conventions:**
- Use `∅` (UTF-8 empty set symbol), never `NULL` or `None`, for empty PGCS slots
- Use `m_core` field for grid row assignment (first character = row)
- Use `core` field for raw positional analysis
- EC/FC threshold: 53%/47% (frequency-based proxy)

---

## Supplements

| ID | Title | File(s) |
|----|-------|---------|
| S1 | Forward cipher v11 | `v11_nomenclator.py` |
| S2 | Nomenclator optimiser + null distribution | `nomenclator_optimizer.py` |
| S3 | CV syllable reader | `cv_folio_reader.py` |
| S4 | Clean baseline forward cipher | `forward_cipher_v11_CLEAN.py` |
| S5 | 84-metric scoring battery | `score_85_metrics.py` |
| S6 | Language discrimination battery (16 languages) | `S6_language_battery_results.json`, `S6_language_discrimination_battery.md` |
| S6a | Greek + MHG extended language tests | `S6a_greek_language_discrimination.md`, `S6a_update_mhg.md` |
| S7 | EVA robustness tests | — |
| S8 | Leave-one-out analysis | — |
| S9 | Folio-level CV enrichment results | — |
| S10 | Negative results + killed hypotheses + gallows×house analysis | `S10_extension_gallows_house.md` |
| S11 | BG42 comparison data | — |
| S12 | Sforza cancelleria cipher catalogue (225 keys) | CSVs at Zenodo |
| S13 | Corpus reversal test | `S13_corpus_reversal_test.py` |

---

## Reproducing key results

### 1. VMS consonant row distribution (§5.3)
```python
import pickle
from collections import Counter

records = pickle.load(open('enriched_records.pkl', 'rb'))
ha_fc = [r for r in records
         if r['section'] == 'Herbal-A' and not r['empty_core']]

rows = Counter(r['m_core'][0] for r in ha_fc if r.get('m_core') and r['m_core'][0] in 'oceadlr')
total = sum(rows.values())

for row in ['o','c','e','a','d','l','r']:
    print(f"  {row}: {rows.get(row,0)/total*100:.1f}%")
# Expected: o=35.4%, c=28.9%, e=15.2%, a=9.3%, d=6.0%, l=3.9%, r=1.2%
```

### 2. Row 'r' language test
```python
# Row 'r' contains consonants b, z, x, j, k, w, y
# VMS: 1.2% | Latin CI: 2.3% | MHG (168k words): 11.6%
# To test your own language: count FC words starting with b,z,w,k,x,j,y
# Divide by total FC words. If >5%, your language is unlikely.
```

### 3. EC/FC split
```python
ec = [r for r in records if r['empty_core']]
fc = [r for r in records if not r['empty_core']]
print(f"EC: {len(ec)} ({len(ec)/len(records)*100:.1f}%)")
print(f"FC: {len(fc)} ({len(fc)/len(records)*100:.1f}%)")
# Expected: EC ~52.7%, FC ~47.3%
```

### 4. Test your own source language

Use `row_r_test.py` with any word list:
```bash
python3 row_r_test.py your_wordlist.txt
```
Outputs three numbers: row-r percentage, χ² under Latin grouping, shape distance. Compare to VMS reference values.

---

## Consonant-to-row mapping (fixed by architecture)

```python
CONSONANT_TO_ROW = {
    'c': 'o', 's': 'o', 'p': 'o',
    '': 'c', 'v': 'c',           # vowel-initial → row 'c'
    'f': 'e', 'd': 'e',
    'm': 'a', 'l': 'a',
    'r': 'd', 'q': 'd', 'h': 'd', 'n': 'd', 'g': 'd',
    't': 'l',
    'b': 'r', 'z': 'r', 'x': 'r', 'j': 'r', 'k': 'r', 'w': 'r', 'y': 'r',
}
```

---

## What has been tested and rejected

Full details in Supplement S10. Summary:

- CI as the specific source text (1/11 consonant rows match)
- Per-section keywords (278 tested, all score 51–53/84)
- Ring model for EC equivalences (transitivity 0.31–0.38)
- Grabadin vocabulary mapping (0/5 matches)
- Gallows-aware generation v12 (no improvement)
- Character-level copy-mutate (σ remains ~2.4)
- Naibbe verbose homophonic cipher (Greshko 2025; reproduces entropy, lacks CV structure)
- Middle High German as source language (row 'r' = 11.6%, p < 0.0001)
- Blind predictions at plant level (insufficient resolution at n = 14–89 tokens per folio)

---

## Known limitations

1. **Forward score circularity.** The forward model draws tokens from VMS pools, reproducing VMS vocabulary with VMS vocabulary. The non-circular evidence is the cross-validated bigram correlation on external Latin text.

2. **Greedy search inflation.** The nomenclator optimiser's greedy search over 23 candidates inflates r. Search-inclusive p = 0.0003 (random) or 0.077 (greedy). The greedy test is the more appropriate comparator.

3. **One genuine function-word recovery.** Only et→Y is independently recovered and dominates the bigram improvement. in→N agrees with the vowel heuristic (not an independent recovery). The remaining assignments are register-dependent.

4. **Two languages tested at corpus level.** Latin and MHG have been tested on parsed pharmaceutical corpora. Italian, Catalan, Occitan, and Portuguese have been tested only with estimated consonant distributions (S6).

5. **No readings.** The architecture identifies cipher class and genre. It does not recover the key, the keyword, or the source text.

---

## How to use this data

**If you have a cipher hypothesis:** Run it through `score_85_metrics.py`. The 84-metric battery tests word length, character frequency, bigram structure, positional patterns, and cross-section transfer. Report the score. Compare to v11's 63/84.

**If you have a source language hypothesis:** Run `row_r_test.py` on your pharmaceutical corpus. Three numbers come back. Compare to VMS reference values.

**If you have a source text candidate:** Parse it, classify words as EC/FC using the 53/47 threshold, run the nomenclator optimiser, cross-validate on a held-out portion. Report r.

**If you want to test the PGCS grammar:** Every token in `enriched_records.pkl` is already decomposed. The `empty_core` field classifies EC/FC. The suffix family field gives the family assignment. Build from there.

---

## Citation

```
Bozzard, E. (2026). A two-table cipher architecture for Beinecke MS 408:
Function-word assignments from cross-validated pharmaceutical Latin.
Data and code: https://github.com/digitalgoldfisj79/Voynichdecomp
Zenodo: DOI 10.5281/zenodo.18812705
```

---

## Contact

Edward Bozzard · [ORCID](https://orcid.org/0009-0002-4052-0994) · GitHub: [@digitalgoldfisj79](https://github.com/digitalgoldfisj79)

Voynich Ninja forum: DG97EEB
