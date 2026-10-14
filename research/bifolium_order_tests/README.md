# Bifolium and ordering tests on the Voynich Manuscript text

Code and inputs for the tests posted on voynich.ninja (threads 6182 and 6199). Everything runs from this folder
with Python 3.10+ and the four packages in `requirements.txt`; no network access is needed except for the
optional 12-transliteration check.

```
pip install -r requirements.txt
./run_all.sh            # or: SLIM=1 ./run_all.sh   to add the 12-transliteration check (downloads one JSON file)
```

Results are written as `results_*.json` in this folder and the figures to `figures/`. `results_expected/` holds the
outputs from our own run so you can compare. Random seeds are fixed in every script.

## What each test asks

| Script | Question | Null / control |
|---|---|---|
| s15 | Is there a sheet (bifolium) term after Currier language, scribe, section and quire? (PERMANOVA) | leaves shuffled between sheets within the same quire and scribe |
| s15 | Are pages on the same sheet more alike than facing pages, once leaf, quire, scribe, section, Currier, book distance and page size are in the model? (regression on distance matrices) | same leaf-shuffle null; delete-one-sheet jackknife |
| s25 | Sheet by sheet: is a page closer to the other half of its own sheet than to other sheets by the same scribe in the same quire? | sign count over sheets |
| s26 | Do the above survive cutting every page to the same length? Does text with no sheet structure produce them? | words shuffled between pages of the same scribe and quire, each page keeping its size |
| s20 | Does the sheet effect hold in every transliteration, rebuilt from scratch? | as s15/s25 |
| s22, s24 | Can the text order sheets at all? Calibrated on two texts of KNOWN order (Circa instans, Galen) cut into 49 single-sheet gatherings with VMS page sizes | the same texts with words shuffled |
| s27 | Is there a reproducible gradient inside one scribe's group (Herbal A, scribe 1)? The order is built twice, once from each leaf of every sheet | random relabelling; shuffled controls |
| s28 | Do newer ordering methods (TempSeq-like smoothness; biostratigraphy-style incidence seriation) find true neighbours better than seriation / shortest path? | known-order texts; shuffled texts |
| s30 | On eo777's Echo chain: is each interior stage more than a mix of the pages before and after it? | whole pages reshuffled among earlier / this / later stages |

`s23_bound_chain.py` is kept only as a function library for s24/s27/s28; its own unequalised test was withdrawn
(it responds to page size; see the size control in s24).

## Inputs (data/)

| File | What it is | Source |
|---|---|---|
| `words_by_page.csv.gz` | one row per word token: folio, line, position, word (EVA), Davis scribe, Currier language, quire | Takahashi-frame EVA transliteration as used in our DINOv3/Nomic word map |
| `folio_bifolium_canonical.csv` | page -> sheet (bifolium), leaf, section, scribe, Currier | sheets from eo777/voynich-clusters `golf-v21-assignments.csv`; sections from our canonical section map; scribes = Lisa Fagin Davis's hands |
| `faces_per_sheet.csv` | the four standard pages of each bifolium | derived from the sheet codes |
| `eo777_chain_pages.csv`, `echo_rule_outputs.json` | Echo chain stages and the word forms its rules produce | github.com/eo777/voynich-rules, commit 42ea839 (13 Oct 2026) |
| `controls/*.txt.gz` | Circa instans (Latin herbal, 52k words) and Galen (Greek pharmacology, 122k words, transliterated), in text order | from the Voynichdecomp repository |

## What is and is not claimed

Claimed: conjugate leaves share a text state beyond scribe, quire, section, Currier language and page size, and it
attaches to the sheet, not to the present opening. Within one scribe's group there is a reproducible gradient.

Not claimed: which end of any gradient is earlier; that a gradient is the order of writing; anything about meaning.
On texts of known order cut to VMS sizes, the best method recovers about 1 true neighbour in 5, so a sheet-by-sheet
order cannot be read off the text. A same-face effect we reported earlier did not survive the equal-length rerun
and is withdrawn.
