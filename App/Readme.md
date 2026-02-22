# VMS Corpus Explorer — Beinecke MS 408

A self-contained analytical toolkit for the Voynich Manuscript, embedding the full 37,465-token corpus with 80 validated statistical metrics, four-slot grammar decomposition, and multi-script document comparison. Runs entirely client-side — no server, no dependencies, no data leaves the browser.

![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Web%20%7C%20Android%20%7C%20iOS%20%7C%20macOS-green)
![Metrics](https://img.shields.io/badge/metrics-80%20validated-brightgreen)
![Corpus](https://img.shields.io/badge/corpus-37%2C465%20tokens-gold)

---

## What Is This?

The Voynich Manuscript (Beinecke MS 408) is a 15th-century codex written in an undeciphered script. This tool provides a complete computational linguistics workbench for analysing the manuscript's text, comparing it against external corpora, and exploring its internal structure.

Everything — the corpus data, the VoynichEVA font, the statistics engine, and the UI — is compiled into a single HTML file. Open it in a browser and you're working.

## Features

### Nine Analytical Panels

| Panel | Purpose |
|---|---|
| **Dashboard** | Top 30 tokens, prefix/suffix-family/section distributions |
| **Folio Browser** | Line-by-line rendering in VoynichEVA glyphs |
| **Grammar View** | Four-slot (P·G·C·S) morphological decomposition with EVA labels |
| **Search** | Exact, prefix, suffix, contains, and core-pattern modes |
| **Frequency** | Sortable token frequency tables |
| **Alphabet** | EVA-to-glyph reference with live preview |
| **Live Stats** | 80 metrics computed on any scope (whole corpus, section, or folio) |
| **Compare** | Side-by-side folio-vs-folio or section-vs-section statistics |
| **Upload & Compare** | Upload any text, tokenise, and compare against VMS |

### 80 Validated Statistical Metrics

Organised into 12 categories covering the full Gaskell-Bowern descriptive framework:

- **Word length** — mean, std, skewness (all tokens and unique types), autocorrelation
- **Word distribution** — max frequency, shape
- **Word positional bias** — 5-bin heat within line, 5-bin heat across document (coefficient of variation)
- **Character distribution** — max frequency, shape, evenness, redundancy, Simpson's D, Yule's K
- **Character positional bias** — 10-bin within line, 5-bin within word (CV)
- **Entropy hierarchy** — H₀ (max), H₁ (unigram), H₂ (2nd-order Markov conditional), h₂ (joint/conditional digraph), h₃ (joint/conditional trigraph)
- **Digraph/trigraph** — unique counts, coverage ratio
- **TTR variants** — TTR, RTTR (Guiraud), CTTR (Carroll), LogTTR (Herdan), Maas a², Uber Index (Dugast), Brunet's W, MSTTR and MATTR at windows 25/50/100
- **Hapax & frequency spectrum** — hapax/dis ratios (token and type), Sichel's S, Honoré's R, f(1)/f(2)/f(3)/f(>10)
- **Lexical richness** — word Yule's K
- **Repetition** — word/character repeat and triple ratios, flipped pair ratio
- **Autocorrelation** — word length, word frequency, TTR-25, hapax-25 (lag-1)

Every metric has been cross-validated to < 1×10⁻⁶ tolerance against an independent Python reference implementation across three test scopes (whole corpus, section, single folio). 17 key formulae additionally spot-checked against textbook definitions.

### Multi-Script Upload Support

Upload `.txt` or `.docx` files in 16 scripts across 5 language families:

| Family | Scripts |
|---|---|
| **Western** | Latin / Pinyin, Greek, Cyrillic, Georgian, Armenian |
| **Semitic & African** | Arabic, Hebrew, Syriac, Coptic, Ethiopic / Ge'ez |
| **CJK & East Asian** | Chinese (character split), Japanese (character split), Korean |
| **South & Southeast Asian** | Devanagari / Hindi / Sanskrit, Thai (character split), Tibetan |

Features:
- **Auto-detection** from first 3,000 characters by Unicode range
- **RTL rendering** for Arabic, Hebrew, Syriac
- **CJK character-level splitting** (automatic for Chinese/Japanese/Thai)
- **Diacritics stripping** — NFD decomposition covering tashkeel, niqqud, polytonic accents, Pinyin tones, Devanagari virama, Thai marks, Tibetan signs
- **Morphological options** — Arabic definite article (ال) stripping, Hebrew prefix stripping (ב,ה,ו,כ,ל,מ,ש)
- **Colour-coded comparison** — green (≤5% of VMS target), amber (≤10%), red (>10%), with proximity score summary

### Tokenisation Options

- Lowercase
- Strip punctuation (Unicode-aware `\p{L}` matching)
- Strip numbers
- Strip diacritics
- Character-level split (CJK/Thai)
- Custom split regex
- Minimum token length filter
- Live preview of first 500 tokens

## Quick Start

### Browser (simplest)

Download `vms_explorer_pwa_index.html` and open it. That's it.

### PWA (installable, offline)

```bash
# Unzip and serve
unzip vms_pwa.zip
python3 -m http.server 8000

# On your device, open http://localhost:8000
# Chrome: tap "Add to Home Screen" for a native-feeling app
```

Works on Android, iOS (Safari), and desktop browsers. Offline after first load via service worker caching.

### Android (native WebView)

```
vms_android.zip
├── app/src/main/
│   ├── assets/          # index.html + mammoth.min.js
│   ├── java/.../        # MainActivity.kt (~100 lines)
│   ├── res/             # Dark theme, launcher icons
│   └── AndroidManifest.xml
├── build.gradle
└── settings.gradle
```

Open in **Android Studio** → Gradle sync → Run. Min SDK 24 (Android 7.0+). The `onShowFileChooser` override handles file uploads through the native Android file picker.

### iOS / macOS (native WKWebView)

```
vms_ios.zip
├── VMSExplorer/
│   ├── VMSExplorerApp.swift   # @main entry (SwiftUI)
│   ├── ContentView.swift      # WKWebView wrapper (~60 lines)
│   ├── Info.plist
│   ├── Assets.xcassets/       # App icon, launch colour
│   └── Resources/             # index.html + mammoth.min.js
└── VMSExplorer.xcodeproj/
```

Open in **Xcode 15+** → set signing team → Run. iOS 16+, iPad, Mac (Catalyst enabled). File uploads work natively via iOS document picker.

## Corpus Data

The embedded corpus contains 37,465 tokens across 5,162 lines from all nine manuscript sections:

| Section | Tokens |
|---|---|
| Stars | 10,702 |
| Balneological | 6,859 |
| Herbal-B | 5,783 |
| Herbal-A | 4,033 |
| Pharmaceutical | 3,870 |
| Rosettes | 1,818 |
| Zodiac | 1,590 |
| Astronomical | 1,469 |
| Cosmological | 1,341 |

Each token record contains: `[word, prefix, gallows, core, suffix, suffix_family, section, folio, line]`

Transcription follows the EVA (European Voynich Alphabet) system with compound glyph analysis based on the EVA-Boz scheme.

### Zodiac Folio Mapping

Definitive assignment from statistical validation work:

| Folio | Sign |
|---|---|
| f70v2 | Pisces |
| f70v1 | Aries (Dark) |
| f71r | Aries (Light) |
| f71v | Taurus (Light) |
| f72r1 | Taurus (Dark) |
| f72r2 | Gemini |
| f72r3 | Cancer |
| f72v3 | Leo |
| f72v2 | Virgo |
| f72v1 | Libra |
| f73r | Scorpio |
| f73v | Sagittarius |

## Grammar Decomposition

The four-slot model (P·G·C·S) decomposes each Voynich word into:

- **P** (Prefix): word-initial element — `∅, ch, d, o, qo, s, sh, y`
- **G** (Gallows): tall character slot — `∅, k, t, f, p` and variants
- **C** (Core): central body — the main vowel/consonant sequence
- **S** (Suffix): word-final element, classified into suffix families: `BARE, L, M, N, R, Y, OTHER`

The Grammar View panel renders each word with VoynichEVA glyphs above and EVA text labels below, for both the full word and each individual slot.

## Validation

The statistics engine has been rigorously validated:

1. **Cross-implementation** — A Python reference (`reference_stats.py`) computes all 80 metrics independently. Both engines were run on three test scopes (whole corpus: 37,465 tokens; Herbal-A section: 4,033 tokens; folio f1r: 207 tokens). Result: **80/80 exact match** (< 1×10⁻⁶ tolerance) across all scopes.

2. **Textbook spot-checks** — 17 metrics verified against hand-computed values from original definitions (Yule 1944, Guiraud 1954, Herdan 1960, Carroll 1964, Maas 1972, Sichel 1975, Brunet 1978, Honoré 1979, Dugast 1979, Shannon entropy). All exact.

3. **Binning consistency** — Positional bias metrics use `floor(position/length × bins)` throughout, ensuring identical behaviour across Python (`int()`) and JavaScript (`Math.floor()`).

To re-run the validation:

```bash
# Requires: Python 3.8+, numpy
# Place corpus_compact.json and test_data.json in the same directory
python3 reference_stats.py
node run_js_stats.js
# Compare the JSON outputs
```

## Technical Notes

- **Self-contained**: The 4.1 MB HTML file embeds the full corpus (JSON), VoynichEVA font (base64 WOFF2), mammoth.js (for .docx parsing), and all application code. No external requests.
- **Performance**: Whole-corpus stats compute in ~1–2 seconds (single-threaded JS). Single folios are near-instant.
- **Offline**: Service worker caches all assets after first PWA load.
- **Privacy**: All processing is client-side. Uploaded documents never leave the browser.
- **Unicode**: Punctuation stripping uses Unicode property escapes (`\p{L}`, `\p{M}`, `\p{N}`) for correct handling across all scripts.

## File Manifest

```
vms_explorer_pwa_index.html   # Standalone app (open in browser)
vms_pwa.zip                   # PWA package (6 files)
├── index.html
├── manifest.json
├── sw.js
├── mammoth.min.js
├── icon-192.png
└── icon-512.png
vms_android.zip               # Android Studio project
vms_ios.zip                   # Xcode project
reference_stats.py            # Python validation implementation
```

## License

MIT

## Citation

If you use this tool in academic work:

```
VMS Corpus Explorer: A validated computational linguistics toolkit
for the Voynich Manuscript (Beinecke MS 408). 2025.
```

## Acknowledgements

- Voynich Manuscript transcription: based on the EVA system (Landini, Zandbergen et al.)
- VoynichEVA font: for glyph rendering of the EVA transcription alphabet
- mammoth.js: for client-side .docx text extraction

