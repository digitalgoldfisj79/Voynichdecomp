# VBM architectural length and scribal feasibility gate
**14 October 2026 · bounded NO-GO for v142b–v145 source granularity; scribal execution NOT QUALIFIED**

## Correction and decision

- The exact frozen word parser uses paired source consonants when the pair belongs to the 222-unit inventory and **falls back to two singles otherwise**. A prior quick audit used a simpler approximation; this standalone replay reproduces the original fallback rule. It counts **29,725** mandatory code cells and **18,928** vowel bridges over **13,232** Hartlieb words, and **434** distinct state/unit assignments used. Earlier provisional counts of 29,713 and 438 are superseded.
- An absolute **source-specific** lower bound exists for the frozen v142b/v145 source segmentation, even before prefix-free distinctness, observed joins, or handwriting are required:

  \[ \operatorname{len}_{\rm EVA}(\operatorname{encode}(w)) \ge 2C(w)+V(w), \]

  where `C(w)` counts the exact mandatory emitted code cells, including an epsilon cell for a vowel without a following consonant, and `V(w)` counts vowels, each triggering an explicit bridge. Every original code cell has at least **two raw EVA characters**, every bridge at least **one**.
- Hartlieb: `C_total=29,725`, `V_total=18,928`, so any such encoder emits at least **78,378 raw EVA characters** for 13,232 source words, i.e. **5.9234 characters per word**. Currier B native mean is **5.0605**; Currier A **4.7834** in the existing frozen references. The bound exceeds B by **0.8628** and A by **1.1399** characters per word.
- The minimal possible Hartlieb output would also contain **9.643%** words of at least ten EVA characters, versus **0.807%** for B and **0.564%** for A. It would contain **0.642%** words of length at least fifteen, versus none in either cited native sample. These are deterministic source-conditional comparisons, not confidence intervals or a language-independent impossibility theorem.
- **Retire optimising code assignment under this frozen one-source-word/one-output-word, source-unit segmentation.** The lower bound is below actual v145 lengths (7.959 B; 8.054 A) but already above observed Voynich lengths; no codebook permutation or join solver can close that gap on these source samples.

## Scope of the lower bound

This does **not** rule out VBM generically, the manuscript encoding a different (unknown) source, or a model that changes the unit granularity, allows source shorthand, changes how vowels are represented, changes output token boundaries, or alters the bridge/cell architecture. In particular the manuscript average is not a formal fixed target for every short historical source extract; the result is a decisive mismatch for the studied German sources under the present architecture, not a plaintext identification test.

The bound is **very generous**: it assumes all source cells could use two-character codes and all vowel bridges a single character, although the frozen inventories have only 10–13 such short cells per vowel state and require 222 distinct source-unit assignments. Actual prefix-free capacity, native seam constraints and positional glyph legality can only increase the shortest attainable length under these frozen rules.

The measured output strings are *reverse-EVA* compound-atom representations. All length comparisons use the same raw character convention. EVA character counts are **not** pen strokes, physical glyph counts, glyph widths or measures of human writing time; a separate ductus assessment is necessary.

## Independent scribal feasibility gate

A manually executable historical operation would need evidence and an explicit procedure satisfying **all** of the following:

1. **Compact action schedule:** Given the previous state and a source unit, a medieval operator can choose the next manuscript glyphs by memorised rules, a manageable physical table, or an attested scribal abbreviation convention. Explain exact lookups and page/column navigation. Do not hide a Hungarian optimiser, random-access data structure, global context search or late permutation minimisation in “the scribe chooses a form.”
2. **Bounded working memory and deterministic reversibility:** The operator needs only locally available information (current source fragment, visible output state, physical boundaries, key if one is genuinely evidenced). Identify state resets and how errors would be detected/repaired. A plaintext-free *copyist* and a plaintext-encoding *composer* are different plausible production stories and must not be conflated.
3. **Ductus and glyph economy:** Demonstrate the required joins in manuscript-like reading order and orientation, not merely admissible reverse-EVA character trigrams. Verify shape, allographic choices, ligatures, pen movements, stroke collisions, positional restrictions and physical line width from Yale canvases or existing glyph masks. Compare to manuscript hands, not only mixed Currier inventories.
4. **Material workflow:** Explain how rules or tables would be accessed across pages, sections and scribes; how a scribe fits line endings around illustrations and uses marginal/special glyphs; how the actual error and correction distribution would arise. Lack of surviving tables alone is not disproof.
5. **No arbitrary extra freedom:** Penalise number of independent mapping entries, exceptional assignments, page-specific parameters, table switches, scribal operations per word and special-case recovery. Show withheld-folio predictions and compare rule complexity to historical documentary cipher tables, not an invented threshold for human ability.

The current v145 explicit tables contain `5×222=1,110` state/unit entries. Under Hartlieb's frozen source distribution they require **2.2464** code-cell selections and **1.4305** bridge selections per word. Across the full German development sample, 705 different state/unit entries are actually used. The model has no demonstrated historical manual lookup algorithm. Counts do **not** prove 1,110 entries were impossible for a fifteenth-century scribe: contemporary Italian cipher key tables and nomenclators are historically documented; the specific executed construction remains unsupported.

In the current source sample, actual v145 B output averages **7.559 compound EVA atoms/word** against **4.578** in the native B reference. These are rough graphic-unit proxies only, not measured pen time. A glyph-mask/ductus assessment has not been run and must not be marked PASS.

## Scientific assessment

**Gate M (word-length feasibility): FAIL for the frozen v142b–v145 German-source architecture, by deterministic lower bound.**

**Gate S (historical scribal execution): NOT QUALIFIED.** The mathematical lower bound alone stops this branch, before a scribal experiment would be cost-effective. Were a **new and explicitly changed** architecture introduced, it would need to pass both gates, then real A/B whole-word/position/section holdouts, and finally independent evidence of an encoded source as opposed to autonomous transduction.

The user’s prior FORM/CONNECT/LINE_ENTRY results remain separate and are not refuted by this source-conditional code-length bound. The cross-word state transfer in v143/v144 also remains a valid *engineering* demonstration independently of failed v145 word representation.

## Historical comparanda (context only)

- Judit W. Somogyi, “Caratteristiche strutturali di cifrari monoalfabetici italiani nei secoli XV e XVI,” *Verbum* 17 (2016), https://ojs.ppke.hu/verbum/article/view/405. The abstract describes 14th- and 15th-century Italian codebook tables and rules. This supports the feasibility of historical table use generally, not this 1,110-entry scheme.
- Lisa Fagin Davis, “How Many Glyphs and How Many Scribes? Digital Paleography and the Voynich Manuscript,” *Manuscript Studies* 5 (2020), DOI 10.1353/mns.2020.0011. Manuscript allography and scribal distinctions need physical evidence. This report has not independently completed a palaeographic classification.

## Reproducibility

The accompanying standalone `reproduce.py` reads only bundled frozen source texts, codebook manifest and native A/B reference word files. It imports no fitted model or external network results. It recomputes counts and lower bounds, native distributions, generated outputs and exact word overlap and writes `RESULTS.json`. `INPUTS` contain checksums for every file, and the run was executed locally.