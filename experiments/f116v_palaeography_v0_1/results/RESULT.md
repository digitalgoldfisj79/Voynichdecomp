# f116v palaeographic extraction result

## Final programme status

`GLYPH_EXTRACTION_PILOT_PASS`

This means the baseline-constrained pipeline recovered a non-trivial set of positionally stable glyph hypotheses across two acquired image views. It does **not** mean that a complete or independently validated transcription has been obtained.

## Executed evidence

- Three source views: laboratory true-colour TIFF, expert monochrome multispectral PCA composite, and expert colour PCA composite.
- Four physically fixed line bands; OCR was not used to locate the lines.
- Primary recognizer: Kraken 7 with CATMuS Medieval.
- Independent hostile recognizer: CATMuS-trained medieval TrOCR.
- Shape control: DINOv2 base, used only for visual-form comparison.
- Blank parchment control and synthetic fading controls.
- No dictionary, language model, abbreviation expansion, word completion, OCR-guided line detection, generative restoration, or semantic inpainting.

Final Hugging Face job: `6a71db2e6b79c09949c21f1c` (L4 GPU, completed in 99 seconds after scheduling).

## Control results

- CATMuS blank-derived confidence gate: **0.7034**.
- Highest blank-control character confidence: **0.6534**.
- High-confidence non-space characters on blank controls: **0**.
- Exact high-confidence line-2 positions retained after synthetic fading:
  - 75% acquired-stroke amplitude: **24/26 = 92.3%**;
  - 50% amplitude: **24/26 = 92.3%**;
  - 25% amplitude: **19/26 = 73.1%**.

The fading result demonstrates recognizer stability under controlled attenuation. It does not prove that every retained label is palaeographically correct.

## Positionally stable CATMuS hypotheses

A `probable positional agreement` requires:

1. approximately the same character cut in true-colour and monochrome-PCA views;
2. the same CATMuS character label;
3. confidence above the blank-derived gate in both views; and
4. local acquired-edge correlation of at least 0.68.

Counts:

- Line 1: **0 probable**, 0 high-confidence ambiguous positions.
- Line 2: **15 probable**, 2 high-confidence ambiguous positions.
- Line 3: **5 probable**, 1 high-confidence ambiguous position.
- Line 4: **9 probable**, 0 high-confidence ambiguous positions.

These are agreements within one recognition architecture applied to independent acquired views. They remain `PROBABLE`, not `SUPPORTED`.

## Best defensible apparatus

Spaces are not inferred. Ellipses mark unresolved spans.

- Line 1: `<unresolved>`
- Line 2: `… <n?> c h i c o <n?> [o|e] l a d a b a … s … r [c|d] <e?> r e … <p?> o <r?> …`
- Line 3: `… <t?> … a r i <x?> … <m?> o u <x?> <t?> <u?> [x|o] … <l?> …`
- Line 4: `… <c?> <o?> <t?> … p a … <e?> a u … r e <n?> … <s?> o <combining-mark?> … <u?> <i?> g a …`

The strongest continuous line-2 model-derived core is therefore approximately:

`…chico<n?> [o|e]ladaba…`

or, including the lower-confidence preceding glyph:

`…<n?>chico<n?> [o|e]ladaba…`

This resembles portions of existing human readings, but the programme does not validate a complete phrase. In particular, the initial characters, the `o/e` boundary, the last character after `ladaba`, word division, and most of the second half of the line remain unresolved.

## Independent-architecture result

TrOCR produced text-like outputs, but cross-view outputs were unstable and only weakly overlapped CATMuS:

- longest line-2 CATMuS/TrOCR common substring: **4 characters** (`s ⁊ `, including spacing/sign context);
- line 3: ` mou`;
- line 4: `a u`.

This is insufficient to promote any sequence to `SUPPORTED`. TrOCR is useful here chiefly as a warning against accepting plausible medieval-looking strings at face value.

## DINOv2 result

DINOv2 loaded successfully and generated visual embeddings for the probable positions. It confirmed that many paired true-colour/BW-PCA crops share substantial local visual structure, but the automatic clustering divided the 29 probable positions into 28 clusters at the frozen conservative threshold. It therefore did not establish repeated-glyph families strongly enough to upgrade labels.

## Answers

### Did multispectral imaging add recoverable glyph information?

Yes, in a limited sense. The monochrome PCA view supplied an independent acquired rendering whose positional agreement with true colour allowed 29 provisional glyph positions to survive blank-calibrated gates. The colour PCA view was generally much less useful for recognition.

### Which positions improved?

Primarily the central portion of line 2, with smaller stable cores in lines 3 and 4. Line 1 remains unresolved.

### Are existing readings confirmed?

Only partially and at the level of local glyph sequences. The line-2 region compatible with `…chicon…ladaba…` receives real model support across two views. A complete historical transcription is not confirmed.

### Is there evidence of retracing or multiple writing phases?

Not from this extraction alone. Differential appearance across views may reflect ink, retracing, staining, illumination, or composite construction. A phase claim requires a separate material/stratigraphic analysis.

### What is the best defensible transcription?

There is no defensible complete transcription. The apparatus above is the strongest defensible output.

## Final interpretation

The programme produced a useful **glyph-location and hypothesis apparatus**, especially for line 2. It did not solve f116v. The next valid step is blinded human palaeographic review of the source crops and positional alternatives, followed by targeted fine-tuning or few-shot calibration on genuinely comparable late-medieval hands. A generic OCR vote should not be used to close the remaining gaps.