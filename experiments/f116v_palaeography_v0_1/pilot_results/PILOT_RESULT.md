# f116v whole-line HTR pilot result

## Verdict

`PILOT_STABILITY_GATE_FAIL`

The whole-line recognition pilot does **not** support a transcription. It is diagnostic evidence that unconstrained HTR is too unstable and too prone to text-like output on parchment controls for direct use on f116v.

## Execution

- Hugging Face job: `6a71d2cba00abefd4b29166f`
- Hardware: L4 GPU
- Status: completed
- Runtime: 284 seconds
- Primary HTR: Kraken 7.0.3 with CATMuS Medieval, Zenodo record version 1.6.2, model file `catmus-medieval-1.6.0.mlmodel`
- Independent controls: `medieval-data/trocr-medieval-base`, `medieval-data/trocr-medieval-cursiva`, and `medieval-data/trocr-medieval-textualis`
- Views: expert monochrome PCA, colour PCA, and true-colour TIFF
- Pilot target: second surviving marginal line

## Stability metrics

### Kraken

- Cross-view normalized string similarity: `0.183–0.210`
- Blank-control output lengths: `62`, `111`, `136` characters
- Synthetic-fading similarity to the unmodified BW crop: `0.269–0.294`

### TrOCR

- Medieval base cross-view similarity: `0.109–0.234`; blank output length `22`
- Cursiva cross-view similarity: `0.219–0.274`; blank output length `49`
- Textualis cross-view similarity: `0.224–0.310`; blank output length `25`

### Cross-architecture

Similarity between Kraken BW output and the three TrOCR BW outputs was only `0.126–0.176`.

## Interpretation

The models produced plausible-looking medieval Latin strings on the target crop, but they also produced substantial pseudo-text on blank parchment and changed materially between spectrally aligned views of the same physical writing. No particular word or character sequence is therefore accepted.

One Kraken true-colour output included the visually suggestive sequence `mehicon ladabas`, close to familiar human readings of this line. That agreement is not evidential: the output is embedded in a longer unstable pseudo-Latin sequence, is absent from the other aligned views, and the same recognizer emits long strings on blank controls.

## Consequence

The programme will not scale whole-line OCR to all four lines. The next stage must invert the workflow:

1. establish cross-view physical stroke support;
2. segment source-supported glyph units;
3. cluster repeated forms without labels;
4. use HTR only to annotate already-supported glyphs or short groups;
5. reject any label that is unstable across views or comparable to blank-control behaviour.

No language model, dictionary correction, abbreviation expansion, or semantic completion was used.