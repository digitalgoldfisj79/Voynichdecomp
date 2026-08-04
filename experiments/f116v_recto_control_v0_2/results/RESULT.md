# f116v stage-2 recto/show-through result

## Final verdict

`RECTO_CONTROL_INCONCLUSIVE`

This is not evidence for erased writing and it is not evidence of absence.

## Executed data

- Raw f116v TIFFs found: **46**
- Raw f116r TIFFs found: **0**
- Registered f116v bands: **46/46**
- Capture families: **28 MB**, **10 WB**, **8 TX**
- Reference: `Voynich_116v+MB570AM_005_F.tif`

## Residual detector

- Raw candidate pixels: **0**
- Recto-independent candidate pixels: **0**
- Candidate components: **0**
- Horizontal line groups: **0**
- Native-confirmed components: **0**

The zero map cannot be interpreted as a negative result because the detector failed its positive-control sensitivity gate.

## Mandatory control failures

### Synthetic acquired-stroke control

A real visible f116v stroke mask was translated into the frozen eligible lower-page field and injected using the signed acquired spectral signature.

- Target pixels: **9,121**
- Best result occurred at full amplitude
- Precision: **1.0000**
- Recall: **0.00493**
- F1: **0.00982**

The detector was highly conservative but grossly underpowered. It recovered only 45 of 9,121 planted pixels at full signature strength.

### Recto-proxy specificity

The Yale high-resolution f116r visible-light image was aligned through the constrained backside transform and compared with unrelated f115r and f115v controls.

- f116r proxy score: **0.06562**
- Best unrelated-folio score: **0.06599**
- Specificity margin: **−0.00037**
- Median show-through-model R²: **0.0269**

The correct recto proxy did not outperform the unrelated control. It therefore cannot identify or subtract f116r show-through reliably.

## Interpretation

The public 2014 archive contains no matching raw f116r multispectral cube. The available visible-light recto proxy is non-specific, and the current high-specificity residual detector lacks adequate sensitivity. Consequently:

1. no erased f116v writing was recovered;
2. no scientifically valid negative conclusion can be drawn;
3. no physical indentation conclusion is possible without calibrated multi-direction raking-light images.

No OCR, language model, diffusion restoration, super-resolution hallucination or semantic inpainting contributed to the verdict.
