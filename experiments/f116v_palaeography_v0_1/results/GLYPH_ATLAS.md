# f116v provisional glyph atlas

## Line 2

The generated atlas displays source crops for positions classified as either `PROBABLE_CROSS_VIEW_SINGLE_ARCH` or `AMBIGUOUS_LABEL` by the corrected pipeline.

![Line 2 provisional glyph atlas](https://v3b.fal.media/files/b/0aa4fc89/ZUzU0FTJwGPqc9ZOZJD-2_line2_glyph_atlas.jpg)

The atlas labels show the position ID and the true-colour/BW-PCA CATMuS alternatives. They are recognition hypotheses, not normalized palaeographic readings.

## Reading discipline

- Review the crop before the model label.
- A repeated label does not establish a repeated glyph unless its visual form also agrees.
- The DINOv2 clustering threshold was deliberately conservative and did not establish robust repeated-glyph families.
- Colour-PCA crops were often too transformed for character recognition and should be treated as ancillary physical views.
- Unresolved positions remain unresolved rather than being filled from surrounding text.