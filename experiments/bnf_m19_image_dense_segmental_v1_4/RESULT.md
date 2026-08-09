# BnF M19 Image Bridge v1.4 — Result

HF job: `6a783fc43e1f34a7e32c0133`
Protocol freeze: `8ac23fe03a7907269edfdc983857ba82c217224a`
Runner: `503261110e275ea1cf1d53b74fe666b08bbfb726`

Verdict: **DENSE SEGMENTAL IMAGE-UNDERPOWERED**.

No language score was generated. The result strengthens the separation between boundary recovery and discrete class recovery: dense DINO yields highly reproducible visual segmentation boundaries, but a 19-class image alphabet remains below the frozen stability threshold.

| lambda | boundary F1 | class stability | coverage | silhouette | mean segments/word | gate |
|---:|---:|---:|---:|---:|---:|---|
| .02 | .8526 | .5396 | .9481 | **.12591** | 3.2483 | fail |
| .04 | .8300 | .4799 | .9479 | .10974 | 3.2480 | fail |
| .06 | .8628 | .6428 | .9484 | .12274 | 3.2480 | fail |
| .08 | .8543 | .6340 | .9458 | .10189 | 3.2480 | fail |
| .10 | .8742 | .6592 | .9457 | .10872 | 3.2480 | fail |
| .12 | **.8864** | **.6890** | .9445 | .11422 | 3.2480 | fail |

The prospectively frozen class-stability threshold was 0.75. v1.4 therefore does not permit a discrete M19 language test. The image evidence does, however, support λ=.12 as the most reproducible segmentation boundary model among the predeclared settings. A subsequent continuous-emission model may use these boundaries while discarding hard class labels; that is a new model class and requires separate positive controls.
