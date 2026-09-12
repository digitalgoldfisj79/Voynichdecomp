# Anonymous word-equality v0.1 nuisance matching addendum — frozen before nuisance outcomes

The parent protocol permits <=2% token-count difference under scan nuisance but requires positional comparison of equality partitions. This addendum fixes the positional matching rule before any nuisance-cell result is inspected.

All perturbations are returned to the original 3600×5400 pixel canvas before segmentation. Original and perturbed retained words are represented by their ink bounding boxes in page coordinates and by `(line_index, RTL_word_index)`.

1. First match words having identical `(line_index, RTL_word_index)` when both segmentations contain that slot.
2. A candidate slot match is accepted only if box IoU >=0.35 OR Euclidean centre displacement <=35 px.
3. Do not content-match, descriptor-match, OCR-match, or search alternative alignments.
4. The matched-position fraction is `matched / max(N_original,N_perturbed)`.
5. A nuisance cell requires token-count relative difference <=2%, matched-position fraction >=0.98, and adjusted Rand index >=0.90 on the equality-cluster labels for the accepted matched positions.
6. Labels are compared as partitions; cluster numeric IDs need not agree.
7. Overall gate remains >=22/24 passing cells.

This is an implementation clarification only; no threshold, descriptor, segmentation constant, perturbation family, or pass criterion from the parent protocol is altered.
