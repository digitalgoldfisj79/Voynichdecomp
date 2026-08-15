# ABC-B ReM length-control recovery — frozen 2026-08-15

This is a **reconstructed sensitivity control**, not a claim to have recovered the missing August-13 `rem_matched.py` source. It is frozen before its result is observed.

## Historical requirements recovered from the August handover

- Control corpus: ReM v2.1 diplomatic Middle High German, original word boundaries restored.
- Matched comparison convention: 50 pseudo-corpora, each copying the VMS folio/line structure exactly (same folio count, line counts, line lengths) while taking contiguous real ReM text.
- ABC-B requirement: apply the same ED1 mean-word-length partition to matched ReM. The original text says ReM should show a crowding gradient if the crowding instrument is meaningful.

## Reconstruction chosen before scoring

1. Rebuild ReM v2.1 from its canonical TEI ZIP using the handover's `fetch_and_build_rem.py` rules: diplomatic `<w>` text, `_m` fragments regrouped by original token id, and the frozen punctuation/digit cleaning regex.
2. Build the exact VMS skeleton from the same `enriched_records.json` used by the ABC/FW recovery, grouping records by `(folio, line_no)` and preserving token order.
3. For each of 50 replicates (base seed 20260813): for each VMS folio, choose uniformly from ReM documents long enough to supply that folio's total token count, then choose a uniform contiguous start position and cut that real-text segment into the VMS folio's exact line lengths. Thus every pseudo-folio is contiguous real ReM text; artificial joins occur only between pseudo-folios.
4. ED1 is literal Levenshtein distance exactly 1, excluding identical tokens, identical to ABC.
5. Partition ED1 pairs by mean endpoint length: short <=4, mid >4 and <=6, long >6.
6. For each class, compare observed adjacent count against the **exact expectation under the same within-line multiset-preserving random-order null**. For a line of `n` token positions, each unordered pair of positions is adjacent with probability `2/n`; summing qualifying position-pairs gives the exact permutation-null mean. This is mathematically the mean of the shuffle null, avoiding Monte Carlo noise.
7. First validate the exact-expectation calculation on the VMS itself: the observed counts must reproduce 598/501/146 and the exact null means must agree closely with the historical 200-permutation means 483.46/471.405/109.88. Failure cancels the ReM interpretation.

## Report

For each length class report across 50 ReM pseudo-corpora:
- mean/SD observed count;
- mean/SD exact-null expectation;
- mean, median, 10th/90th percentile of observed/null ratio;
- fraction of replicates with ratio >1.

Report the fraction of pseudo-corpora showing a monotonically decreasing short > mid > long ratio and the fraction satisfying the historical crowding pattern `long <= 1.05` with monotonic decrease.

## Interpretation guardrail

The ReM arm is a known-answer/sensitivity control for ABC-B's proposed length discriminator. It cannot overturn the directly preregistered VMS fact that long-word ED1 is 1.329x null at z=3.78; it can determine whether the proposed *generic crowding-gradient interpretation* is itself calibrated on real historical language. If ReM does not show the predicted gradient, ABC-B still rejects the specific VMS short-word-crowding account by its frozen VMS criterion, but the claim that this discriminator is a generic diagnostic of crowding must be withdrawn.