# S57 Herbal text-boundary experiment — 19 October 2026

## Decision
**UNIDENTIFIED:** The test does not establish that each Herbal page is a standalone prose entry, nor does it identify continuous prose across leaves. The robust finding is physical/textual affinity within and across leaves of original bifolia.

## Method
- Positive/negative control on 341 likely named entries in Culpeper's later English herbal: 279 matched pairs, a genuine within-entry paragraph seam against a distinct-plant-entry boundary.
- Five held-out folds for inverse-document-frequency vocabulary; 4,000 paired sign-swap draws. Provisional qualification pre-specified: AUC >=0.70; permutation p<=0.01; >=30 examples. **Exact-token feature qualified:** AUC 0.72594, p 0.00025. **Character trigram failed:** AUC 0.68869 despite p 0.00025.
- ZL Herbal text, preferred reading: 127 pages (95 A, 32 B), 11,493 readable tokens, 1,608 line loci.
- Actual first/last 40-word windows compared to same-quire, same-Currier, same-scribe, 0.4-2.5x length wrong-sheet controls.
- Physical contrasts keep (i) recto-verso same leaf, (ii) consecutive different leaves, and (iii) nonadjacent pages on **different leaves of the same bifolium** separate.
- Monte Carlo n=4,000; uncertainty also estimated by resampling seven quire clusters. Page-pair comparisons are dependent within quires.

## Confirmatory qualified exact-token comparisons

| Family | n | Observed cosine | Matched mean | Excess | Monte Carlo one-sided p | Quire-bootstrap 95% interval | Forward-vs-reverse p |
|---|---:|---:|---:|---:|---:|---|---:|
| Same leaf recto→verso | 46 | 0.04761 | 0.03488 | +0.01273 | .00025 | [.00717,.01784] | .172 |
| Consecutive leaf verso→recto | 31 | 0.04424 | 0.03479 | +0.00945 | .01325 | [-.00273,.02297] | .690 |
| Different conjugate leaves, nonadjacent | 84 | 0.0460 | 0.0353 | +0.01067 | .00025 | [.00010,.02146] | .152 |

The tabulated conjugate observed/matched means are rounded approximations; source-of-truth numeric values are in S57_RESULTS_20261019.json and GitHub Actions artifact. Multiple-testing caution: p=.013 for consecutive leaves does not survive conservative Bonferroni for six family-by-feature contrasts. The unqualified character-trigram contrasts are strictly exploratory.

## Entry-start effects
Within-page swap permutation, opening token 2-character initial vs next paragraph's opening token:
- A: 64 pairs; p=.01025
- B: 23 pairs; p=.04474

This is compatible with page-opening layout conventions and does not demonstrate semantic entry segmentation. Neither survives a conservative multi-test correction.

## Identifiability
The instrument is imperfect even on readable English botany (AUC=.726) and is **not calibrated for an unknown 15th-century source language under potential encipherment**. The positive control is from a 19th-century printed edition of seventeenth-century English herbal material, not a language-matched medieval source. Under our identifiability-first rules, do not interpret a missing directional lexical seam as proof of independent entries, and do not count conjugate-sheet vocabulary as prose continuity. Current manuscript foliation may differ from original narrative order.

Retraction of any stronger claim in the earlier exploratory message that continuity definitively fails between Herbal leaves: **no such conclusion follows**.

## Sources and reproducibility
- Executable: `research/herbal_entry_continuity/s57_known_entry_boundary.py`
- Full JSON: `research/herbal_entry_continuity/S57_RESULTS_20261019.json`
- Execution logs / downloadable run artifact: https://github.com/digitalgoldfisj79/Voynichdecomp/actions/runs/39202126375
- Control source: https://www.gutenberg.org/files/49513/49513-h/49513-h.htm
