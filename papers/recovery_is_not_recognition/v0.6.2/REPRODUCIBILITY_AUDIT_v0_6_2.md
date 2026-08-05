# Reproducibility audit v0.6.2

## C2ST closure

The candidate builder (`step8a_candidates.py`) and full 13-feature evaluation harness (`step8b_eval_harness.py`) were recovered from the dated 11 June 2026 session transcript. They are preserved verbatim apart from no path edits in the archival copies. A transparent builder reconstructs `corpus.pkl` from committed `enriched_records.pkl`; it validates all 37,465 token positions, 5,162 line boundaries, declared line lengths, and first/last-word flags.

A clean rerun under Python 3.12.13, NumPy 2.4.4 and scikit-learn 1.8.0 reproduced the five published full-feature AUC means and standard deviations exactly:

- authentic-B: 0.485 ± 0.063;
- line shuffle: 0.872 ± 0.053;
- word shuffle: 0.970 ± 0.022;
- template generator: 0.992 ± 0.008;
- delexicalised generator: 1.000 ± 0.000.

The recovered full harness uses 13 features, not ten. The original ablation edit was not textually recovered. A preregistered exhaustive search over all 1,287 eight-feature subsets identified one subset meeting the <=0.005 maximum-cell criterion while excluding the explicit opener and adjacent-repeat artefact features. The supplied ablation is labelled a reconstruction and reproduces the five published cells within 0.002. A clean 64-CPU forensic run searched all 1,287 subsets: exactly one met the preregistered maximum-cell tolerance of 0.005 (best 0.002448; next best 0.008688).

## Information-budget correction

The historical code exactly reproduces plug-in MI 2.6343 bits over H(quad)=9.1242 bits, or 28.87%. A 200-permutation audit holding the 345 context cells fixed produced null mean 2.1222 bits and null 95th percentile 2.1302 bits. The apparent 28.9% is therefore dominated by high-cardinality plug-in bias and has been withdrawn. The observed-minus-null difference of 0.5121 bits (5.61%) is retained only as a post hoc diagnostic, not a replacement headline.

## Remaining limitations

Several earlier heavy calibration runs retain exact code, deterministic inputs, reports, immutable jobs and hashes but not committed copies of every historical row-level file. External third-party replication has not been completed. These are now the material reproducibility limitations; the C2ST full-harness source is no longer the principal gap.
