# Amendment 002 — Hebrew positive-control source extension

Date: 2026-08-09

K-SECTION reached its metadata/sample census and then stopped before any Voynich language optimization because its synthetic piecewise control required 81,518 Hebrew letters while the frozen UD Hebrew 20% holdout contains 81,456 letters: a 62-letter shortfall.

No K-SECTION Voynich language score existed at amendment time.

Prospective repair:

- The Hebrew positive-control plaintext pool is extended, after the frozen UD Hebrew holdout, with the independent Sefaria Hebrew source already used in the v0.2 historical-control programme: `Mishneh Torah, Torah Study`, Hebrew, version `Torat Emet 363`.
- This extension is used **only as known plaintext for positive controls**. It is not added to the Hebrew language-model training corpus.
- UD Hebrew holdout remains first in the concatenated pool, so K-CURRIER control material is unchanged and K-SECTION differs only after the 81,456th normalized Hebrew letter.
- Other language control pools and all Voynich data are unchanged.

Protocol-deviation note: the v0.4 implementation uses a deterministic representative training-page sample targeting ~35k glyphs across groups, while the prose preregistration did not state that sampling rule explicitly. Therefore v0.4 is treated as a bounded screening ladder. Any positive piecewise-key signal must be rerun under a separately frozen confirmatory protocol with its training-sample rule explicit before being reported as evidence.