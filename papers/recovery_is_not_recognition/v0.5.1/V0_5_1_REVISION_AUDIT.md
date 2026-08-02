# Recovery Is Not Recognition v0.5.1 Revision Audit

## Decision

Version 0.5.1 is a controlled editorial compression of v0.5. The scientific result set and claim boundaries are unchanged. The revision is suitable for external review but remains short of submission freeze pending clean-room archive assembly and inherited reference completion.

## Quantitative editorial result

- Main source reduced from approximately **14,102** to **9,405** words including references: **4,697 words**, or about **33%**, removed.
- Abstract reduced from **387** to **296** words.
- Main-text tables increased from **1 overloaded ledger** to **4 compact reader-facing tables**.
- The full **18-row** experimental ledger was preserved as Supplementary Table S1.
- Main LaTeX rendering reduced from **32** to **28** pages; Word-layout rendering from **63** to **43** pages.

## Numerical sufficiency

The revision preserves the numerical values that determine each principal inference:

- generator score 59/84 versus line shuffle 74.6/84 and generator C2ST AUC 0.992;
- Family P recovery 94.32% versus operational recognition 34.7%;
- true-map matched-source success 16/16 versus complete transfer 0/16;
- intact compression macro accuracy 0.9922 versus shuffled 1.000 and surface macro 0.3798;
- CoReMA order +0.9268 bits/token versus lexical F1 0.5721 and HMM F1 0.3270.

Secondary solver outcomes, confidence intervals, thresholds, and retained failures remain in the experimental sections and complete supplement.

## Abstract strategy

The abstract now performs five functions: it defines the pre-decipherment problem, states the five-level evidence hierarchy, reports three decisive numerical contrasts, states supporting bounded results, and closes with the general inference and abstention standard. It no longer reproduces the entire programme ledger.

## Production audit

- Main and supplementary Markdown, DOCX, TeX, LaTeX PDF, and Word-layout PDF built.
- Four main DOCX tables checked for header repetition and protected rows.
- Supplement rebuilt in landscape after the first portrait rendering proved too narrow.
- Table 4 heading level corrected after accessibility audit.
- Uncertainty typography standardised to `±`.
- All main and supplementary pages visually inspected.
- Accessibility, style, heading, PDF-preflight, machine-verification, and checksum gates completed.

## Remaining work

1. Assemble the anonymised reviewer-accessible archive.
2. Clean-room recompute the generator audit and inherited cipher-family results.
3. Complete page-level audit of inherited historical references.
4. Resolve the exact formal Mauro reference only if it is to be named.
