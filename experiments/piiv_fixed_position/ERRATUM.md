# Erratum: historical book number

The original protocol and branch name describe this as a *Polygraphia IV* fixed-position test. That book number is incorrect.

Inspection of the 1620 French edition (*Polygraphie et universelle escriture cabalistique*, Gallica ark:/12148/bpt6k5833371s) shows:

- the fixed-position invented-coverword table begins under **“Troisiesme livre de Polygraphie”** at Gallica frame 179;
- the table continues through the epilogue at frame 214;
- **“Quatriesme livre de Polygraphie”** begins at frame 215 and concerns transposition tables.

Accordingly, the experiment tests the fixed-second-character invariant of **Polygraphia III**, not Polygraphia IV.

This correction changes no data, model, candidate position, seed, null distribution, threshold, result, or scientific interpretation. The literal mechanism tested remains:

> one plaintext character per coverword, recoverable from the same internal character position—historically the second character—under a global monoalphabetic correspondence.

The frozen original `PROTOCOL.md` is retained unchanged for auditability. References to “Book IV” in the original protocol, path and branch should be read as “Book III”.
