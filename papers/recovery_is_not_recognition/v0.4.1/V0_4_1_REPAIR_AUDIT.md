# Recovery Is Not Recognition v0.4.1 Review-Repair Record

---

# Recovery Is Not Recognition v0.4.1

**Revision date:** 2026-08-01  
**Base:** v0.4 external-review build  
**Status:** review-repair build complete; external clean-room verification remains required.

This revision repairs the missing structural/generative experimental section, corrects the scope and entropy accounting of the onomancy example, records the Chardonnens research-copy checksum, verifies Edge and Bruton, fixes reference order and dash style, and rebuilds the manuscript and audit package.

---

# Changelog v0.4.1

## Structural repair

- Added a full results section for the structural and generative baseline.
- Reported the 37,465-token representation, 28.9% contextual entropy result, twenty-three-model hierarchy, 84-metric protocol, 81/84 split-half ceiling, 59/84 best fully generative score, and non-comparable transcription-conditioned upper bound.
- Moved the former orphan transition paragraphs into the new section and renumbered subsequent sections.
- Corrected the abstract’s description of the model panel.

## Historical and numerical repair

- Distinguished the 30-state residue capacity (4.9069 bits) from the diagram lookup.
- Limited exact one-bit language to the two complete 15/15 redactions.
- Reported primary scored counts and observed-entry entropies for incomplete redactions 2 and 3.
- Replaced “multiple schemes” with seven printed small-number columns, one incomplete.
- Added a verified Edge clause on transmission corruption and practical ambiguity.

## Reproducibility and editorial repair

- Recorded the Chardonnens research-copy filename, byte count, and SHA-256 with a provenance caveat.
- Corrected reference alphabetisation.
- Standardised the compatibility-to-recovery transition to an unspaced em dash.
- Rebuilt Markdown, DOCX, LaTeX, LaTeX PDF, Word-rendered PDF, verifier output, style report, accessibility report, hashes, and package ZIP.

---

# Paper Revision Protocol v0.4.1

**Paper:** *Recovery Is Not Recognition*  
**Protocol date:** 2026-08-01  
**Revision type:** consolidated external-review repair  
**Target:** *Cryptologia*

## Frozen inputs

1. Canonical v0.4 Markdown and compiled package.
2. Repository generator specification `Paper/S3_Generator_Hierarchy_v2_fixed.md` at base commit `a8a84e6b5acc2956288eeaec92191e3e736c703d`.
3. Repository extended-analysis specification `Paper/S9_Extended_Analysis.md` at the same commit.
4. ONOM brief and Chardonnens research copy.
5. Consolidated v0.4 review supplied on 2026-08-01.
6. Edge 2014 full publisher text and Bruton, Beloucif, and Megyesi 2026 arXiv record.

## Mandatory repairs

- Reconstruct the missing generability arm from frozen specifications; do not invent generator results.
- Separate fully generative and transcription-conditioned models.
- State the 84-metric scoring and tolerance protocol.
- Correct the redaction-completeness and entropy claims.
- Record the research-copy checksum without promoting it to publisher provenance.
- Correct reference order and dash consistency.
- Preserve all existing Voynich and class-level claim firewalls.

## Shipping gates

1. `verify_v041.py` returns `PASS`.
2. LaTeX compiles twice without unresolved fatal errors.
3. DOCX renders to page images and every page is visually inspected.
4. PDF renders are visually inspected.
5. Accessibility and style audits contain no high- or medium-severity findings.
6. SHA-256 manifest matches every delivered artefact.
7. Repository branch and draft PR are updated with text sources and audit records.

---

# New and Changed Claim Ledger v0.4.1

| ID | Claim | Class | Evidence | Boundary |
|---|---|---|---|---|
| G01 | The structural analysis used 37,465 parsed tokens and a four-slot representation. | MACHINE-CERTIFIED / inherited | Repository S3/S9 specifications | Descriptive corpus analysis. |
| G02 | Section, line position, preceding suffix family, paragraph status, and quire jointly accounted for 28.9% of decomposed-token entropy. | MACHINE-CERTIFIED / inherited | Repository S9 specification | Target-derived association, not causal identification. |
| G03 | The hierarchy contained twenty-three models: twenty-two fully generative comparators and one transcription-conditioned upper-bound model. | DERIVED-EXACT | Repository S3 model inventory | The upper-bound model is not comparable as a full generator. |
| G04 | The headline suite contained 84 metrics, with six Levenshtein subsampling measures reported separately. | MACHINE-CERTIFIED / inherited | Repository S3 scoring specification | Metric passes are heterogeneous compatibility checks. |
| G05 | Gen-SP passed 59/84 metrics, seed range 56–61; no other fully generative model exceeded 46. | MACHINE-CERTIFIED / inherited | Repository S3 result table | Not classification accuracy or a posterior probability. |
| G06 | Manuscript split-halves passed 81/84 under the same tolerance rule. | MACHINE-CERTIFIED / inherited | Repository S3 tolerance specification | Empirical ceiling, not theoretical maximum. |
| O01 | Seven printed small-number Latin letter-value columns are present; one is incomplete. | PUBLISHED / table inspection | Chardonnens 2007 Table 30 | “Columns” does not assert seven independent traditions. |
| O02 | Mod-30 reduction has a 30-state maximum capacity of 4.9069 bits before lookup. | DERIVED-EXACT | $\log_2 30$ | State-space capacity, not empirical input entropy. |
| O03 | Complete redactions 1 and 4 divide 30 entries 15/15 and have one-bit lookup entropy. | DERIVED-EXACT | ONOM closed-table count | Applies to the diagram lookup only. |
| O04 | Redaction 2 divides 29 scored entries 14/15 (0.9991 bits); redaction 3 divides 25 scored entries 12/13 (0.9988 bits) under the primary policy. | DERIVED-EXACT | ONOM closed-table count | Observed-entry entropy; missing/ambiguous cells prevent complete-table entropy claims. |
| O05 | A sensitivity policy scores 28 entries for redaction 3 without changing its two-class conclusion. | DERIVED-EXACT | ONOM sensitivity output | No inferential statistic. |
| O06 | Edge describes Latin-transmission corruption and argues that ambiguity could have practical value to physicians. | PUBLISHED | Edge 2014 full publisher text | Historical interpretation, not a Voynich mechanism claim. |
| R01 | The research copy has the recorded filename, byte count, and SHA-256. | MACHINE-CERTIFIED | Local hash recomputation | Identifies the analysed copy, not canonical publisher provenance. |

## Preserved exclusions

- No claim that the Voynich Manuscript uses onomancy.
- No fitted Voynich letter-value map.
- No inferential statistics on the closed diagram tables.
- No promotion of metric compatibility to historical mechanism identification.
- No class-level impossibility claim from a bounded solver or compressor failure.

---

# Reference Audit v0.4.1

## Verified in this revision

| Reference | Verification | Decision |
|---|---|---|
| Edge 2014, *Historical Research* 87(238):611–632, DOI 10.1111/1468-2281.12067 | Full publisher text inspected; metadata and conclusion checked | Retain; add bounded clause on transmission corruption and practical ambiguity. |
| Bruton, Beloucif, and Megyesi 2026, arXiv:2606.05078 | Primary arXiv record checked; title, authors, date, and shared-pool scope confirmed | Retain as a shared-code-pool positive result, not arbitrary fresh-key recovery. |
| Chardonnens 2007 | Research copy hash recomputed | Retain; archive note must distinguish copy identity from publisher provenance. |

## Checksum

- File: `Anglo_Saxon_Prognostics_900_1100_Study_a.pdf`
- Bytes: `9,570,658`
- SHA-256: `1a90e584399aa3627dc28588d0691265b2829b0191696a194d59733479d580f7`
- Boundary: this makes extraction and counts re-derivable from the analysed copy; it does not verify canonical distribution provenance.

## Reference-order corrections

- Barron 1998 now precedes Benedetto 2002.
- Chu 2020 now precedes Cilibrasi 2005.

## Remaining limitations

- Juste 2011 remains abstract-sourced for the specific sentence used.
- Sigerist 1942 remains unread and is not used for an inversion claim.
- The inherited reference list has not been comprehensively rechecked against every publisher record.

---

# Cold Review v0.4.1

## Decision

**Suitable for external review after archive assembly; not yet submission-frozen.**

## Resolved blockers

1. The generability arm now has an explicit corpus, representation, model inventory, scoring protocol, results, and interpretation.
2. The 59/84 result is scoped to the best fully generative model and is separated from the transcription-conditioned upper bound.
3. The onomancy calculation now distinguishes residue-state capacity from diagram lookup and complete from incomplete redactions.
4. Exact one-bit language is restricted to the two complete 15/15 diagrams.
5. The source-copy checksum, reference order, and dash inconsistency are repaired.

## Residual risks

- The structural/generator results are inherited from repository artefacts and have not been clean-room recomputed in this revision.
- Several earlier cipher-family outputs remain pending a unified reviewer-accessible archive.
- Juste 2011 is still used at abstract-level resolution.
- The manuscript is long for the target journal and may require editorial compression after scientific review.

## Claim discipline

The paper continues to distinguish exact, machine-certified, published, interpretive, and open claims. It makes no Voynich mechanism claim from onomancy, generability, compression, or CoReMA.

---

# Build and QA Report v0.4.1

**Build date:** 2026-08-01  
**Decision:** PASS

## Builds

- Canonical source: `manuscript_v0_4_1.md`
- DOCX: 53 rendered pages
- LaTeX PDF: 38 pages
- Word-rendered PDF: 53 pages
- LaTeX compiled twice with LuaLaTeX.

## Automated checks

- `verify_v041.py`: PASS (28/28 checks; 11,604 words; 71 headings).
- PDF preflight: openable, 38 pages, unencrypted, text-based, no XFA.
- DOCX accessibility audit: 0 high, 0 medium, 25 low findings; low findings are raw-link advisories.
- DOCX style lint: 0 direct paragraph-formatting exceptions; reported compact table-header cells are intentional.

## Visual QA

- All 53 DOCX-rendered pages inspected after the final no-row-split OOXML patch.
- All 38 LaTeX-PDF pages inspected.
- No clipping, overlap, broken glyphs, orphaned table cells, unreadable tables, or header/footer defects observed.
- The multi-page principal-results table repeats its header and keeps rows intact.

## Scientific boundaries retained

- Generator compatibility is not mechanism identification.
- The transcription-conditioned model is not compared as a fully generative system.
- The onomancy discussion does not propose a Voynich mechanism.
- No Voynich compression matrix was produced.
- Bounded negative calibrations are not promoted to class-level impossibility claims.

## Remaining external work

- Assemble the anonymised reviewer-accessible archive.
- Clean-room recompute inherited generator and earlier cipher-family outputs.
- Complete page-level verification of inherited references, including Juste 2011 beyond the abstract.

---

# Manifest v0.4.1

## Submission artefacts

- `manuscript_v0_4_1.md` — canonical editable manuscript source.
- `cryptologia_recovery_not_recognition_v0_4_1.docx` — Word submission manuscript.
- `cryptologia_recovery_not_recognition_v0_4_1.tex` — LaTeX source.
- `cryptologia_recovery_not_recognition_v0_4_1.pdf` — LaTeX-rendered submission PDF.
- `cryptologia_recovery_not_recognition_v0_4_1_word.pdf` — Word-layout QA PDF.

## Revision and audit artefacts

- `README.md`
- `CHANGELOG_v0_4_1.md`
- `PAPER_REVISION_PROTOCOL_v0_4_1.md`
- `CLAIM_LEDGER_v0_4_1.md`
- `REFERENCE_AUDIT_v0_4_1.md`
- `COLD_REVIEW_v0_4_1.md`
- `BUILD_REPORT_v0_4_1.md`
- `verify_v041.py`
- `VERIFY_RESULT_v0_4_1.json`
- `a11y_report_v0_4_1.json`
- `style_lint_v0_4_1.txt`
- `pdf_preflight_v0_4_1.txt`
- `SHA256SUMS_v0_4_1.txt`

## Excluded intermediates

Render PNGs/contact sheets, LaTeX auxiliary files, logs, and the copyrighted Chardonnens source PDF are not included in the distribution package.
