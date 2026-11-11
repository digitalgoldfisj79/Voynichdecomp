# Technical erratum — CoReMA Gr1 XML recovery in v0.6

**Date:** 2026-07-25  
**Scientific gates:** unchanged  
**Original reconstructed-runner SHA-256:** `213603fd38d99725cc99cb640bccb428eea0480de48298630dd3834426a49616`

## Defect

The first completed workflow treated the public `gr1.recipes/TEI_SOURCE` endpoint as unavailable even though it returned HTTP 200 and a 559,924-byte TEI document. Strict `lxml.etree.fromstring` rejected the document because the TEI header declares the XML ID `Klug2019` more than once. The duplicate declarations are XML validity/ID-uniqueness errors; they do not prevent recovery of the TEI element tree or the annotated recipe text.

The frozen protocol required every publicly accessible annotated-detail object and required unavailable endpoints to be audited rather than silently replaced. Excluding Gr1 on this basis was therefore a technical acquisition/parsing defect, not an admissible corpus decision.

## Repair

The original runner is still reconstructed and verified against its frozen SHA-256. A deterministic post-reconstruction patch then:

1. parses TEI with `lxml.etree.XMLParser(recover=True, huge_tree=True)`;
2. requires a recovered root element;
3. records every parser error in `parse_audit.xml_recovery_issues`;
4. applies the same recovery rule to direct acquisition validation; and
5. writes both original and patched SHA-256 values to `scripts/PATCHED_RUNNER_SHA256_v0_6.txt`.

The parallel wrapper, feature definitions, semantic-role precedence, manuscript groups, folds, estimators, random seed, metrics and all frozen thresholds are unchanged. The corrected calibration supersedes the 27-manuscript run. Git history and the initial workflow artifact preserve that preliminary output for audit.

## Acceptance criterion

The repaired run is valid only if:

- Gr1 is downloaded and parsed;
- the two duplicate-ID issues are recorded rather than suppressed;
- no unrecoverable parse failure is recorded for Gr1;
- the result is recomputed from scratch; and
- the Voynich transfer remains sealed unless the corrected run passes all original gates.
