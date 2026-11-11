# E-LAUTE research-data access request — v0.6 closeout

**Prepared:** 2026-07-25  
**Purpose:** external historical-notation calibration for a recoverability-first study of unknown structured texts  
**Status:** ready to send; no data access is assumed until the project team replies.

## Public basis for the request

The E-LAUTE public edition describes a corpus of more than 2,000 pages of German-speaking lute tablature from 1450–1550 and reports continuing source releases through June 2026. Its public MEI conventions distinguish four aligned representation types:

1. diplomatic transcription in original German lute tablature (`enc_dipl_GLT`) with transliterations;
2. edited tablature encodings;
3. diplomatic Common Western Music Notation (`enc_dipl_CMN`);
4. edited CMN (`enc_ed_CMN`).

Public sources consulted:

- E-LAUTE edition platform: https://edition.onb.ac.at/context:elaute
- MEI conventions: https://edition.onb.ac.at/fedora/objects/o:lau.red-editionguidelines/datastreams/MEI_CONVENTIONS/content
- Project contact page: https://e-laute.info/contact/

## Proposed recipients

**To:** PD Dr. Kateryna Schöning — kateryna.schoening@univie.ac.at  
**Cc:** Dr. David M. Weigl — weigl@mdw.ac.at; DI Christoph Steindl — christoph.steindl@onb.ac.at

## Ready-to-send request

**Subject:** Research request for an E-LAUTE MEI snapshot for blinded recoverability calibration

Dear Dr. Schöning,

I am conducting a recoverability-first computational study of unknown structured texts. The methodological question is whether the operational variables of a historical notation can be recovered under strict source or manuscript holdout without relying on literal symbol identity. The study does not seek to infer musical or semantic values for an unidentified manuscript unless a method first succeeds on real historical positive controls.

E-LAUTE appears uniquely suitable because the project supplies diplomatic German lute tablature alongside aligned transliterations and Common Western Music Notation. I am writing to ask whether the project could provide, or point me to, a stable research snapshot of the machine-readable encodings for calibration.

The minimum useful package would contain:

- diplomatic German lute tablature MEI (`enc_dipl_GLT`);
- aligned diplomatic or edited CMN (`enc_dipl_CMN` and/or `enc_ed_CMN`);
- persistent source, manuscript/print and piece identifiers;
- page/system/measure or other alignment anchors;
- schema, controlled-vocabulary and editorial-convention versions;
- licence and citation requirements;
- a release tag, commit hash or dated manifest permitting exact reproducibility.

The planned test would group all folds by source document, train only on other documents, and assess recovery of documented fields such as rhythmic value, course/string, fret or pitch-equivalent event structure. Literal symbol identities would be excluded from the identity-neutral arm. Thresholds and analysis code would be frozen before inspecting held-out results. Only aggregate metrics and non-copyright-restricted diagnostics would be published unless the licence expressly permits row-level release.

I would be grateful for access to either the current internal research snapshot or the public repositories underlying the released edition. A partial snapshot is still useful provided source grouping and alignment are preserved. I am also happy to use an embargoed or non-redistributable copy and publish only hashes, protocols and aggregate results.

The research is independent and unfunded. I would acknowledge E-LAUTE and its funders and cite the edition and technical documentation in any resulting publication.

Yours sincerely,

Edward Stewart Anthony Bozzard  
ORCID: 0009-0002-4052-0994

## Requested response checklist

A useful reply would resolve:

- where the MEI files can be obtained;
- which encoding types are available per source;
- whether diplomatic and CMN versions share stable alignment identifiers;
- the licence governing files and derivative metrics;
- whether redistribution of a frozen research snapshot is permitted;
- the preferred edition citation and acknowledgement;
- a stable version identifier or date.

## Decision rule

Receipt of data would authorise a new, separately frozen E-LAUTE calibration. It would not reopen or revise the completed CoReMA v0.6 verdict and would not by itself authorise Voynich transfer.
