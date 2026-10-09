# LANG2 | Source-language fairness and shared normalisation
Date: 2026-10-09
Status: PREREGISTERED S0; CONFIRMATORY TESTS LOCKED
Owner: Voynich source-language programme (separate from active Residual Recovery Ladder)

## Retractions / exclusions (read first)
1. C-LANG one-epoch source-language verdicts are RETRACTED: synthetic positive control failed. Do not treat Latin, Padua or Czech null results as evidence against those sources.
2. The historical PGCS S6 consonant-grid ranking (CI Latin ranked ahead of Italian) is not a qualified independent Voynich language discriminator: it depends on the PGCS assignment. It may guide control discovery but cannot enter confirmatory scoring.
3. GER2 ALEM near miss was not promoted; validation effect=0.0534874108, null SD=0.0338745105, z=1.579; sealed effect=0.0539532956, null SD=0.0217382874, z=2.482; full gate FAIL.
4. Do not equate a distributional collapse after many-to-one family recoding with evidence of a multilingual upstream layer.

## Controller independence
The active Recipes residual-recovery ladder has a frozen controller:
`voynich_vms_recovery_ladder_controller_20261008`.
It explicitly locks source-language interpretation. LANG2 is an independently
authorised, prospectively frozen branch; no R4 instruments, results, or
hyperparameters are reused or changed. The main controller stays untouched.
Any shared Voynich test folds require a firewall against selection leakage.

## Scientific questions
H1: one historical language family dominates upstream text.
H2: >=2 distinct source languages, and one source-invariant transformation T,
generate Voynich-like output while retaining section/Currier residuals.

These hypotheses are NOT exhaustive. Unknown artificial sources, non-monotonic
cipher, nonlinguistic origin and unspecified mechanisms remain live alternatives.

## Test order (do not invert)
S0 A priori corpus ledger: witness identifiers, original language, date,
medical/scientific/recipe genre, region, transcription channel, independent
manuscript group, license, exact source SHA, tokeniser SHA, OOV and type support.
No inferred place-of-holding-as-origin. Author/title variants de-duplicated.
Primary strata require >=3 independent witnesses per source-language arm;
independent witness = manuscript/textual tradition, never arbitrary pages.
Named source labels must be independently catalogued. Padua mixed Latin/
Romance cannot be relabelled as pure Tuscan. Only independently transcribed
period medical/scientific texts are confirmatory; Dante, modern Italian and
classical Latin are sensitivity/diagnostic only.

S0a Frozen VMS reproduction: pinned ZLZI SHA
26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f
and physical fold-map SHA
e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888.
Use original GER2 cache/loader; 5 folds (discovery 2/3; validation 4;
sealed final 0/1), +P0 strict, line-entry first two excluded. The audit
here obtained 24109 physical-pair mapped events, not the archived GER2 21763:
do not assume reproduced or train until this discrepancy is diagnosed.
No independent transcript or post-hoc folio reassignment.

S1 Instrument qualification per language (before ANY Voynich candidates):
known-source -> hidden source-to-target edit/substitution controls from held-out
historical manuscript witnesses, plus both synthetic fixed one-to-one glyph
substitution and harder orthographic/allographic transformations.
Use frozen full 10-round, 150-epoch NeuroCipher EM/mincost solver, shared
demand/capacity/checkpoint/seed schedule (1234, 2026, 17, 73).
Four independently assigned null scrambles (9001,7001,11003,17011).
Positive controls must recover actual held-out gold word/cognate mappings
on unseen source types and reject matched scrambled source; calibrate power
and false-positive rates in a second blind panel. Passing GER15 calibration
is NOT proof of adequate Latin, Italian or French power.
Any arm failing S1 is UNRESOLVED, not rejected.

S2 Fair direct-transduction tournament (only S1 qualified arms):
Latin medical + university, central-Italian Tuscan/vernacular medical,
Bavarian, Alemannic, historical French/Franco-Italian medical.
Czech/Slavonic is secondary after quality gate.
Same VMS physical folds, top 735 discovery novel, 735 validation novel,
735 sealed strict novel, target 4103 train/10000 final if support allows.
All candidates must satisfy exact support thresholds; no quietly reduced K.
Same solver, 4 paired training seeds, nulls, restart/checkpoint budget,
and full source metadata. No decoded-word inspection until adjudication.

Per-language REAL vs within-word exact-anagram NULL for both fold4 and
sealed strict final. Primary effect d_L = mean(null MCF cost) -
mean(real MCF cost), scale s_L=sample SD of null-cost runs; z_L=d_L/s_L.
Preregister existing GER2 promotion gates on BOTH folds, >2 null SD,
>=3/4 matched paired signs, no heldout-tail selection.
Cross-language comparisons use normalized effects z_L / blinded
manuscript-level bootstrap contrasts, NEVER raw NLL or raw edit cost.
Bootstrap re-samples manuscript witnesses, not pages or words.
Run robustness to target-vocab rank perturbation, mode/genre frequency,
dialect spelling normalisation and transcription variants.
If source arms share a witness, block jointly and flag dependence.
All five arms must have comparable power or language ranking ABSTAINS.

S3 Shared-normalisation discriminant (separate from S2):
Define a SINGLE shared encoder T across all source languages, with
identical mapping code and parameter count per source. No language-ID
branch within T. Fit T and language-specific source distributions on
synthetic/paired (known-input, known-output) positive controls ONLY.
Freeze T before any VMS heldout scoring.
Compare on exactly matched bifolium blocks:
  A single-language source + shared T;
  B two-or-more-language source mixture + same T;
  C one-language matched-mixture-capacity null;
  D source labels scrambled within genre/manuscript blocks;
  E trivial many-to-one collapse with matched output entropy;
  F root/FORM family-only model with equal complexity budget;
  G language-specific T (diagnostic, NOT a valid shared T).
Same token count, source-document count, trained parameter budget,
genre mix and optimization/restarts. Primary test is heldout
log-predictive gain of B vs best A after penalising mixture complexity,
standardised on matched manuscript-level nulls; report effect and
null SD in same sentence and lead 'the metric does not resolve this'
whenever |z|<2. Physical-fold consistency mandatory.

To qualify a 'normalisation' claim, T must SIMULTANEOUSLY:
(1) account for Voynich ED1/ED2, length, boundary structure and OOV;
(2) make independent source languages more alike at the surface than
pre-transform, beating entropy-matched random collapse;
(3) retain independently recoverable source-specific residual information
in positive controls and correctly recover hidden source labels;
(4) reproduce real section/Currier residual effect sizes within calibrated
null intervals, without using those labels when fitting T;
(5) pass independent manuscript, random-seed, phonographic and
transliteration representation checks.

The specific causal test is heldout B-vs-best-A under a FROZEN T with
genre/manuscript matched nulls, then prospective validation of source
assignments on independent iconography/label evidence without using it
to learn assignments. Mere improved surface likelihood is not source
identification; without aligned plaintext, inverse mixture is generally
nonidentifiable. A model may support *viability* of a shared renderer,
not prove multilingual origin or locate a compiler in Siena.

## Concrete data status, 2026-10-09 S0
- VMS ZLZI SHA verified; full GER2 fold geometry NOT locally reproduced.
- Circa instans Latin archive SHA 377daa2...; 52004 stored words
  (51999 ASCII-filtered in local preflight), one identifiable source.
- Padua Transkribus export SHA 63c6f70...; 263 XML pages/5735 HTR
  lines/51459 locally tokenised words; MIXED Latin and vernacular.
- German 15c ReF BAV/ALEM qualified legacy comparanda.
- French: CHrOMed lists 1,665,663 words of medieval French medicine;
  access/licence and extractable independent witnesses NOT yet verified.
- Italian: CorTIM lists 202 central-Italian dialect texts through 1400s;
  suitable medical witnesses/download rights NOT yet verified.
- Latin/French/Italian: CoMMA contains large HTR multilingual material,
  but modelled HTR and manuscript dependencies require QA and licensing.
All named external corpora are DISCOVERY LEADS, not ingested.

## Execution discipline
1. Every S0 witness audit atomic and persisted as pickle + JSON.
2. Preregister control pass/fail and blind seeds BEFORE opening VMS.
3. Source-specific confounds: date, transcription pipeline, genre,
   manuscript, regional spelling, source size, lexicon/rank, vocabulary
   entropy, grapheme/phoneme representation, paper/ink/HTR noise.
4. Do not launch expensive full 5-arm runs with <3 true witnesses per
   principal arm or without S1 multilingual qualification.
5. All negative gates recorded as explicit retractions at file top;
   zero post-hoc redefinition of control power or folds.
6. No interpretive use of locations/regions until S2 or S3 succeeds.