# Cipher Coverage Closeout Programme v1 — Gate 1 coverage audit

Date: 2026-08-09
Repository: `digitalgoldfisj79/Voynichdecomp`
Branch: `experiment/cipher-coverage-closeout-v1-20260809`
Audit base: Tranchedino/STA closeout commit `ae1e5b4cc879a3ab1a8081649157f2b5b12654b8`

## 1. Purpose and classification rule

This audit asks whether a materially distinct, historically defensible cipher mechanism circa roughly 1400–1450 remains inadequately tested against Voynich. It does **not** treat a solver branch being administratively "closed" as equivalent to a mechanism having been tested negative.

Classification is deliberately conservative:

- **CLOSED** — the mechanism/instrument passed the relevant prospective control gate and then produced a legitimate negative target result, or a target-independent structural contradiction sufficient to reject the frozen mechanism.
- **PARTIAL** — meaningful members/components were tested, but the mechanism did not receive a qualified target test, usually because recovery/recognition/identifiability failed first.
- **UNTESTED** — a materially distinct, finite mechanism remains with independent historical justification and a falsifiable prediction.
- **NON-IDENTIFIABLE** — the proposed freedom is broad enough that arbitrary text/encoding can be fit without external information.

A PARTIAL row is not automatically a licence to reopen it. If the only missing step exists because the old instrument failed its preregistered gate, it remains scientifically blocked unless a materially new, independently motivated instrument exists.

## 2. Mechanism × experiment coverage matrix

| Mechanism | Repository evidence | Classification | Gate-1 disposition |
|---|---|---|---|
| Simple/fresh monoalphabetic substitution | Recoverability frontier v0.5.1: large locked synthetic recovery PASS. Terminal v0.6 Stage 7 family-recognition gate then failed before any Voynich family application. No legitimate generic mono target result located in audited branches/results. | **PARTIAL** | **SURVIVES as M0 evidence-bridge baseline.** Do not redevelop the solver; only calibrate absolute target evidence prospectively. |
| Fresh-key generic homophonic substitution | v0.5.2/v0.5.3: favourable basins, but locked reliability gate failed; no Voynich target. | **PARTIAL** | Blocked. No optimiser/restart/neural retuning under Closeout v1. |
| Fixed historical homophonic substitution | Tranchedino f69v 36-sign H21 control-qualified; Voynich failed the absolute preregistered floor. | **CLOSED** | Negative. Do not reopen. |
| Shared/reused homophone pool | v0.5.3 positive control recovered essentially/exactly; this is an easiness control, not a Voynich result. | **PARTIAL** | No separate terminal arm; historical fixed-table evidence is already represented by Tranchedino/STA work. |
| Nulls represented by dedicated cipher signs | Tranchedino v2.3/v2.4 historical models include null signs; mapping-recovery qualification failed before target. MDL v0.7 also included null-bearing renderer controls but failed source-transfer qualification. | **PARTIAL** | Blocked as a generic null-code family. |
| Deterministic positional null insertion using otherwise ordinary cipher symbols | Family G tests sparse payload extraction (every nth selected position / grilles), not the complementary dense-message + scheduled-null construction. Tranchedino nulls are code-value nulls, not position-disambiguated ordinary symbols. | **UNTESTED** | **SURVIVES narrowly as NQ**, subject to exact historical rule freeze and synthetic qualification. No arbitrary null placement. |
| Arbitrary/content-dependent null placement | No bounded rule identifies which occurrences are null without plaintext/external information. | **NON-IDENTIFIABLE** | Exclude. |
| Nomenclators / opaque whole-word code symbols | v0.5.4 component gates failed: sparse codebook observations were not identifiable and residual key search was bimodal. Tranchedino historical nomenclator variants also failed recovery gates. | **PARTIAL** | Blocked; no codebook enlargement or post-hoc lexical fishing. |
| Geminate units | Explicitly represented in Tranchedino v2.3/v2.4 historical templates alongside letters/nulls/codes. | **PARTIAL** | No distinct terminal mechanism: gemination is a component, not a separate surviving family. |
| One visible sign → syllable / mixed plaintext unit | Tranchedino v2.4: historical one-sign alphabet+CV syllabary+geminate+null+lexical model passed feasibility but failed binding mapping-recovery qualification, even with oracle assistance. | **PARTIAL** | Blocked. Do not reopen Tranchedino. |
| Multiple visible glyphs → one plaintext unit; variable 1–3 visible-symbol groups | Terminal Family S explicitly implemented variable visible 1/2/3-symbol code-unit lattice and failed development. Family S protocol also covered digraphs, mixed characters/syllables, ambiguous separators and limited code units. | **PARTIAL** | **Candidate 3 killed.** No materially new polygraphic direction remains from the proposed gap. |
| Periodic changing alphabets | Family P executed periodic mode and achieved very high known-family recovery and locked synthetic success. Stage 7 family recognition failed before target. | **PARTIAL** | No target inference. Historically outside the primary 1400–1450 Western anchor if interpreted as Alberti-style changing alphabets; do not reopen. |
| Line-reset periodic alphabets | Family P executed `line_reset` and passed known-family locked recovery; Stage 7 recognition failed. | **PARTIAL** | Same disposition as periodic P. |
| Irregular/signalled changing-key alphabets | Listed in the high-level v0.6 protocol but not implemented in the executed Family P generator/solver, which contains only periodic and line-reset modes. | **UNTESTED** computationally | **Killed at historical gate for Closeout v1 primary window.** Alberti's `De componendis cifris` is 1466–1467; no independently established pre-1450 state/reset mechanism of this type was found. |
| M19 / STA family | v1.9 qualified K22/K26/K36 synthetic controls. First legitimate held-out target arm K22 ranked Spanish first but only ~0.0223 nats/unit over French versus frozen ≥0.05; later target arms cancelled by stop rule. | **CLOSED** | Negative. |
| Tranchedino programme overall | v2.0–v2.4: direct fixed homophony, variant carriers, mixed-unit historical templates and f134v–135r syllabary; final programme closeout records no qualified Voynich signal and prohibits retuning. | **CLOSED** for the tested historical templates | Negative. Do not reopen by optimiser, stratum, per-section key or representation shopping. |
| Fixed repeated block transposition | Recoverability v0.5.5: oracle components passed but joint development failed; no target. | **PARTIAL** | Blocked. |
| Repeated columnar transposition, global or line-reset | Terminal Family T actual frozen implementation: widths 5–10, global/line-reset repeated columnar; development failed. | **PARTIAL** | Blocked. |
| Geometric route / turning grille / arbitrary path | Family T's actual frozen protocol explicitly did not authorise these despite broader wording in the umbrella v0.6 protocol. | **UNTESTED** computationally | No primary 1400–1450 historical anchor established in this audit; arbitrary routes are also parameterically dangerous. Exclude unless new external evidence appears. |
| Within-word/local transposition | Not present in v0.5.5 or executed Family T. Independently documented in medieval Arabic cryptological tradition (Ibn al-Durayhim; preserved/discussed by al-Qalqashandi and modern scholarly histories). | **UNTESTED** | **SURVIVES as TQ.** This is the principal genuinely new historical mechanism. |
| Composite local transposition + substitution | Existing repo tested fixed block/columnar + mono, not within-word permutation + substitution. Arabic cryptological sources explicitly describe composite/super-ciphering by transposition + substitution. | **UNTESTED** | **SURVIVES as TQ-composite**, bounded to historically specified permutations and one global substitution key. |
| Bounded positional/steganographic carriers | Terminal Family G: oracle carrier recovery strong, blind G2 failed AUROC/recovery gate after sole amendment; no target. | **PARTIAL** | Blocked. Arbitrary carrier search is non-identifiable. |
| Arbitrary semantic/folio-specific steganography | Explicitly excluded by Family G and cannot be multiple-testing calibrated without independent carrier information. | **NON-IDENTIFIABLE** | Exclude. |
| Morpholocal latent-order detector | v0.3.4 generator-disjoint smoke: 2/16 positive sensitivity, 10/16 ordered-control false positives; stopped before large validation/Voynich. | **PARTIAL** as recognition instrument | Closed instrument line; not cipher-family coverage. |
| Frozen MDL homophony accounting | Synthetic equal-class homophony policy validated, but scope explicitly excludes mixed units/nulls/multiple keys etc. | **PARTIAL** as instrument | Use only within validated scope; do not count as a broad cipher negative. |
| Cipher-generated/source-transfer MDL v0.7 | Even with true mapping supplied, leave-source-out transfer gate failed; no historical panel or Voynich. | **PARTIAL** as recognition instrument | Closed comparator line; not cipher-family coverage. |
| Compression-transfer recognition | Stage calibration failed before Voynich; it is a recognizer/distance layer, not a generative cipher mechanism. | **PARTIAL** as recognition instrument | Closed instrument line. |
| CoReMA / procedural-semantic recoverability | Calibration/transfer programme closed before Voynich; not a historical cipher primitive. | **PARTIAL** as recognition instrument | Do not count as cipher-family coverage. |
| BnF numerical/onomancy coarse piecewise keys | v0.4 K-CURRIER, K-SECTION, K-SECTION×CURRIER passed synthetic controls and failed target language-separation criterion; K-QUIRE failed controls. | **CLOSED** for the qualified coarse partitions; **PARTIAL** for quire | No further arbitrary partition refinement: increasing flexibility elevated all languages. |
| BnF unmarked deterministic schedules | v0.6 structurally rejected all 25 frozen schedule×rotation models before language fitting. | **CLOSED** | Structural negative. |
| BnF marked M57 single-value codes | v0.5 failed the first binding control catastrophically; job cancelled; no target. | **PARTIAL** | Instrument not qualified; no target inference. |
| Hidden/free table choice, arbitrary fine per-section/per-folio keys | Finer partitions rapidly become flexible enough to fit many languages; no externally fixed state sequence. | **NON-IDENTIFIABLE** | Exclude. |

## 3. Audit corrections to the inherited narrative

### 3.1 Family T did not actually close route/local transposition

The umbrella terminal protocol named line-local route/grille systems, but the actual frozen Family T protocol restricted the executed family to repeated columnar transposition and explicitly excluded geometric routes, turning grilles, Cardan grilles and arbitrary route ciphers. Therefore "Family T closed" cannot be used as evidence that all local transposition was tested.

### 3.2 Family P did not actually execute irregular/signalled state schedules

The umbrella protocol named irregular and signalled state changes, but the executed generator/solver implemented only `periodic` and `line_reset`. The known-family recovery success applies to those two modes only.

### 3.3 Generic monoalphabetic substitution is not a legitimate Voynich negative

v0.5.1 demonstrates strong recoverability on fresh controls. The later blind family-selection programme failed before any Voynich family was admitted. In the audited repository no separate legitimate generic-mono Voynich target result was located. Thus the mechanism is PARTIAL, not CLOSED.

### 3.4 Recognition failures are not mechanism negatives

Compression-transfer, morpholocal, CoReMA and source-transfer MDL failures establish limitations of those evidence instruments. They do not independently reject every cipher renderer used as a positive control.

## 4. Historical screen performed after the repository audit

No historical mechanism was added to the computational gap list before the repository scope was fixed.

### 4.1 Medieval Arabic local transposition — passes historical gate

Primary/scholarly tradition: Ibn al-Durayhim (1312–1361), `Miftah al-kunuz fi idah al-marmuz`, with the broader tradition preserved/discussed by al-Qalqashandi's `Subh al-a'sha` (completed 1412). C. E. Bosworth, "The Section on Codes and Their Decipherment in Qalqashandi's Subh al-A'sha," *Journal of Semitic Studies* 8.1 (1963), 17–33, is a key scholarly translation/discussion.

Treccani's *Storia della Scienza* synthesis of the Arabic manuscript tradition gives explicit finite transposition rules, including:

- reverse every word;
- move last letter to first;
- swap first/last in a bounded word permutation;
- alternating outside-in word order, symbolically `1234567 -> 1726354`;
- the complementary outside-in order `1234567 -> 7162534`;
- skip/vertical transposition equivalent to fixed-width columnar reading.

The last item overlaps existing Family T and is therefore **not** a new gap. The within-word rules do not.

The same source explicitly describes composite/super-ciphering using transposition followed by substitution. Therefore testing global mono substitution composed with a historically specified within-word permutation is not a modern invented hybrid.

Historical sources:

- C. E. Bosworth (1963), DOI `10.1093/jss/8.1.17`.
- Treccani, *La civilta islamica: condizioni materiali e intellettuali. Criptologia e criptoanalisi*, Storia della Scienza.
- David Kahn, *The Codebreakers*, discussion of al-Qalqashandi/Ibn al-Durayhim cipher classes.

### 4.2 Stateful changing alphabets — fails primary-period historical gate

The executed Family P leaves an abstract computational gap for irregular/signalled alphabet changes. But the independently established Western anchor for a genuine changing-alphabet disk is Leon Battista Alberti's `De componendis cifris`, composed 1466–1467. That is outside the primary circa-1400–1450 window.

No pre-1450 source was found in this audit that specifies the proposed observable-local-structure changing-alphabet state machine. Medieval Arabic multiple substitutes/homophony should not be relabelled as a stateful changing alphabet without evidence.

Accordingly this family is **not** admitted to Terminal Cipher Programme v1. It can be reconsidered only if a pre-1450 source independently supplies a concrete state/reset rule.

### 4.3 Polygraphic direction — does not survive

The Arabic repertoire includes multi-letter encodings and letter-pair constructions, but the proposed computational direction (multiple visible glyphs encode one plaintext unit) is already the central Family S direction. Tranchedino v2.4 covers the inverse one-visible-sign-to-multi-letter direction. There is no remaining justification for another generic polygraphic search.

### 4.4 Deterministic inserted nulls — narrow surviving secondary gap

Arabic cryptological sources also describe inserting extraneous letters according to fixed within-word or periodic rules. This is not identical to:

- a dedicated cipher symbol whose global code value is NULL (Tranchedino), or
- Family G's sparse-message carrier model, whose preregistered regular rule extracts every nth position as payload rather than treating most positions as message and a small scheduled subset as ambiguous ordinary-symbol nulls.

A finite position-driven insertion rule is therefore retained as **NQ**, but only in source-specified bounded forms. Arbitrary null placement is NON-IDENTIFIABLE; omission/lossy deletion is excluded from recovery claims because the missing characters are not uniquely recoverable without external plaintext information.

## 5. Gate 1 decision

The audit does **not** justify broad reopening of cipher search.

Three bounded items survive to protocol design:

1. **M0 — simple monoalphabetic evidence bridge.** Not a new solver family. Reuse the already qualified fresh-mono recovery machinery; construct an absolute historical/synthetic evidence calibration that can legitimately admit or reject a sealed target test.
2. **TQ — medieval local transposition.** Test only exact within-word permutation rules independently documented in the Arabic cryptological tradition, first as pure transposition and then as the historically attested composite with one global substitution alphabet.
3. **NQ — deterministic scheduled null insertion.** Test only fixed source-specified insertion schedules that are materially distinct from dedicated-null codes and sparse steganographic carriers.

The originally proposed third candidate (generic polygraphic visible grouping) is killed. The proposed irregular/stateful changing-key candidate is computationally incomplete but excluded from v1 by the historical gate. Arbitrary routes, arbitrary nulls, free section keys, semantic carriers and unconstrained codebooks remain NON-IDENTIFIABLE.

This is **Decision A at Gate 1 only**: genuine uncovered historical mechanisms remain and warrant one bounded terminal calibration programme. It is not permission to inspect Voynich. Voynich remains sealed until each relevant arm independently passes its binding control and recognition/recovery gates.

## 6. Mandatory next step

Freeze `TERMINAL_CIPHER_PROTOCOL_V1.md` before implementing or scoring any target arm. The protocol must preregister:

- exact historical transformations;
- plaintext corpora and period-language controls;
- representations and segmentation assumptions;
- nuisance parameters and search bounds;
- recovery versus recognition metrics;
- absolute positive-control calibration and negative controls;
- one-way stop rules;
- sealed Voynich partitions;
- no plaintext inspection before qualification;
- no post-hoc parameter restriction;
- no per-section key freedom;
- bounded compute and immediate termination of failed arms.
