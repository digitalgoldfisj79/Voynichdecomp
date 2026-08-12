# M19 Czech Diagnostic v1 — protocol freeze

Date: 2026-08-12
Parent archive: Cipher Coverage Closeout v1.
Status: diagnostic language extension only; this does **not** reopen the closed cipher research programme or alter its stopping-point verdict.

## Motivation

BnF M19 STA/aaa v1.9 was the prior cipher experiment in which **language identity itself** remained the live numerical discriminator after the inference instrument qualified. On the first binding RF STA-family H19 arm, the frozen ranking was Spanish > French > Greek > German > Latin > Hebrew > Italian > Arabic, but Spanish exceeded French by only 0.02232173435 nats/retained unit, below the frozen 0.05 margin. Therefore v1.9 correctly reported NO STA/AAA M19 SIGNAL and kept C19 sealed.

This experiment asks one bounded question: **where does Czech rank under exactly that already-qualified M19/STA mechanism and scoring pipeline?**

## What is not changed

- BnF numerical tables/emission law;
- RF/STA/AAA parsing and fixed source hashes;
- RF T/H/C split;
- vocabulary selection at family K=22, connected AAA K=26, full STA K=36;
- normalization (`unidecode`, lowercase, `j->i`, `v->u`, `w->u`, frozen 23-letter alphabet);
- LM construction and smoothing;
- v1.9 100,000-step paired-ensemble optimizer;
- convergence criteria;
- positive-control mapping-recovery thresholds;
- H19 coverage threshold 0.97;
- H19 top-language margin threshold 0.05;
- hierarchy/stopping rule;
- C19 remains sealed unless the full hierarchy would legitimately pass.

No Czech-specific substitution map, key, threshold, folio subset, representation, normalization or optimizer parameter may be selected after target inspection.

## Czech source

Official Universal Dependencies treebank: `UniversalDependencies/UD_Czech-CAC`.
Pinned repository commit:
`798f89716ae5a96e86042df7d394d56787e2e213`.

Pinned files at that revision:
- `cs_cac-ud-train.conllu` (Git blob `759377eabd739e06583d182707cff98dc6b4545b`);
- `cs_cac-ud-dev.conllu` (Git blob `861ec3a61a6d59349b140fc7d12a2b1ebae1064a`);
- `cs_cac-ud-test.conllu` (Git blob `e1e69afe9fd6b8a8e7e91c99e3e6b2b3642742a7`).

The Czech LM uses the train file with the same v1.7/v1.9 deterministic LM sentence-residue selection (`i % 10 in {3,4,8,9}`) used for every existing language. Czech qualification plaintext is drawn from dev+test exactly as the existing qualified-language controls are drawn from their dev+test files.

Modern Czech is a diagnostic language model, not a claim about fifteenth-century Czech orthography. Diacritics are transliterated through the already-frozen `unidecode` normalization; no Czech-specific orthographic normalization is introduced.

## Qualification gate

Before any Voynich Czech fit is permitted, Czech must pass a fresh positive-control qualification at **all three frozen representation sizes**:
- K=22 STA family;
- K=26 connected AAA;
- K=36 full STA.

For each K, reuse v1.9 support-complete-span selection, generated M19 numerical values, opaque surface homophones, paired optimizer, and held-out map recovery.

Czech qualification passes a scale only if:
- Czech ranks #1 among the nine language LMs on its own held-out Czech control;
- language margin >= 0.05 nats/unit;
- mapping accuracy >= 0.85;
- paired-map agreement >= 0.90;
- optimizer convergence passes;
- best-minus-true oracle score >= -1e-6.

The stricter original programme-wide median conditions remain historical properties of the already-qualified 8-language instrument; this extension adds a Czech-specific gate without rerunning or weakening them.

If Czech fails **any** representation-scale qualification, target scoring stops and no Czech Voynich score is opened.

## Target hierarchy

If Czech qualifies all three scales:

1. Run the RF H19 **STA-family K=22** language ranking with nine languages, using the exact frozen v1.9 target split and optimizer.
2. The original eight languages are rerun under the same executable so the Czech score is compared in-run, not spliced into a new run post hoc.
3. Family H19 must satisfy the same frozen v1.9 gate: coverage >=0.97, top-vs-runner margin >=0.05, top map agreement >=0.90, top convergence, all nine fits converged.
4. If family H19 fails, stop immediately; connected AAA/full STA target fits are **not** launched, mirroring v1.9 compute/stopping discipline.
5. Only if family passes may connected AAA H19 run; only if that passes may full STA H19 run.
6. C19 remains sealed unless all three representations independently pass H19 and select the same top language, exactly as v1.9 required.

## Interpretation

Possible outcomes:
- Czech ranks poorly: no Czech diagnostic signal.
- Czech ranks near top but margin gate fails: numerical curiosity only, not a candidate language.
- Czech becomes a separated top language at family H19 but later hierarchy fails: representation-sensitive diagnostic, not a language identification.
- Full three-representation hierarchy passes with Czech: this would be genuinely new diagnostic evidence and would justify reconsidering the closeout trigger, but still would not by itself decrypt Voynich.

No plaintext is inspected or emitted during H19.
