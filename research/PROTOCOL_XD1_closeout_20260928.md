# XD1 closeout and historical-recipe control protocol

**Protocol ID:** XD1-CLOSEOUT-20260928  
**Frozen:** 2026-09-28, before any CoReMA lag-1/lag-2 repetition outcomes were computed.  
**Parent:** XD1_20260920.  
**Branch:** supergrammar-xd1-closeout-20260928.

## Retractions/corrections carried forward

1. XD1 v1 P2-P5 sequence results are invalid because line ordering could become 1,10,11,...,2. Only v2/order-fixed results are admissible.
2. The frozen v03 certifier independently retained a related line-order defect. v03.1 records that 646/3,893 ZLZI LINE_BREAK joins (16.6%) were physically non-adjacent. v03 remains frozen historical record; no silent mutation.
3. Nuremberg falsifies any claim that SPACE>LINE_BREAK attenuation is uniquely Voynich-specific. One chancery corpus does not establish that attenuation is universally ordinary.
4. The within-line repetition contrast is not a new discovery. It is a corrected external-control replication of earlier Voynich lag-1/lag-2 work.
5. No semantic, linguistic, cipher, or historical-mechanism conclusion is licensed by this protocol.

## Audit order

Interpretation is prohibited until the following checks are addressed in order:
circularity -> leakage -> confounds -> matched nulls -> control fairness -> measurement degeneracy -> representation dependence -> decision-rule fragility -> audit completeness.

## Phase A — infrastructure and missing XD1 cells

A1. Execute the existing numeric-order/segment-reset regression fixture at commit lineage fa67ee6/f20a03f.
A2. Re-run corrected P2-P5 for VMS ZLZI generic reference and both Gaskell-Bowern human pseudo-writing arms. Existing P1 rows are retained.
A3. Run corrected P3/P4 for Nuremberg Letterbooks 2-5; existing corrected P2/P5 rows are retained.
A4. No v1 P2-P5 row may be backfilled or interpreted.

## Phase B — historical recipe-register repetition test

### Control selection

Selection is based only on pre-existing SG89 source manifests, genre, physical-line preservation and sample size, not repetition outcomes.

Primary CoReMA controls (claim-bearing):
- A1 — 9,606 prior-manifest tokens / 799 physical lines.
- BS1 — 24,534 / 3,563.
- SO1 — 15,491 / 1,767.
- W1 — 20,836 / 1,914.

Primary gate: prior-manifest >=9,000 tokens and >=750 physical lines.

Secondary sensitivity controls, if source-QC passes and >=2,000 tokens / >=200 lines:
A1B_LATIN, BS2, KA1, KO1, W2.
KA3 is descriptive only because it is below the token gate.

The prior SG89 manifest is used only as source/QC metadata. SG89 scientific outcomes are not used to select witnesses.

### Source contract and QC

Source: current GAMS CoReMA TEI_SOURCE objects. Raw bytes and SHA-256 are recorded.

Tokenization is the already-frozen XD1 tokenizer:
- Unicode NFC;
- lower-case;
- letters/combining marks/digits retained inside tokens;
- punctuation/markup stripped;
- at least one Unicode letter required.

Physical folio/line boundaries are preserved from TEI milestones.

For each primary witness the fresh extraction must be within:
- 3% of the prior SG89 physical-line count; and
- 10% of the prior SG89 token count.

A primary witness failing QC is excluded from claim-bearing analysis. The parser may not be tuned after seeing repetition outcomes; a parser correction requires a new protocol version and a fresh run.

### Primary statistic

For each physical line and lag d in {1,2}:
observed = fraction of eligible token pairs with token[i] == token[i-d].

Null: independently permute the exact token multiset *within the same physical line*. This preserves line length, vocabulary, token frequency, and the complete per-line multiset.

Primary Monte Carlo: 200 deterministic permutations, matching the frozen XD1 bridge.
Cross-check: analytic permutation expectation from exact within-line token multiplicities.
Sensitivity: 2,000 deterministic permutations for primary witnesses.

Report together:
- observed rate;
- null mean;
- effect = observed-null;
- null SD;
- |effect|/null SD;
- observed/null;
- opportunities and line count.

If |effect|/null SD < 2, the conclusion must begin: **the metric does not resolve this**.

### Length-confound sensitivity

Repeat lag 1/2 in frozen line-length bins:
ALL, 2-5, 6-10, 11-20, 21+.
A bin is formal only with >=50 physical lines.

Additionally report a VMS-opportunity-weighted standardized control statistic across the fixed bins so different line-length mixtures cannot create the headline contrast.

### Sampling uncertainty / 95% bands

For VMS and each primary control, bootstrap physical lines within fixed length bins (2,000 deterministic replicates). Report the 2.5/97.5 percentiles for effect and observed/null.

A cross-corpus difference is considered resolved only when:
- the point difference in effect divided by the pooled bootstrap SD is >=2 in magnitude; and
- the VMS lag-2 point estimate lies outside the 95% bootstrap interval of every primary recipe witness.

### Frozen falsifier

The proposed VMS repetition discriminator fails if **any primary recipe witness** simultaneously has:
- lag-1 observed/null >= 0.90; and
- lag-2 observed/null >= 1.10.

This threshold was specified before the CoReMA repetition outcomes were computed.

Survival of this falsifier is necessary but not sufficient for promotion.

### Promotion rule

The repetition result may enter a future kernel only if all of the following hold:
1. no primary recipe witness triggers the frozen falsifier;
2. VMS lag-2 remains resolved under the same multiset null;
3. VMS-vs-control lag-2 differences resolve against every primary recipe witness under the bootstrap rule;
4. the opposition survives the formal 6-10 and 11-20 line-length strata where both sides have support;
5. an independent representation sensitivity does not reverse the result.

Otherwise SG21/SGK10-12 remain open/excluded.

## Phase C — representation dependence

Replay the surviving exact-repetition result under a non-EVA segmentation/transliteration (STA) if a canonical STA corpus with physical lines is available.

If STA is not recoverable, record **BLOCKED_MISSING_REPRESENTATION**. Do not substitute another EVA-family transcription and call it independent.

## Phase D — propagation audit

The XD1 numeric-order fixture becomes a mandatory regression test for every sequence-sensitive Super-Grammar loader/certifier. A candidate release cannot freeze if its loader fails the fixture or if it carries an untested duplicate line-order implementation.

## Stop rule

After Phases A-D:
- do not add new VMS-only grammar descriptors;
- do not add another generator or language merely to increase the control count;
- further work requires genuinely independent evidence or representation (crib/semantic anchor, new palaeographic units, or a separately qualified decipherment mechanism).

## Reproducibility

Every phase writes a separate pickle checkpoint before proceeding, plus canonical JSON/CSV summaries and SHA-256 hashes. A failed phase remains preserved; later phases do not overwrite it.
