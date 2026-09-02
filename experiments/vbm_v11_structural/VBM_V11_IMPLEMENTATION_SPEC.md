# VBM v11 — implementation specification

Date: 2026-09-02
Status: **FROZEN BEFORE V11 SCIENTIFIC OUTPUT**
Parent: `VBM_V11_STRUCTURAL_CONSTRAINTS_PROTOCOL.md`

This note resolves execution details without changing any V11 hypothesis, primary statistic, null family, gate, parser, or evidence firewall.

## Branch B split-half nulls

The primary TRAIN statistic and both TRAIN-A / TRAIN-B replication statistics each use the full **10,000** matched-null samples specified in the protocol. No reduced split-half null count is permitted.

## Branch C EMPTY context

The literal sentinel `EMPTY` denotes an empty nucleus and is assigned to the dedicated EMPTY context bucket. It is not folded into OTHER. This is an implementation correction made before any Branch-C result was exposed.

## Branch D familywise permutation statistic

For each language reference model, the permutation statistic is the **absolute held-out average log probability** of each morphology-to-length rule. Each null preserves that rule's global assigned length multiset while permuting the assigned lengths among TRAIN-seen nucleus types within occurrence-frequency deciles. The familywise null for a permutation is the maximum held-out score across D1–D5.

This is equivalent to testing whether the observed morphology-to-length association yields a held-out run-length sequence unusually compatible with the reference model, conditional on the rule's assigned length distribution. The protocol's word “improvement” refers to this permutation advantage, not subtraction of an additional baseline.

Because D1–D5 are deterministic morphology functions, they are evaluable for nucleus types appearing only in INTERNAL_HOLDOUT. Therefore the unpermuted rule map is defined for the union of TRAIN and HOLDOUT nucleus types. Null permutations alter only TRAIN-seen types, because frequency-decile strata are defined from TRAIN. HOLDOUT-only types retain their deterministic rule value in nulls. This prevents an unseen-type execution failure without introducing learned information from HOLDOUT.

## Branch E line interpretation

The primary sequential closure E1 is line-level: for each folio, take the final right half of the last valid segment on one nonempty transcription line and pair it with the initial left half of the first valid segment on the immediately following nonempty transcription line. Pairs never cross folio boundaries.

The cyclic E2 exploratory statistic pairs the final right half and initial left half of the same nonempty transcription line. E2 cannot rescue E1.

## Execution

The runner is split into two source files only to stay within repository-write limits. A launcher concatenates them in fixed order and executes the combined source. This packaging has no effect on scientific computation.

The programme is CPU-only. No paid GPU allocation is authorised.
