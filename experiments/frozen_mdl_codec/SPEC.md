# Frozen MDL Codec v0.1

Status: **operational candidate**  
Date: **2026-07-14**  
Seed for all bundled stochastic tests: **20260714**

This specification defines a deterministic accounting layer for comparisons between bounded cipher models and bounded production models. It does not decide whether the Voynich manuscript is encoded or generated. Its purpose is narrower: prevent a table, codebook, state machine, plaintext model, hidden path, duplicated key, or exception list from being included or excluded from the bill after a result is known.

The codec exposes two independent conventions:

- **H** — literal historical key-sheet serialization;
- **I** — enumerative structure plus Krichevsky–Trofimov universal categorical coding.

Each is evaluated in two modes:

- **full** — the surface inventory and its spellings are charged;
- **conditional** — the observed surface inventory is treated as given, but its partition, mappings, state machinery and selection rules remain charged.

The four reported model costs are therefore:

- `H-full`;
- `H-conditional`;
- `I-full`;
- `I-conditional`.

No comparison is called robust unless all four conventions produce the same sign.

## 1. Evidential status

The implementation and synthetic tests may be used internally once committed and hashed. They are not externally reviewed. A result produced with this codec should be labelled:

1. **internal** after the initial run;
2. **provisional-strong** only after independent third-party LLM demolition and reproduction;
3. **externally reviewed** only after independent human review.

## 2. Non-negotiable rules

1. The codec, registry, schema and decision rule are frozen before a manuscript comparison.
2. Any implementation correction creates a new version and forces a complete rerun.
3. Unknown model fields are rejected.
4. Floating-point fitted parameters are rejected. Stochastic choices are represented by non-negative integer multiplicities or counts.
5. Surface forms, plaintext units, contexts, states and keys use contiguous integer identifiers. Identifiers are structural references, not free semantic labels.
6. Every non-null surface form belongs to exactly one plaintext class within each key. Null forms are separately declared. Together they must partition the surface inventory.
7. Each duplicated key is serialized and charged in full.
8. Hidden paths are either marginalized exactly or transmitted explicitly. An uncharged Viterbi path is prohibited.
9. An external plaintext model, lexicon, context definition or production constraint must occur in the frozen registry before a run.
10. A post-run change to the external registry invalidates the run because the registry hash is embedded in every serialized envelope.

## 3. Model schema

A model contains:

- `surface_inventory`;
- `plaintext_inventory`;
- `contexts`;
- one or more `keys`;
- `key_partitions`;
- a `plaintext_model`.

### 3.1 Surface inventory

Each surface item has:

- contiguous integer `id`;
- literal `form`;
- optional integer `house`;
- optional integer `gallows`.

In `H-full` and `I-full`, these fields are charged. In conditional modes, only the inventory cardinality is charged; all surface spellings and derived labels are treated as observed.

### 3.2 Plaintext inventory

Each plaintext unit has:

- contiguous integer `id`;
- a frozen unit type: character, syllable, morpheme, word, phrase, number, category or mixed;
- either a literal string or an index into a frozen external lexicon.

Literal plaintext entries are transmitted verbatim. External entries are charged by bounded registry and entry indexes.

### 3.3 Contexts

Contexts are either literal or externally registered. A context can denote line position, section×Currier stratum, previous-length bin, house, or another preregistered conditioning variable. The codec does not infer context semantics.

### 3.4 Keys

Every key contains:

- a codebook partition;
- a declared null set;
- a finite state inventory and initial state;
- integer transition-count rows;
- sparse integer emission-count rows;
- reset rules;
- references to frozen production constraints.

Emission support is restricted by the codebook. A failed held-out case may not be repaired by adding a new homophone or exception.

### 3.5 Key partitions

A partition maps a preregistered selector to one key. Supported scopes are global, section×Currier, hand, quire and explicit. The selector itself is externally registered.

A hand-specific key is therefore not a free flag. It requires an externally frozen selector and a complete separately charged key.

### 3.6 Plaintext models

A plaintext model is either:

- an externally registered model; or
- an embedded categorical n-gram model of order 1–3.

Every embedded n-gram row is charged with the KT universal code. An externally registered model is charged by its bounded registry index. Post hoc model search requires a larger frozen registry and therefore a larger selection cost.

## 4. Convention H: historical key-sheet serialization

Convention H is an actual prefix-free bitstream.

### 4.1 Primitive codes

- Non-negative integers use Elias delta on `n + 1`.
- Bounded integers use a fixed width of `ceil(log2(cardinality))` bits. Unused codewords are invalid.
- Strings use Elias-delta byte length followed by UTF-8 bytes.
- Lists use a coded item count followed by their canonical items.
- Optional integers use one presence bit followed by the integer when present.

### 4.2 Envelope

Every serialized object contains:

- magic bytes `FMDL01\0`;
- one mode byte;
- an unsigned 64-bit payload-bit count;
- the first 128 bits of the frozen registry SHA-256;
- the padded payload.

The constant envelope is excluded from model-cost comparisons. It is retained for framing, registry verification and concatenation safety.

### 4.3 Canonical ordering

The following are sorted before transmission:

- codebook rows by plaintext id;
- transition rows by state and context;
- emission rows by state, context and plaintext id;
- sparse emission counts by surface id;
- reset rules by scope and state;
- instruction references by opcode and registry index.

Surface, plaintext, context, state, key and partition inventories must already have contiguous IDs in order. Reassigning these IDs after seeing a result constitutes a new model.

### 4.4 Full and conditional modes

`H-full` transmits literal surface spellings and optional surface labels.

`H-conditional` transmits the number of surface types but not their spellings or labels. It still transmits:

- every plaintext unit;
- every codebook assignment;
- null membership;
- every state and count row;
- reset and production instructions;
- key partitions;
- the plaintext model.

The conditional mode is intentionally favourable to a cipher explanation and acts as a lower-bound accounting convention.

## 5. Convention I: enumerative structure plus KT

Convention I avoids an arbitrary floating-point precision convention.

### 5.1 Codebook partition

For `V` surface types, `N0` nulls, `P` labelled plaintext classes and positive class sizes `h1…hP`, the codebook cost is:

```text
log2 C(V, N0)
+ log2 C(V - N0 - 1, P - 1)
+ log2 ((V - N0)! / product_i hi!)
```

The second term is zero for one plaintext class. This charges:

1. which forms are nulls;
2. the class-size composition;
3. the assignment of payload forms to labelled classes.

### 5.2 State topology

For each state/context row with `S` possible destinations and `d` non-zero destinations, Convention I charges:

```text
L_delta(d + 1) + log2 C(S, d)
```

in addition to the row identifiers. The transition counts then receive a KT code over all `S` destinations.

### 5.3 Universal categorical code

For counts `n1…nK`, `N = sum_i ni`, the KT codelength is:

```text
-log2 [ Gamma(K/2) / Gamma(N + K/2)
        × product_i Gamma(ni + 1/2) / Gamma(1/2) ]
```

It is used for:

- transition probabilities;
- emission selection within each legal codebook support;
- embedded plaintext n-gram rows.

Zero counts remain part of the categorical support and are not silently removed.

### 5.4 External model selection

An external registry pool of size `R` costs `ceil(log2 R)` bits for the selected model. A pool of one costs zero selection bits. This means a broader postulated plaintext search is automatically charged when the candidate set is frozen honestly.

## 6. Hidden paths

For hidden-state or variable-alignment models, one of two routes is admissible:

1. exact marginalization by a forward or equivalent dynamic programme;
2. explicit transmission of the latent path under the frozen state code.

The implementation includes a reference finite-state forward calculation and an explicit-path calculation. Conformance tests verify that the marginalized negative log likelihood is no greater than the cost of any individual path.

Approximate marginalization may be added only in a later codec version with a separately validated error bound.

## 7. Cost-envelope decision rule

Define positive `Delta L` to mean that the cipher model is longer than the production model.

Report:

```text
Delta H-full
Delta H-conditional
Delta I-full
Delta I-conditional
```

Then:

- all four positive: `ROBUST_PRODUCTION_ADVANTAGE`;
- all four negative: `ROBUST_CIPHER_ADVANTAGE`;
- mixed signs: `UNRESOLVED_TABLE_COST`.

Sampling uncertainty is assessed separately. Agreement of accounting conventions does not remove the need for held-out uncertainty intervals.

## 8. Synthetic selection-policy gate

The bundled calibration harness tests a known homophonic class cipher under four frozen selection policies:

- iid uniform;
- cyclic;
- frequency weighted;
- sticky with line resets.

Each trial uses a new key. The decoder:

1. clusters surface symbols into equal-sized homophone classes from incoming and outgoing transition profiles;
2. assigns the recovered classes to an external plaintext model;
3. selects between a true and decoy plaintext register on training data;
4. measures plaintext recovery and language-model gain on held-out data.

For every held-out policy, calibration thresholds are derived only from the other policies. Production controls use the same family structure and selection policies but a different latent transition register.

The gate is a synthetic demonstration that selection-policy variation need not automatically seal a class-level experiment. It is not evidence about Voynich and does not establish that the final stateful morpho-local class is identifiable.

## 9. Conformance suite

The release contains:

- two canonical fixtures;
- exact H payload lengths;
- exact serialized-envelope SHA-256 hashes;
- I-cost component vectors;
- 18 unit tests;
- a deterministic 300-model fuzz/roundtrip test;
- a selection-policy decoder smoke test.

Any conforming implementation must reproduce `CONFORMANCE_VECTORS.json` exactly.

## 10. Production registry

`registry_fixture.json` is deliberately synthetic. Its hashes are placeholders used only to test framing, bounded references and registry invalidation.

No Voynich manuscript comparison is admissible until a separate production registry has been generated from verified ledger artefacts and committed before the run. That registry must contain the exact hashes and candidate-pool sizes for:

- plaintext lexica and language models;
- context definitions;
- the house-conditioned gallows constraint;
- the verified function-word inventory;
- the frozen adjacent-length selector;
- allowed key-partition selectors.

## 11. Current limits

Version 0.1 intentionally does not support:

- continuous parameters;
- neural plaintext models embedded inside the key;
- approximate hidden-path inference;
- overlapping codebook classes;
- sparse-payload locus priors;
- arbitrary user-defined instructions;
- unconstrained post-encryption realisers.

Adding any of these requires a new version, a new conformance suite and a complete rerun.

## 12. Reproduction

```bash
python -m unittest -v test_codec.py
python run_conformance.py
python fuzz_codec.py
python synthetic_gate.py --quick
```

The full synthetic gate is:

```bash
python synthetic_gate.py --workers 32 --output SYNTHETIC_GATE_RESULT.json
```
