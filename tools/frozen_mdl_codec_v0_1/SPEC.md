# Frozen MDL codec v0.1

Status: implementation candidate. No Voynich comparison may use this codec until the conformance and adversarial synthetic suites pass and the repository commit is frozen.

## Purpose

This package removes discretionary table charging from cipher-versus-production comparisons. It supplies two reports from the same explicit model record:

1. **H-full:** the bit length of a canonical prefix-free serialization containing every key entry, surface spelling, state instruction, external-model choice and exception represented in the record.
2. **I:** an information-theoretic component report using enumerative structure codes and Jeffreys/Krichevsky–Trofimov categorical mixture codes.

Neither report is chosen after observing the result. Future comparisons must publish both.

## Canonical value language

Only the following values are admitted:

- null;
- booleans;
- signed integers;
- UTF-8 strings;
- ordered lists;
- dictionaries whose keys are UTF-8 strings.

Floating-point model parameters are prohibited from the canonical key record. Counts, integer multiplicities or externally computed marginal log-probabilities must be used instead. A marginal log-probability is a reported data cost and is not treated as historical key material.

Every value starts with a three-bit type tag:

| Tag | Value |
|---:|---|
| 000 | null |
| 001 | false |
| 010 | true |
| 011 | signed integer |
| 100 | UTF-8 string |
| 101 | list |
| 110 | dictionary |
| 111 | reserved; decoder error |

Natural numbers are encoded as Elias delta of `n + 1`. Signed integers use zig-zag conversion followed by the natural-number code. Strings encode their byte length followed by literal UTF-8 bytes. Lists encode their length followed by each element. Dictionaries encode their size and then key/value pairs sorted by raw UTF-8 key bytes.

The resulting complete-object code is self-delimiting and canonical. Byte output is zero-padded only after the final bit; the separate exact bit length is mandatory metadata.

## H-full convention

`canonical_serialization_bits` is the exact number of bits in the complete canonical model record. The model author must include all operational material required to reproduce decoding or generation, including:

- codegroup spellings or indexes into a separately frozen inventory;
- plaintext units and unit types;
- codegroup-to-class assignments;
- nulls and structural markers;
- state names and transitions;
- reset rules;
- key partitions;
- selection instructions;
- exceptions;
- external plaintext-model choices;
- any frozen surface-production tables.

A missing field means that mechanism is unavailable. It cannot be inferred silently from the tested manuscript.

## Enumerative structure convention

### Codebook partition

For `C` non-empty, semantically ordered classes containing `V` canonically ordered surface items with sizes `h_1 ... h_C`:

`L = L_delta(C+1) + log2 binom(V-1, C-1) + log2(V! / product h_c!)`

The first term is the v0.1 natural-number code length of `C`. The second transmits the ordered composition; the third transmits membership.

### State topology

For `S` states and outdegree `d_s` from each state:

`L = L_delta(S+1) + sum_s [L_delta(d_s+1) + log2 binom(S, d_s)]`

The identities of allowed destinations are charged. Transition probabilities or multiplicities are charged separately.

## KT categorical code

For a categorical row with fixed alphabet size `K`, counts `n_1 ... n_K`, total `N`, and Jeffreys parameter `alpha = 1/2`:

`L_KT = -log2 [ Gamma(K alpha) / Gamma(N + K alpha) * product_i Gamma(n_i + alpha) / Gamma(alpha) ]`

Zero-count categories remain part of `K`. The caller must not remove unused categories after observing the data unless that reduced alphabet is separately transmitted as structure.

Rows of transition and emission tables are coded independently and summed.

## Latent paths

Latent structure is never free. Exactly one mode must be declared:

- `none`: no latent path;
- `explicit`: transmit states at reset points under a uniform state code and all other transitions under row-wise KT codes;
- `marginalized`: provide a finite exact or deterministic dynamic-programming marginal log2 probability and charge its negative.

A Viterbi path selected only because it minimizes test-data cost is inadmissible unless the path is transmitted explicitly.

## External plaintext models

A model selected from a preregistered set of `M` external plaintext models is charged `log2(M)` bits. Model adaptation, vocabulary extension and fitted grammars must appear explicitly in the canonical model record and be charged there. The codec does not permit post hoc declaration that a selected model was externally fixed.

## Required model fields

The reference `cost_model` function requires:

- `codec_version = "frozen-mdl-codec-v0.1"`;
- `class_sizes`: list of positive integers, or empty;
- `num_states`: non-negative integer;
- `outdegrees`: one integer per state;
- `transition_counts`: optional rectangular or ragged categorical rows;
- `emission_counts`: optional categorical rows;
- `latent_path_mode`: `none`, `explicit` or `marginalized`;
- latent-path fields required by that mode;
- `external_model_count` and `external_model_index`.

Additional fields are allowed and are charged by H-full even when they do not enter convention I. This is deliberate: convention I reports only its declared mathematical components, while H-full charges the entire operational record.

## Comparison register

For every future model comparison publish at least:

- H-full difference;
- I structural/universal difference;
- held-out predictive difference;
- sampling uncertainty for predictive terms.

A result whose sign changes between H-full and I is `UNRESOLVED_TABLE_COST`. The codec does not supply a retrospective tie-breaker.

## Freeze discipline

Any change to tags, ordering, integer coding, formulas, schema interpretation or conformance vectors creates a new version and requires complete reruns. Implementation bug fixes that alter any expected bit length also create a new version.
