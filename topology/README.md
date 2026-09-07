# Voynich production topology v0.1

Status: **FROZEN_DEV / ordering unlicensed** (2026-09-07)

This programme treats the physical bifolium as the candidate primitive production unit and asks whether the present codex order preserves original production topology.

## Non-negotiable inference order

1. hard physical evidence
2. material-production evidence
3. bifolium textual-state evidence
4. content continuity
5. present binding/foliation

Hard constraints override soft scores. No chain is forced. Direction is not inferred from undirected similarity.

## Prohibited leakage

Topology is frozen without consulting downstream entropy, hapax, R64/cache, Currier-transition, change-point or generative-harness results. A topology may never be selected because it improves one of those statistics.

## Qualification regions

Q13 and Q20 are opened first. Herbal ordering remains sealed.

The database source of truth is Supabase protocol `vms_prod_topology_20260907_v01` and tables `vms_topology_*_v01`.

## First exploratory graph

Q13 does **not** yield a representation-stable linear sequence. Two undirected candidate proximity edges are stable across the eight current channels (ZL/RF/IT/GC x word/character-trigram):

- f75/f84 <-> f78/f81: 8/8 channel-optimal paths
- f76/f83 <-> f77/f82: 7/8

Q20 has a more stable textual-proximity backbone, but it is not a reading-order result:

- f106/f113 <-> f107/f112: 8/8
- f103/f116 <-> f108/f111: 8/8
- f107/f112 <-> f108/f111: 7/8
- f104/f115 <-> f105/f114: 7/8
- f104/f115 <-> f106/f113: 6/8

Removing the anomalous f104/f115 unit (f115r contains a known Scribe-2 intervention) leaves the best undirected path family 103/116 - 108/111 - 107/112 - 106/113 - 105/114.

## Mandatory calibration result

A Tier-A known-sequence control was run on public-domain Caesar, *De Bello Gallico* I-IV, with unit lengths exactly matched to Voynich Q13/Q20 strict-token counts. Exact open-path enumeration used the same word-cosine + within-token character-trigram scoring.

- Q13-shaped windows: n=57, mean adjacency recall 0.5921, exact-path rate 0.0702, false-edge rate 0.4079.
- Q20-shaped windows: n=42, mean adjacency recall 0.6286, exact-path rate 0.0952, false-edge rate 0.3714.

**FAIL. Maximum textual similarity is not qualified to reconstruct reading order.**

Therefore all Voynich path outputs remain exploratory production-proximity summaries. A Tier-B known manuscript/bifolium control is still required before any original-order claim.

## Current licensed inference

The bifolium carries a reproducible local production state in several regions of MS 408. The present nested-quire arrangement is not licensed as the original production topology throughout the codex. The original linear reading order, where one existed, remains unresolved.
