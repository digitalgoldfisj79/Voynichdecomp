# Tranchedino × STA v2.2 — Final T20 optimizer qualification

Date: 2026-08-09
Protocol freeze: `82ad190bac1e8a3bb46e45843de7554436a08e73`
Runner: `39a286931421d496d86e3fa46519e6be2cbe0ad3`

Six fresh independent 96-restart ensembles were run on T20 only:

| ensemble | HF job | final objective |
|---|---|---:|
| K | `6a78800d3e1f34a7e32c0685` | -2.4740162795606953 |
| L | `6a78805dda2af92a634f02bc` | **-2.472681680372687** |
| M | `6a788079da2af92a634f02be` | -2.4799529166942156 |
| N | `6a788097da2af92a634f02c4` | **-2.472681680372687** |
| O | `6a7880b03e1f34a7e32c068f` | **-2.472681680372687** |
| P | `6a7880c73e1f34a7e32c0693` | **-2.472681680372687** |

The highest objective is independently attained by four of six ensembles. L, N, O and P recover the exact identical map, so occurrence-weighted agreement among the top ensembles is 1.0.

Canonical frozen K36→19 map in RF top-36 vocabulary order:

`[4,0,4,18,16,17,8,17,15,3,10,8,18,10,16,11,15,9,9,3,11,12,13,7,13,2,12,1,14,6,6,5,14,7,5,1]`

RF vocabulary order:

`A1,A2,A3,K1,B2,B1,J1,Q1,C1,Q2,D1,K2,L1,G1,C2,Ba,F1,P1,B3,U2,U1,Aa,F3,M1,P2,Lb,Ab,T1,E1,H1,G3,B4,E2,Z1,Qa,F2`

T20 retained-symbol coverage is 0.9955456815107289.

## Verdict

**V2.2 T20 FIT QUALIFIED.**

The frozen v2.2 criterion required the top objective to be attained by at least two fresh ensembles within 1e-7 with >=0.90 occurrence-weighted map agreement. Four ensembles attain the same top objective and the same map exactly.

Under the frozen protocol the pristine H21 panel from the never-scored v2.0 C20 folios is now unlocked. C21 and IT/ZL/GC replication remain sealed until H21 passes.
