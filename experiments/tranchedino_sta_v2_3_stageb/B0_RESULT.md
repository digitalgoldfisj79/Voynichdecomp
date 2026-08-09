# Tranchedino × STA v2.3 — Stage B0 result

Date: 2026-08-09
Protocol: `PROTOCOL_B0.md`
Status: **PASS — B1 CALIBRATION DESIGN AUTHORISED; NO VOYNICH TARGET FIT AUTHORISED**

## B0.1 — full-STA representation geometry

Primary RF1b source reproduced the frozen SHA-256:
`81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`.

The binding Stage-A parser reproduced 157,254 full-STA characters and 166 observed member types. The strict f.69v mixed key requires K=92 visible signs.

| stream | full-STA chars | observed types | top-92 coverage | tail events | gate |
|---|---:|---:|---:|---:|---|
| RF1b | 157,254 | 166 | **0.999046004** | 150 | PASS |
| ZL3b | 156,460 | 216 | **0.999054071** | 148 | PASS |
| GC2a level 1 | 157,096 | 190 | **0.998605948** | 219 | PASS |

Independent stream hashes used:

- ZL3b: `8438ba1c45f47fe1d06b5262cbcdf60ce69158a0edbd4dd802612896f3217e2a`;
- GC2a level 1: `0c0d1eea4b5ab87f8a65fb7f4346864cd90758ad993812b4f2122b3899d4ac88`.

All three exceed the frozen >=0.995 K92 coverage gate. The 92-sign historical geometry is therefore representable at essentially complete occurrence coverage in full STA and is not an RF-only inventory artefact.

## B0.2 — genuine Paduan word source recovered

Library artefact recovered:
`tranchedino_paduan_payload_program_complete.zip`.

Archive SHA-256:
`ddae949a2d4ff13714204f3751feaf9e836333ef57a45def77c803cd87fc7b61`.

Recovered binding source files:

- `paduan_cipher_letters.txt`: 227,702 bytes/letters; SHA-256 `9d21818c13a425639a68ae2c6fb400f35d3f81a49a77bb1f9d610162012f39fe`;
- `paduan_lines.csv`: 5,735 rows; SHA-256 `c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6`;
- `tranchedino_homophone_cells.csv`: SHA-256 `d0b96b4c5311e7d1f620a0e63742fc949ec3c7b382f0430478f53f782db97053`.

The recovered `paduan_lines.csv` preserves the raw text, word boundaries, line boundaries, page identity and the old chronological split. Under the same frozen 19-letter normalisation used in v2.0 it reproduces exactly:

- chronological cut page: **183**;
- LM partition: **4,119** text-bearing lines / **172,347** retained characters;
- held-out partition: **1,423** text-bearing lines / **54,750** retained characters.

Word inventory under that same normalisation:

- LM/train: 39,011 word tokens / 11,248 types;
- held-out: 12,113 word tokens / 5,067 types.

B0.2 therefore passes. No replacement corpus is needed.

## B0.3 — source-only nomenclator observation feasibility

Before any mixed-unit solver result, fresh 38-word codebooks were sampled uniformly without replacement from the top-N Paduan training words. Twelve deterministic 12,000-letter held-out controls were censused for code identities/occurrences only.

| candidate pool | median distinct codes observed | minimum distinct | median code occurrences | minimum occurrences |
|---|---:|---:|---:|---:|
| top 64 | 34.0 / 38 | 27 | 463.0 | 279 |
| **top 96** | **31.0 / 38** | **26** | **309.0** | **171** |
| top 128 | 29.0 / 38 | 22 | 237.5 | 133 |
| top 192 | 25.5 / 38 | 21 | 182.0 | 131 |

The prospective B1 regime is therefore frozen at **38 fresh nomenclator words sampled from the top 96 training words, with 12,000-letter controls**. This directly addresses the old v0.5.4 identifiability failure, where approximately 384-character chunks exposed only 2–7 code symbols.

### Binding geminate occurrence census

The eleven f.69v geminate classes are genuinely heterogeneous in Paduan frequency. Held-out occurrences are:

`bb=2, cc=34, dd=19, ff=25, gg=3, ll=459, nn=49, pp=18, rr=40, ss=280, tt=69`.

Consequently B1 recovery gates must be occurrence-weighted and may not demand recovery of unobserved/near-unobserved key entries such as `bb` or `gg` in every control.

## Verdict

**STAGE B0 PASS.**

The historical K92 key geometry is viable under full STA, the original genuine Paduan word-level source has been recovered byte-for-byte, and a prospective nomenclator observation regime has been selected using source-only data.

This authorises B1 instrument development/calibration only. No Voynich mixed-unit mapping, language score or decoded text may be generated until a separately frozen B1 instrument passes its positive-control and recognition gates.
