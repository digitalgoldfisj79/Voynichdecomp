# Super-Grammar XD1 public audit report — 2026-09-28

**Status:** completed analytical closeout; SGT12 promoted into the *candidate bounded* kernel; independent replication still open.

This report is deliberately narrow. It is not a decipherment, language identification, cipher identification, semantic reading, or historical production claim.

## Corrections first

1. Frozen v03 certifier line-order bug: 646/3,893 ZLZI LINE_BREAK pairs (16.6%) were physically non-adjacent. v03 remains frozen; v03.1+ corrects it.
2. SPACE > LINE_BREAK attenuation is not VMS-specific. Nuremberg reproduces it strongly. Earlier mechanistic interpretation from attenuation alone is withdrawn.
3. Lag-2 exact-word enrichment alone is not VMS-specific. A 12-document ReM panel reproduces aggregate lag-2 enrichment. The surviving result is narrower.
4. RF1b/STA is a representation sensitivity, not an independent palaeographic transcription.
5. No charm recurrence metric has yet been executed. An earlier internal claim that a charm lag-2 result already existed was audited and remains retracted. However, the stronger statement that no corpus route was known is corrected: the archive contains Koen Gheuens’ 2025 charm-source thread and f116v expert follow-up, while Cambridge’s Curious Cures project and Heather A. Taylor’s independent 48-manuscript / ≥250-item charm survey provide an externally defined corpus route. Physical-line preservation must be verified before these sources can enter SGT12.

## SGT12 — within-line exact-token recurrence geometry

Statistic: exact token equality at lag 1 and lag 2 within a physical line.

Null: permute the *same exact token multiset within the same physical line*. Token counts, vocabulary, line length and line membership are therefore fixed.

### ZLZI, 2,000 permutations

- lag 1: **the metric does not resolve this** — effect +0.000226, null SD 0.000479, +0.47 SD, observed/null 1.024.
- lag 2: effect +0.002141, null SD 0.000515, +4.16 SD, observed/null 1.224.

### Frozen primary culinary-recipe controls

Controls were selected from the pre-existing SG89 CoReMA manifest by source recoverability, physical-line preservation and size before the repetition outcomes were computed.

Frozen falsifier: kill the proposed discriminator if any primary witness has lag1 observed/null >=0.90 **and** lag2 >=1.10.

| control | lag1 obs/null | lag2 obs/null | VMS-control lag2 difference / pooled bootstrap SD |
|---|---:|---:|---:|
| A1 | 0.054 | 0.647 | 5.86 |
| Bs1 | 0.081 | 0.640 | 7.30 |
| So1 | 0.111 | 0.739 | 5.03 |
| W1 | 0.110 | 0.463 | 11.72 |

No primary witness fires the falsifier. All four source extractions pass the frozen source-count QC. The opposition survives the formal 6–10-token line stratum for all four and the 11–20 stratum wherever the control has >=50 lines.

### ReM bound

12 documents; 28,841 lines; 116,220 tokens.

- lag 1: -0.003627 / null SD 0.000173 = -20.93 SD; observed/null 0.172.
- lag 2: +0.000576 / 0.000241 = +2.38 SD; observed/null 1.119.

Therefore **lag-2 enrichment alone is not diagnostic**. What remains unusual in the tested controls is no detectable lag-1 avoidance combined with lag-2 enrichment.

### Nuremberg Letterbooks 2–5

- lag 1: -0.007499 / 0.000144 = -51.9 SD; observed/null 0.093.
- lag 2: -0.002388 / 0.000285 = -8.39 SD; observed/null 0.650.

Nuremberg also reproduces the SPACE>LINE_BREAK attenuation, which kills that feature as a VMS-specific discriminator.

## Non-EVA representation

RF1b rendered in STA1, no mapping back to EVA.

Long-word convention:
- lag 1: +0.000257 / 0.000417 = +0.62 SD — unresolved.
- lag 2: +0.001253 / 0.000459 = +2.73 SD.

Frozen representation gate: PASS.

The RF source supports a second word-boundary convention because of 761 uncertain-space markers. A post-outcome short-word test was registered as downgrade-only:

- lag 1: +0.000257 / 0.000418 = +0.62 SD — unresolved.
- lag 2: +0.001453 / 0.000467 = +3.11 SD.
- downgrade triggered: false.

This post-outcome sensitivity does not strengthen SGT12; it only removes one obvious decision-rule fragility.

## Frozen promotion criteria

1. no primary recipe falsifier — PASS
2. VMS lag2 resolves — PASS
3. all primary VMS-control lag2 bootstrap differences resolve — PASS
4. formal length-stratum opposition — PASS
5. non-EVA representation does not reverse — PASS

SGT12 is therefore promoted into the **candidate bounded** kernel. It is not independently replicated.

## Important control-fairness limits

CoReMA controls are culinary recipe manuscripts, not charms:
- W1: 1350–1450
- Bs1: 1460
- A1: 1475–1500
- So1: 1487–1500

They are register + physical-line controls, not a perfectly date-matched historical cohort. ReM supplies a separate recipe/medical sensitivity. The Gaskell–Bowern human-gibberish arms remain small (~4–5k tokens per arm) and underpowered for manuscript-scale recurrence. The charm arm remains open.

## Main reproducibility pointers

Protocols:
- research/PROTOCOL_XD1_closeout_20260928.md
- research/PROTOCOL_XD1_STA_RF_20260928.md
- research/PROTOCOL_XD1_STA_SHORT_SENS_20260928.md

Key runs:
- recipe closeout + corrected XD1 completion: Actions 36458666044
- Nuremberg P3/P4: 36456117906
- ReM extension: 36458266568
- STA/RF long-word: 36462073007
- STA/RF short-word downgrade test: 36462857378

Hashes:
- VMS source SHA256: 26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f
- recipe metrics SHA256: dcffd4225cfcaf856418a0b118c1607511b1308de901053a733a7c79310deb3d
- STA source SHA256: 81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17
- STA long result checkpoint: b907afefacad8be6a880601b8751bd3f84fdf33a24c00da45016fc94950e8891
- STA short result SHA256: ff48fac886c916e1f884b3ff0a8869360176d9d44c8cd58048b21b9c4f9b4a63

## Licensed conclusion

Within the tested physical-line controls, Voynich running text shows a robust exact-repetition geometry in which immediate identical-token repetition is not detectably suppressed while identical-token recurrence at distance two is enriched. The tested chancery and recipe controls strongly suppress immediate identical-token repetition; some recipe text can nevertheless show lag-2 enrichment, so lag 2 alone is not diagnostic. The pattern survives a materially different STA representation.

That is the claim. Nothing semantic is being inferred from it.

## Charm follow-up corpus route

The charm arm is now a recoverable external follow-up, not an undefined literature gap. Prior art recovered from the Voynich Archive includes Voynich Ninja thread 4885 (“Charms etc. with crosses”, Koen Gheuens, 2025), the subsequent f116v/Katherine Hindley discussion (thread 5162), Cambridge’s Curious Cures digitisation/transcription project, Aine Widdicombe’s manuscript-level charm survey, and Heather A. Taylor’s independent survey of 48 English-provenance medical manuscripts with at least 250 non-medical charms/experimenta. A German-language sensitivity manifest also exists in Ninja thread 6047, but because it was assembled explicitly for Voynich comparison it should not define the primary control cohort.

The primary admissibility gate is technical: SGT12 needs diplomatic physical lines. Before any recurrence outcome is opened, the selected Curious Cures/Taylor witnesses must demonstrate explicit original line segmentation in TEI/PAGE/Transkribus export and independently defined charm boundaries. If available, the strongest design is within-manuscript charm-versus-recipe comparison, with bootstrap clustering by charm/manuscript. This is a post-closeout follow-up and cannot retroactively strengthen the five original promotion gates; it can downgrade SGT12 if it reproduces the VMS joint geometry.

## 2026-09-28 XD1-CHARM follow-up result

The previously open charm-control follow-up has now been executed on Cambridge UL MS Add. 9308 under a frozen downgrade-only protocol. Fourteen independently source-defined charm units (215 physical lines; 1,050 lag-2 opportunities) were read from CUDL diplomatic line geometry.

Primary lag1 observed/null = 0.1591, effect/null SD = -5.13. Primary lag2 observed/null = 1.0521, effect/null SD = +0.29: **the metric does not resolve this**.

A stricter ten-charm subset gives lag1 0.1204 / -4.61 SD and lag2 1.3154 / +1.44 SD: again **the metric does not resolve lag2**. All fourteen leave-one-charm-out populations fail the frozen downgrade gate.

SGT12 therefore remains CANDIDATE_BOUNDED_PROMOTED; this result does not constitute independent replication or broaden the claim beyond tested controls.

The German sensitivity (Amberg Ms. 77, from Ninja thread 6047) remains BLOCKED_PHYSICAL_LINEATION because the available route supplies images/OCR rather than an independently supplied diplomatic physical-line transcription. See research/XD1_CHARM_ADJUDICATION_20260928.md.