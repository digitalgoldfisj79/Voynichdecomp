# Shmuel-bukh 1544 anonymous whole-word source transfer v0.2 — CONFIRMATION RESULT

Status: **`SHMUEL1544_ANON_WORD_SOURCE_TRANSFER_V02_FAIL`**.

This is a failure of the frozen **visual source-transfer prerequisite**, not a rejection of Yiddish and not a Voynich result. Voynich remained sealed throughout.

## Provenance / freeze

Protocol freeze: `f683611685f310447110ee2d9415885f4e751012`.
Audit-rubric freeze: `7c4416a9a798cb24cc999d573311e9f765897089`.
Development pass recorded before confirmation access: `9f00c209dc5aaea0e9c30d19758eb16ab53bedfa`.
Executable detector: `5f593d881a981e32bebd210f0f392843d28cdc29`.

Fresh witness: Augsburg 1544 `Shmuel-bukh`, BSB/MDZ `bsb10170884`.

Machine census GitHub Actions run `34706037631`; artifact `shmuel1544-anon-v02-census`, ID `10300634880`, digest `sha256:262ea8844bc3f2e3819fa05c84d295e0e9ca87fb8a4803222d48298db0b19d8f`.

## Manifest / quantity

Manifest SHA-256: `31de0faaf00540f6b4187e25c2113d5b8d20ca54c1a20d6acaa46e6d0706633f`.

- manifest canvases: 202;
- frozen eligible 15%–85% interval: one-based canvases 31–171 inclusive (141 canvases);
- automatically text-bearing: 137/141;
- retained words across automatically text-bearing eligible pages: 22,002;
- registered long cell: first exactly 4,128 retained words, canvases 31–59, taking only the first 70 retained words of canvas 59.

Therefore the quantity gate PASS.

## Prospectively hash-selected visual-audit sample

`AUDIT_SAMPLE.json` was written by the machine census before any confirmation page image was visually opened. The six selected canvases were:

1. 32 — digest `01fbe9a381603bfd544326f4546edab2d0aa24f67906b3fdef8935d2fe005038`
2. 105 — digest `03b1f6d987ace71e1cea9de61bd7389d2bce26080e76f0b7d1497723d5d66135`
3. 57 — digest `054af26ba1397d6210203fb7bef04ecf6e2f8ad1f4042a69c5e6cddde5ee6f8e`
4. 47 — digest `05f5e8f8afa36ef167c3a4ac7ceddf987d7dcc9092d9805529c217df882f5a73`
5. 108 — digest `0770097900128930ea66f268921aec4162f4b683ca4e8aff8633f13cf479ff2f`
6. 126 — digest `0edeaf811ebf887b0057be5996bea0a1b29e72e1f7718fbf4601997875ba4407`

## Visual audit

The separately frozen rubric distinguishes running body text from spatially separated headings, stanza/section labels, signatures, catchwords and closing formulae.

| Canvas | Retained words | Retained lines | Missing substantive running body | Decision |
|---:|---:|---:|---|---|
| 32 | 185 | 25 | none observed | PASS |
| 105 | 162 | 26 | **one full top running-body line omitted** | **FAIL** |
| 57 | 153 | 27 | none observed | PASS |
| 47 | 133 | 29 | none observed | PASS |
| 108 | 182 | 27 | none observed | PASS |
| 126 | 143 | 26 | **two full top running-body lines omitted** | **FAIL** |

The omissions on 105 and 126 occur because their first running lines begin above the frozen `0.03*H` admissibility boundary. They are ordinary verse/body lines immediately continuous with the retained text below and cannot be classified as paratext under `AUDIT_RUBRIC.md`.

The protocol requires all six visual-audit pages to pass with zero missing substantive running-body lines. Therefore the confirmation fails at this prerequisite.

## Stopped stages

Because the visual-audit prerequisite failed, the following registered downstream stages were **not run** and must not be used to qualify v0.2:

- 60 predicted-equal pair audit;
- 60 hard-negative pair audit;
- 48 scan-nuisance cells;
- confirmation-cell nondegeneracy scoring.

No `tau`, descriptor, segmentation constant, page selection or audit interpretation was changed after inspecting the Shmuel pages.

## Interpretation

v0.2 successfully removed the Bovo/Basel absolute-page-position defect on DEVELOPMENT but the residual 3% relative top exclusion is still not source-portable. The Shmuel 1544 witness is now **consumed for v0.2 confirmation**. It may only be used as DEVELOPMENT in a future separately frozen version. The result says nothing against historical Yiddish as an M0 plaintext hypothesis; it says this image extractor is not yet sufficiently source-portable.