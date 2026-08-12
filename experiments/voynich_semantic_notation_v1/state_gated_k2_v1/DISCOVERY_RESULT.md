# VSN-B3-v1 — State-Gated K2 Discovery Result

Date: 2026-08-12
Status: **DISCOVERY PASS — HOLDOUT UNLOCKED**
Binding scientific freeze: `924ef5edd25f884227410dbfad0b59998b33f62f`
Execution workflow commit: `32c498d05df89669ce6f9ef2a9febb2d1e28f161`
GitHub Actions run: `31575192384`, job `94045629787`
Raw discovery artifact ID: `9133240790`
Raw JSON SHA-256: `29d469868bc8897c079e0c70334a4bfcf46ccd0ca8e02bf3de855630a0ca02f2`
Artifact ZIP SHA-256 reported by Actions: `03d471a6d13e37188d643effd50a9d9107c8df926b482ed2048cc98bb4c10928`
Frozen runner SHA-256: `a534e38d0110c189fe1b008ad53bc844b481344e9af4c41a67aaa2a9fd1cb56c`
Frozen target-bundle SHA-256: `2feabfff8ec10ca53c9d3fb4dfdb55f0ce0070104860567a27df237a5ae558ff`
Latin lexicon SHA-256: `5a139a6e7a3b9bfe9ef0b0e98e5178fb1c42be66dc3034c3f6f5e3d91b099b9c`

## Winner

Frozen discovery winner:

```json
{"family":"LINE","id":"LINE-w21-l2","w1":2,"w2":1,"line_w":2}
```

Interpretation of the implementation only:
- section/state gate on K2 slot 1: width 2 of 4 synthetic categories;
- section/state gate on K2 slot 2: width 1 of 4 synthetic categories;
- persistent line state: width 2 of 4 synthetic line categories.

These category assignments are deterministic SHA-256 abstractions and are **not medieval semantic labels or Voynich decipherments**.

## Frozen discovery comparison

Winner mean loss: **3.8698518398131068**
BASE mean loss: **6.765504826200853**
Relative improvement: **0.42800250103639204 = 42.80%**
Discovery sections improved vs BASE: **4 of 4**
Holdout unlocked: **TRUE**

Per-section mean loss:

| section | BASE | LINE-w21-l2 | winner better? |
|---|---:|---:|---|
| Stars | 4.6276572082 | 2.9901668053 | yes |
| Herbal-A | 5.8172938863 | 2.9644431266 | yes |
| Balneological | 7.5776735724 | 6.4226614854 | yes |
| text-only | 9.0393946379 | 3.1021359420 | yes |

Winner median discovery loss: **3.0656603846**.
BASE median discovery loss: **6.6929916167**.

## Unlock decision

The preregistered holdout rule required all of:
1. winner is gated rather than BASE — PASS;
2. mean discovery loss improves by >=20% — observed 42.8%, PASS;
3. winner improves at least 3/4 discovery sections — observed 4/4, PASS.

Therefore the previously sealed primary holdout sections **Pharmaceutical** and **Herbal-B** are now legitimately open for the exact frozen winning configuration only.

## Next binding step

Run `LINE-w21-l2` unchanged on Pharmaceutical and Herbal-B with the frozen 20 holdout seeds `2026081301..2026081320`.

No configuration reselection, gate-width change, seed selection or metric change is permitted after holdout opening.

A held-out section median passes only if all six preregistered criteria pass:
- pair ratio `[0.80,1.25]`;
- edit-location TV `<=0.08`;
- line-enrichment difference `<=0.25`;
- `H(next|prev)` difference `<=0.35` bits;
- right-minus-left negative and difference `<=0.10` bits;
- mean type length difference `<=0.75` characters.

Overall structural-gating PASS requires both held-out section medians to pass all six.
