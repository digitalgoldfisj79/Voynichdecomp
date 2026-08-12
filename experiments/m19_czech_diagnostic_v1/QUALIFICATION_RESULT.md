# M19 Czech Diagnostic v1.1 — Qualification Result

Date: 2026-08-12
Binding protocol: `PROTOCOL_V1_1.md`
Scientific runner freeze: commit `28c2d16f08ccbfb0d0279003eace5431b5401619`.
Runner SHA-256 on all three Actions workers: `c26e284cab6fd16ac2e8c971df64211c13fe14d4e935a5e023470d311a650e77`.
GitHub Actions run: `31575371275`.

## Result

Czech **PASSES all three preregistered positive-control scales**. H19 STA-family target access is therefore unlocked.

| K / representation scale | Czech rank | top margin | mapping accuracy | paired agreement | converged | best - true score | gate |
|---|---:|---:|---:|---:|---|---:|---|
| 22 / STA family | 1 | 0.15208724068843837 | 1.0 | 1.0 | yes | 0.0 | PASS |
| 26 / connected AAA | 1 | 0.14778224667027873 | 1.0 | 1.0 | yes | 0.0 | PASS |
| 36 / full STA | 1 | 0.15218500123251433 | 1.0 | 1.0 | yes | 0.0 | PASS |

Every scale converged after the first batch of 6 restarts per ensemble. Both ensembles attained the true-map training score exactly and agreed 100%.

## K=22 held-out Czech-control ranking

1. Czech -2.8650781775453042
2. Italian -3.0171654182337426
3. French -3.0249656965959897
4. German -3.0643084809583776
5. Latin -3.0730384765708854
6. Spanish -3.0757822003583537
7. Greek -3.123342624293146
8. Hebrew -3.505053358943385
9. Arabic -3.5952110379907074

## K=26 held-out Czech-control ranking

1. Czech -2.8522975430003714
2. Italian -3.00007978967065
3. French -3.0055025014158425
4. German -3.047171465298376
5. Latin -3.049586921739279
6. Spanish -3.0549788773483013
7. Greek -3.1086115811538644
8. Hebrew -3.4768296627673942
9. Arabic -3.571091742400691

## K=36 held-out Czech-control ranking

1. Czech -2.8638372824915903
2. French -3.0160222837241046
3. Italian -3.0176097057765583
4. German -3.060885342731518
5. Spanish -3.0657037297270717
6. Latin -3.070247322251976
7. Greek -3.119895815040932
8. Hebrew -3.50829203401393
9. Arabic -3.5954695717304235

## Czech source evidence

UD_Czech-CAC pinned commit: `798f89716ae5a96e86042df7d394d56787e2e213`.
Control dev SHA-256: `06ab3b41d0192641a063048c58f27ecc10640b61c0bae3112dd778ea1f4201f7`.
Control test SHA-256: `71c03ef1ab14451e294bbc0d7896ea4f8a3faf26790fc2f04cc473e62e43162c`.
Control pool after frozen normalization: 1,231 sentences; 109,380 letters.
Czech LM training subset: 9,390 sentences; 931,658 normalized letters.

## Raw-output hashes / artifacts

- K22 JSON SHA-256 `28b630488875ac3fcf65668eaa31d9b5de810616ed002b1c1db72c90a5e296db`; artifact ID `9132969330`.
- K26 JSON SHA-256 `867854057487f9021f7f2428b9544b10381da4cc7a5d6e6b60061faa70daeea2`; artifact ID `9132971208`.
- K36 JSON SHA-256 `73985c8039b7cb6f60bbd8b3c2d215951636168fd8904427661c17bbd1ca7ba3`; artifact ID `9132973973`.

## Next permitted action

Run **only** RF H19 STA-family K=22 Czech target fit and insert the resulting Czech score into the immutable v1.9 eight-language H19 ranking. If the resulting nine-language top margin remains below 0.05, stop; do not launch AAA/full-STA H19 and keep C19 sealed.
