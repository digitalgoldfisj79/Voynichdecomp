# Grammar vs graphotactics — executed audit, 18 October 2026

## RETRACTED / DOWNGRADED FINDINGS (TOP OF FILE)

- **R9:** Earlier suggestion that the predecessor-initial effect evidences grammatical productivity is withdrawn. In crossed folio×stem holdouts, the real-vs-surface-only graphotactic emission contrast is unresolved: Q13 +0.00312 bits/event above emulator (null SD 0.00421; 0.74 SD), Stars +0.00419 (SD 0.00352; 1.19 SD). Emulator uses authentic word skeletons and is not a free-running Voynich generator.
- **R10:** Negative long-range result does NOT exclude syntax. The lag-2..5 detector failed its OR4 planted-positive power qualification, especially in Q13 (detections across 8 planted controls: 0/8, 0/8, 0/8, 1/8; required ≥7/8).
- **R11:** Four-language fingerprint correctly classified 28/28 real windows **and** 28/28 first-order word-Markov surrogates as their source languages. This is an unqualified orthography/lexicon/register detector, not independent grammatical identity evidence. No Voynich source ranking was run.
- **R12:** Gold-labelled Latin/Hebrew/Czech agreement establishes gold sanity, NOT blind grammar recovery; Yiddish Number features absent. Previous project retractions remain in force.

## Results overview

| Stage | Measured finding | Qualification |
|---|---|---|
| 0: gold grammar controls | Latin 330/330 number agreement; Hebrew 517/519; Czech 1881/1889; Yiddish 5 separable particle edges but no comparable Number features | Gold-label sanity passes only; source identification not qualified |
| 1: productive onset selection | Q13 conditional graphotactic excess 0.74 SD; Stars 1.19 SD; TTLI 0.30 SD | No grammar-specific excess |
| 2: nonlocal distance2–5 | No real lag gains ≥2 crossed-folio×stem SD in Balneological/Stars/Herbal A/B | Detector underpowered in OR4 planted positive |
| 3: Hebrew/Czech/Yiddish/Latin fingerprint | 28/28 real windows correctly classified; also 28/28 first-order Markov surrogates | Fails hard nonlinguistic negative |

All five **exact, untruncated raw VoynichStats line-level exports** are in `data/`: `q13_zlzi.json`, `q13_ttli.json`, `stars_zlzi.json`, `herbalb_zlzi.json`, `herbala_zlzi.json`. This is a new isolated branch; `main` remains unchanged.

Source corpus Git blob SHAs:
- Latin CIRCSE: `43516eba40c13e58fe9d1e5e2f8d8ce901c62232`
- Hebrew HTB: `6caf67d0ff84031511a99b497b925dee29528974`
- Czech CLTT: `93be043e9cd406f0240e50080d0e1e56c8648640`
- Yiddish YiTB: `ad6ef1dca3cab10c4f3705c3f4d319c06d5c87dd`

The full detailed report and atomic pickle checkpoints are provided in the 2026-10-18 chat audit archive. Be explicit: the original ephemeral connector-side code and RNG streams have not been archived byte-for-byte. A separate independently written Python verification script is supplied, but its numerical outputs are not claimed as byte-exact reproductions.

**Final conclusion:** the existing CONTACT/CONNECT contextual pattern remains real as a surface phenomenon. No candidate language is identified or excluded, and nonlocal grammar cannot be ruled out.
