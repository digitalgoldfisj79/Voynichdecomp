# f116v DINOv3–CATMuS hybrid inference

## Status

`F116V_HYBRID_MODEL_HYPOTHESES_ONLY`

Final f116v job: `6a71ed706b79c09949c22079`

- Hardware: one L4 GPU
- Running time: 179 seconds
- Training corpus: 512 CATMuS lines
- Development/test: 96/96 lines
- Shelfmarks: 13 train, 2 development, 2 test; mutually disjoint
- Held-out hybrid test CER: **0.5952**
- Exact held-out line accuracy: **0.0000**

The passing hybrid architecture was retrained deterministically, then applied to the same fixed line regions in laboratory true colour, expert monochrome multispectral PCA and expert colour PCA.

## Raw outputs

| Line | True colour | BW PCA | Colour PCA |
|---|---|---|---|
| 1 | `acaceamanal laancane` | `dt tu crete lilcl a cs` | `⁊ mmmmmmmmn m mmm` |
| 2 | `actur cehlltchegligt cort carct car` | `dacurhelilli ctetctii dot ccotc` | `f acncants n⁊ ctatsataut` |
| 3 | `aecroce ccoeocotcoctc cd` | `deecoceticeco coccttcec` | `d aoaamcaaamamn ma mucmmm` |
| 4 | `pce coraoacaopagp a` | `poe cet co coguoopcae` | `pocamntcmnmuoa camae` |

These strings are retained for audit only. They are not proposed readings.

## Cross-view ordered subsequences

| Line | True-colour/BW-PCA LCS | All-three-view LCS |
|---|---|---|
| 1 | `ce lca` | one space only |
| 2 | `acurhllli cot cctc` | `acct ct` |
| 3 | `ecocececococtcc` | `occ` |
| 4 | `pce cocogpa` | `pccoca` |

Longest-common-subsequence agreement is weak evidence: it preserves order but ignores substitutions, spatial offsets and the high base frequency of common CATMuS letters. None of the all-view strings is promoted to a glyph reading.

## Consequences

1. The hybrid construction itself is viable: DINOv3 materially improved held-out CATMuS recognition over the matched pixel-only control.
2. The current small model is not accurate enough for f116v transcription.
3. Its f116v predictions are strongly view-dependent and do not reproduce the earlier Kraken–CATMuS `…chicon…ladaba…` hypothesis.
4. This weakens, rather than corroborates, any claim that the earlier fluent CATMuS output is securely determined by the source strokes.
5. The model may still be useful as a feature extractor or as an initialization for a substantially larger, script-matched training programme.

## Checkpoint persistence

The job attempted to create `Digitalgoldfish79/f116v-dinov3-catmus-htr-v0.1` on Hugging Face. The available token could read gated DINOv3 weights but lacked repository-creation permission, so checkpoint upload failed with HTTP 403. The executable code, frozen architecture and numerical results are committed here; the binary checkpoint is not represented as persistently archived.

## Evidence discipline

- No dictionary or language model was used.
- No abbreviation expansion or word correction was used.
- No generated or restored pixels entered the recognizer.
- Agreement with Kraken–CATMuS would not be fully independent because both systems use CATMuS labels.
- The final outcome is a successful architecture proof of concept and an inconclusive f116v reading experiment.
