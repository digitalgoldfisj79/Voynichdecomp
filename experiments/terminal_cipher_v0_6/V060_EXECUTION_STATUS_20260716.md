# Voynich terminal cipher programme v0.6 — execution status

Snapshot date: 2026-07-16

Repository: `digitalgoldfisj79/Voynichdecomp`

Branch: `experiment/terminal-cipher-programme-v0.6-20260716`

Protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`

## Family P — pending final development execution

The original final-amendment monolithic job `6a588de285d9643ce16d5f4a` exceeded its 14,400-second timeout while still reported as running and emitted no trial rows. It was cancelled and recorded as an infrastructure/runtime failure.

Execution-only single-trial sharding was committed at `5a80c33ad23e331195157a4b5eb710375be300c9`. It imports the frozen final solver and preserves every scientific constant.

Cheap execution smoke job `6a58d03db1669a49bf0777e2` passed.

The first full benchmark shard, periodic replicate 0, is running as job `6a58d06885d9643ce16d638d` on `cpu-xl` with a 12-hour timeout. The remaining 15 shards are withheld until this trial emits a recoverable row and proves the runtime envelope.

No Family P test or Voynich data has been opened.

## Family S — neural S3 blocked on persistence preflight

The injected credential authenticates as `Digitalgoldfish79`, but direct upload to the existing dataset and pull-request upload both failed with HTTP 403. Dataset mounts are read-only.

Persistence-preflight report commit: `8e7a6160ec94fb44e01eed9119ae925fa71d3a2b`.

The A100×4 rerun was not launched. Neural S3 remains scientifically pending, not failed. The next legal action is the same CPU-only existing-repository upload probe using a write-capable credential.

No neural evaluation, Family S test or Voynich data has been opened.

## Family T — closed at development

Combined result report commit: `4bc183dc4311287c33ece25e7ba41b620c545953`.

- mean recovery: 65.5273%;
- median: 98.6979%;
- minimum: 19.7917%;
- at least 80%: 9/16;
- mode accuracy: 16/16;
- width accuracy: 13/16.

The frozen gate failed. Family T is closed with no locked test and no Voynich application.

## Family G — closed at development

G1 oracle recovery passed. Both initial and sole-amended G2 blind searches failed.

Final closure report commit: `d6e8cca1f4ecc50ea18ebe26792df12d492bc36e`.

Final amended G2:

- job: `6a58d7b985d9643ce16d6405`;
- result SHA-256: `c5d4bd663f19a14101fa0d2c3f31b6cb91c0411488cb125371cc8939087c8bc5`;
- AUROC: 0.785522;
- false-positive rate: 0.78125%;
- detected payloads: 38/64;
- carrier-class accuracy among detected: 94.7368%;
- exact parameter accuracy among detected: 92.1053%;
- mean recovery with abstention as zero: 57.6497%;
- at least 70% recovery: 38/64.

The frozen gate failed. Family G is closed with no locked test and no Voynich application.

## Blind model selection

Not yet legal to execute. Family P remains pending and neural S3 remains blocked before evaluation. No synthetic locked test, Voynich scoring or Davis-hand labels have been opened.
