# VBM BG conditional v5 — Q0 topology result

Date: 2026-08-11
Namespace: `VBMBGCONDV5`
HF job: `6a7b886c27caad61c6eac5f4`

## Binding result

**Q0 PASS.**

Discovery validation:
- Bavarian windows: 48; recall 1.0000.
- German windows: 41; recall 1.0000.
- balanced accuracy: **1.0000**.

Independent transfer, with the classifier frozen before transfer:
- non-Wikipedia MaiBaam Bavarian windows: 13; recall **0.9230769231** (12/13).
- German PUD windows: 48; recall **1.0000** (48/48).
- balanced accuracy: **0.9615384615**.

Frozen pre-target strength threshold:

`TAU_BG = 1.6272712366587183`

This is `max(0, p10(logit))` over correctly classified Bavarian transfer windows, exactly as preregistered.

## Corpus scale

- Bavarian discovery train: 1,801,255 normalized characters.
- Bavarian discovery controls: 1,306,758.
- German GSD train: 1,301,450.
- German GSD controls: 140,089.
- non-Wikipedia MaiBaam transfer: 17,449 characters / 556 sentences.
- German PUD transfer: 104,121 characters / 1,000 sentences.

## Nonbinding MaiBaam dialect diagnostic

Using 600-event windows:
- central: 3/3 Bavarian direction; median logit 1.9128.
- south: 1/2 Bavarian direction; median logit -1.7122.
- unknown: 6/6 Bavarian direction; median 5.7198.
- unknown central/southcentral/south: 11/13 Bavarian direction; median 3.5052.

The explicit south subset is only two windows and is therefore too small for a binding inference. Q0 supports broad Bavarian-vs-German transfer, not a specific South-Bavarian/Tyrolean claim.

## Target integrity

No Voynich H1 or C1 data were read or scored by Q0. H1 and C1 remain untouched by v5 at this point.
