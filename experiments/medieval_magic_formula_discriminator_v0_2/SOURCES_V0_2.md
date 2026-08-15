# Frozen source manifest v0.2

## Lecouteux
Derived corpus is the v0.2 extraction used by the v0.1 preflight: 827 entries. It contains short formula strings and metadata only, not the dictionary prose. The seven gzip+base64 parts reconstruct the CSV. No entry has yet been primary-source-checked, so the dictionary corpus is treated as a secondary-source discovery/statistical corpus; any load-bearing historical example must be checked separately.

## Ordinary medieval A controls
The fixed seven-source excerpt panel was frozen before the v0.1 scores and is reused without change:
- Buch von guter Speise (~1350), German recipe prose
- Engelthal Büchlein (pre-1350), German religious prose
- Alexiuslegende witness (~1422), German religious prose
- Erhard Knab, gout regimen (1469), German medical prose
- Albrecht von Eyb, Marina (1472), German non-medical prose
- Regimen sanitatis Salernitanum, Latin medical/dietetic prose
- Thomas Aquinas, Summa theologiae, Latin religious/scholastic prose

Source work is the A split unit. These are frozen excerpts, not newly selected after the v0.1 result.

## Voynich RF / STA
Voynich is not acquired until `external_freeze.json` has been written by the v0.2 runner.
- RF1b reduced STA1: https://voynich.nu/data/sta/RF1b.txt ; SHA-256 81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17
- bitrans.c: https://www.voynich.nu/software/bitrans/bitrans.c ; SHA-256 3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a
- STA-aaa.bit: https://www.voynich.nu/software/bitrans/STA-aaa.bit ; SHA-256 622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64

The RF/STA/aaa parsing follows the already-used Zandbergen hierarchy conventions: long-word boundary (`.` certain, `,` uncertain and retained inside the long word), unknown Z units excluded, official STA-to-aaa conversion, colon-connected aaa pairs treated as one unit.
