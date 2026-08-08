# G0D — identifiability audit

Date: 2026-08-08

This audit uses no VMS similarity metric.

## Question

Can provenance class be crossed with digitisation source strongly enough that a downstream hand comparison is not merely a source classifier?

## Existing Stage-5 object panel

The sealed Stage-5 panel is not perfectly source-nested:

- BSB contributes both a corridor-core manuscript (Cod.icon.242, Venice) and Bavarian/Swabian controls (Clm 14684; Clm 14622).
- DigiVat contributes both a corridor-core manuscript (Vat.lat.4082, Padua) and a German control (Pal.lat.1362 B, Constance).

Therefore source × geography crossing is possible in principle. This does not validate the Stage-5 object panel as a handwriting panel.

## Prospective hand panel: Padua/Veneto vs German/Bavarian

Two independent acquisition platforms can be crossed:

### BSB / MDZ

- Paduan hand: BSB Clm 184, Hermann Schedel, copied at Padua 1440–1441 (paper's existing documented witness; digital object bsb00120557).
- German controls: BSB holdings already include multiple dated Bavarian/Regensburg manuscripts in the frozen corridor registry, including Clm 14622 and Clm 14684.

### DigiVat

- Paduan hand: Vat.lat.4082, portions copied at Padua by Petrus de Fita in 1401–1402.
- German controls: DigiVat Pal.lat.1145 (`Antidotarium Nicolai`, 15th c., Northwest Germany) and Pal.lat.1369 (astronomical-astrological miscellany, mid-15th c., South Germany) are fully digitised on the same platform and expose IIIF manifests.

This is enough to design a balanced two-source × two-class calibration block prospectively. Sampling must cap each source at <=60% of either provenance class and keep manuscripts, not crops, as the independent unit.

**Verdict: REPAIRABLE / NOT YET ASSEMBLED.** The original near-confounding is not mathematically unavoidable.

## Prospective hand panel: Padua/Veneto vs Lombardy/Pavia

A second crossing is also plausible but requires stricter metadata verification before assembly:

- British Library currently exposes the digitised Egerton MS 2020 (Carrara Herbal) on its 2024+ digitised-manuscripts platform, while the frozen panel already contains BL Sloane MS 4016 as a Lombard control.
- DigiVat supplies the Paduan Vat.lat.4082 and also contains fifteenth-century Lombard material (for example Vat.lat.1145 is catalogued as Bergamo, mid-15th c.), but an appropriate script/content/date-matched Lombard hand still needs selection.

The Carrara witness is not automatically an admissible handwriting comparator merely because its codex is Paduan-associated; hand, date, script type, and usable folios must be verified independently.

**Verdict: REPAIRABLE, conditional on hand-level verification and balanced assembly.**

## VMS target

Beinecke MS 408 remains a single target from a single acquisition platform. That is not itself fatal if:

1. all representation selection and nuisance calibration are completed externally;
2. provenance classes are source-crossed in the comparator corpus;
3. Beinecke is treated strictly as an out-of-domain target and never used to tune preprocessing, representation, classifier, thresholds or source correction.

## G0D overall

**REPAIRABLE / NOT A PASS.**

The design is not intrinsically unidentifiable, but no provenance inference is licensed until the actual source-crossed hand panel is assembled and G0A, G0B and fixed-glyph G0C have passed.
