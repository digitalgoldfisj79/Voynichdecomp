# Demolition packet: paste into a FRESH chat. Instruction to the reviewer: "Find what is wrong with this finding."

## Claim
In Beinecke MS 408, ink darkness measured per sheet (bifolium) from the 2014 Yale photographs is more similar between neighbouring sheets in
eo777's text-derived "Path 1" sheet order than chance, and is not more similar between neighbouring sheets in the present binding order.

## Numbers claimed
- Statistic: mean absolute difference in sheet darkness between neighbours along an order (48 sheets). Null: shuffle sheets within section x hand, 3,000 times.
- Path 1: observed 0.0964, null mean 0.1337, null SD 0.0099, z 3.79. Binding order: observed 0.1417, null mean 0.1408, null SD 0.0109, z -0.08.
- After removing parchment brightness: Path 1 z 3.49. After adjusting for focus and stroke width: z 2.39. Leave-one-quire-out: z 2.76 to 3.52.

## Data (no reasoning supplied on purpose)
Repo digitalgoldfisj79/Voynichdecomp, branch eo777-ink-session-2026-11-14, folder experiments/eo777_ink_session_2026-11-14:
- inkdirect/words_y2014.csv.gz   word-level measurements (darkness, width_px, sigma_hat, B, page, word)
- inkdirect/phase2d_sheets.csv, phase6_sheets.csv   sheet-level darkness
- inkdirect/instrument.py, analysis6.py, db1_bounding.py, phase6.py   code
- spatial/common.py   how the three orders and the strata are built
- eo777 voynich-clusters golf-v21-assignments.csv (bifolium_path_rank) = Path 1; vms-spatial physical_map.csv = section, hand, quire

## What the reviewer is asked to do
Re-derive the result from the word-level file, then try to break it: measurement artefacts, confounds, the choice of null, the choice of strata,
circularity between a text-derived order and an image-derived measure, multiple testing across the session, anything else.
