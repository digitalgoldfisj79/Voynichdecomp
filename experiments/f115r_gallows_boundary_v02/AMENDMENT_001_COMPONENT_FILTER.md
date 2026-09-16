# Amendment 001 — component-filter calibration

Date: 2026-09-16
Status: frozen after v0.2 development failure and before any valid target morphology extraction.

## Trigger

v0.2 produced zero exact gallows-count matches on its development region. Inspection of aggregate counts showed systematic over-detection of ordinary upper/body fragments. No target `k` structures were assigned, so no target morphology result exists.

## Prospective repair

The upper-structure detector may additionally calibrate two transparent fragment filters:

- minimum connected-component vertical extent as a fraction of the calibrated upper-band height;
- minimum original dark-ink pixel area.

No other morphology feature, target boundary, hand label, or target image statistic may be used for parameter choice.

## Split development region

To prevent count overfitting:

- calibration subset: physical lines 25–34;
- sealed validation subset: physical lines 35–45;
- target remains physical lines 7–24.

Choose `(upper_frac, join_px, min_height_frac, min_area)` only on calibration lines by:

1. highest exact gallows-count match rate on lines containing at least one transcript gallows;
2. lowest mean absolute count error;
3. highest zero-gallows correctness;
4. simpler filter tie-break: larger minimum-height fraction, larger minimum area, then smaller join width.

After parameter choice, open validation lines 35–45 once. The instrument qualifies only if at least 70% of validation lines containing transcript gallows have exact detected counts. Target lines 7–24 remain unopened if validation fails.

The morphology contrast and its thresholds remain exactly as in `PROTOCOL.md`.
