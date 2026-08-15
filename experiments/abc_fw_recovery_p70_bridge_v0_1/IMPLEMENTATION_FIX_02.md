# Implementation fix 02 — 2026-08-15

Run 3 completed the full permutation analysis but failed before result serialization while building the optional P70 per-carrier profile. At least one frozen August-13 FW carrier spelling has no occurrence in `enriched_records.json`; `Counter(...).most_common(1)[0]` therefore raised `IndexError`.

The frozen carrier list is **not changed**. Zero-occurrence carriers are retained and reported with frequency 0 / no P70 parse. This is a reporting-frame mismatch, not grounds to silently rename or remove a carrier.

No target statistic had been serialized or inspected before this fix. No seed, null, statistic, threshold, corpus, or hypothesis is changed.