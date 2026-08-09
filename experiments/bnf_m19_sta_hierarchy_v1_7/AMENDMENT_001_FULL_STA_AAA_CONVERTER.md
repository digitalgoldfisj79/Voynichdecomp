# Amendment 001 — full STA to aaa converter

Date: 2026-08-09
Status: **prospective; no language score had been generated when this amendment was committed.**

The frozen protocol named `STAR-aaa.bit`, which is specifically the reduced-STA -> aaa rule set. RF1b is reduced STA, but IT2a, ZL3b and GC2a level 1 can retain full-STA member codes. For a common cross-transliteration conversion, v1.7 will therefore use Zandbergen's full-STA rule set for **all** four files:

- `https://www.voynich.nu/software/bitrans/STA-aaa.bit`
- SHA-256 `622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64`

The pinned bitrans source remains unchanged: SHA-256 `3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a`.

A data-only equivalence check before amendment showed that `STA-aaa.bit` and `STAR-aaa.bit` produce byte-identical converted output for RF1b (SHA-256 `c14f43c731f46274f35b604356c6bb96a1186e0836aa9aa2b518666cce854167`). Thus the amendment changes no RF representation; it only ensures that full-STA codes in IT/ZL/GC are converted by the intended official rules.

No hypothesis, language panel, split, vocabulary rule, M19 law, optimization gate, H17/C17 threshold or confirmation criterion is changed.
