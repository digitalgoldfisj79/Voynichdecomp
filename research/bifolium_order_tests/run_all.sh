#!/usr/bin/env bash
# Runs every test in order. Each stage caches its intermediate results in data/*.pkl, so a rerun skips finished work.
# Total time on a laptop: roughly 1-1.5 hours (most of it in s28). Add SLIM=1 to also run the 12-transliteration check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; S="$ROOT/scripts"; cd "$ROOT"
python3 "$S/s00_prepare.py" "$ROOT" ${SLIM:+--slim}
python3 "$S/s14_features.py" "$ROOT"            # page profiles of glyph transitions
python3 "$S/s15_t1b_t2_t5.py" "$ROOT"           # PERMANOVA sheet term; regression: sheet vs opening; NCD
python3 "$S/s25_per_sheet.py" "$ROOT"           # per-sheet table (Figure 1)
python3 "$S/s26_posted_rarefied.py" "$ROOT"     # equal-length rerun + shuffled-text control
if [ -n "${SLIM:-}" ]; then python3 "$S/s20_translits.py" "$ROOT" "$ROOT/data/voynich_transcriptions_slim.json"; fi
python3 "$S/s22_chain_calibration.py" "$ROOT" "$ROOT/data/controls"   # builds sheet units for VMS + known-order controls
python3 "$S/s24_rarefied.py" "$ROOT"            # split-leaf replication at equal size, with shuffled controls
python3 "$S/s27_gradient.py" "$ROOT"            # within-group gradient: reliability, identity, stability
python3 "$S/s28_order_methods.py" "$ROOT"       # ordering-method calibration on known-order texts (slow)
python3 "$S/s30_chain_mixture.py" "$ROOT"       # Echo chain: stages vs mixtures of earlier and later pages
python3 "$S/make_figures.py" "$ROOT"
