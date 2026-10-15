# Reproducing the v142 in-word and pair-bridge experiment

Python 3.10+; no pip, no internet. Requires the author's uploaded original `Generator_v138_Pruefpaket_2026-10-14.zip` extracted with `gen138/` holding `vbm_tafel_v138.py`, `panel_v126.py`, `Paket_v100`, `klartext/` and `vms_B_REVA_atomstream.txt`.

Place this entire experiment folder anywhere. For rerunning encoder, set `VBM_SOURCE_DIR` to the extracted `gen138` and `VBM_SOURCE_ZIP` to the unchanged source archive. The author module creates an original-style output and prints a message as a side effect on import; this does not change the original ZIP.

```sh
export VBM_SOURCE_DIR=/path/to/gen138
export VBM_SOURCE_ZIP=/path/to/Generator_v138_Pruefpaket_2026-10-14.zip
python encode_v142.py
python decode_v142_independent.py 01_frozen_manifest.json train.vbm train.decoded.txt
python decode_v142_independent.py 01_frozen_manifest.json hartlieb.vbm hartlieb.decoded.txt
cmp train.reference.txt train.decoded.txt
cmp hartlieb.reference.txt hartlieb.decoded.txt
python verify_v142.py

# The second branch restores original vowel-bridge pairs.
python encode_pair_bridge.py
python decode_pair_independent.py 04_pair_bridge_manifest.json train.pair.vbm train.pair.decoded.txt
python decode_pair_independent.py 04_pair_bridge_manifest.json hartlieb.pair.vbm hartlieb.pair.decoded.txt
cmp train.reference.txt train.pair.decoded.txt
cmp hartlieb.reference.txt hartlieb.pair.decoded.txt
python create_compact_manifest.py
python decode_pair_compact.py 06_pair_compact_manifest.json train.pair.vbm train.pair.compact_decoded.txt
python decode_pair_compact.py 06_pair_compact_manifest.json hartlieb.pair.vbm hartlieb.pair.compact_decoded.txt
cmp train.reference.txt train.pair.compact_decoded.txt
cmp hartlieb.reference.txt hartlieb.pair.compact_decoded.txt
python verify_pair_surface.py
python bound_raw_glyphs.py
```

The `decode_*` scripts do not read the original archive, plaintext or encoder. `parent_v141_manifest.json` is included as a frozen starting inventory for `encode_pair_bridge.py`; no prior absolute experiment directory required. The data-derived manifests are frozen before the holdout encoding. The provided `.pkl` stages are atomically written; results are saved as JSON. No historical mechanism identification is attempted.
