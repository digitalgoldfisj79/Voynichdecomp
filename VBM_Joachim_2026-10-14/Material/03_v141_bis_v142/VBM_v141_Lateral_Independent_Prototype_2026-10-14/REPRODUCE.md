# Reproduce VBM v141 lateral experiment

Requires Python 3.10+ and the user's original `Generator_v138_Pruefpaket_2026-10-14.zip` (not modified in this work). Run in an ordinary Python environment without network access. The supplied author generator may create its own `GENERAT_v138_kurzertext.txt` in the extracted working folder; **the ZIP remains unchanged**.

1. Extract author's ZIP into a working folder so `gen138/` contains `vbm_tafel_v138.py`, `belegung.json`, the VMS reference, and `klartext/`.
2. Set paths for the original package and local extracted folder, then run the encoder:

```bash
export VBM_SOURCE_ZIP=/path/to/Generator_v138_Pruefpaket_2026-10-14.zip
export VBM_SOURCE_DIR=/path/to/extracted/gen138
export VBM_OUTPUT_DIR=/path/to/vbm_lateral_results
mkdir -p "$VBM_OUTPUT_DIR"
python build_encoder.py
```

3. Copy these scripts into `$VBM_OUTPUT_DIR`, or run by their original path. Independently decode both messages:

```bash
python independent_decoder.py "$VBM_OUTPUT_DIR/01_frozen_manifest.json" "$VBM_OUTPUT_DIR/train.vbm" "$VBM_OUTPUT_DIR/train.decoded.txt"
python independent_decoder.py "$VBM_OUTPUT_DIR/01_frozen_manifest.json" "$VBM_OUTPUT_DIR/hartlieb.vbm" "$VBM_OUTPUT_DIR/hartlieb.decoded.txt"
cmp "$VBM_OUTPUT_DIR/train.decoded.txt" "$VBM_OUTPUT_DIR/train.reference.txt"
cmp "$VBM_OUTPUT_DIR/hartlieb.decoded.txt" "$VBM_OUTPUT_DIR/hartlieb.reference.txt"
```

4. Full independent controls and surface diagnostics (these scripts assume they are located beside the outputs):

```bash
export VBM_VMS_REFERENCE="$VBM_SOURCE_DIR/vms_B_REVA_atomstream.txt"
cd "$VBM_OUTPUT_DIR"
python verify.py
python extra_checks.py
```

The decoder needs **neither the original ZIP nor any plaintext file**. It uses only the ciphertext plus the public frozen JSON table. The verifier must read reference plaintext to compute exact-match scores, never for the decoder's input.

The JSON numbers and `.pkl` files are recomputed; every stage file is written via a temporary file, fsync and atomic replacement. Build in a fresh directory to avoid modifying the packaged experiment's existing results.

**What this test cannot prove:** visual fidelity, manuscript-compatible tokenisation, unknown-key decoding, historical VBM mechanism identification, or fidelity to unnormalised German spellings.
