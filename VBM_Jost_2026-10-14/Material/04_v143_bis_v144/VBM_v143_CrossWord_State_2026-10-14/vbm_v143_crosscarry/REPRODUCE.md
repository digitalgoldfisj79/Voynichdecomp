# Reproduce v143 from frozen v142b

Requires Python 3.10+, standard library only. Unzip the standalone archive and run from its top-level `vbm_v143_crosscarry` directory:

```sh
python encode_v143.py > encoder_stdout.json
python decode_v143_independent.py 01_v143_frozen_manifest.json hartlieb.carry.vbm carry hartlieb.carry.decoded.txt
cmp hartlieb.carry.decoded.txt vbm_v142_inband/hartlieb.reference.txt
python decode_v143_independent.py 01_v143_frozen_manifest.json train.carry.vbm carry train.carry.decoded.txt
cmp train.carry.decoded.txt vbm_v142_inband/train.reference.txt
python verify_v143.py > verify_stdout.json
python surface_bound.py > surface_stdout.json
```

`verify_v143.py` also verifies exact independent recovery for both sources and all three arms, adverse table checks, and the synthetic paired-source Hartlieb source-aware conditional null. JSON reports are `02_results.json` and `03_surface_bound.json`. Each stage is atomically pickled (`stage0_...` through `stage5_...`). The independently authored decoder does **not** import encoder functions or use plaintext to decode.

Data provided are normalised Bavarian and Hartlieb source texts, original v142b emitted ciphertexts, v142b's original frozen manifest, and the original author's B REVA reference text used solely for the inherited necessary glyph-seam screen. They are reused unmodified, not regenerated from v138. The input original author ZIP is not necessary for this standalone reproduction.

Limitations: v142b encodes spaces as source word boundaries and puts original pairs inside words; v143 does not test the original JoJo cross-EVA-space proposal. The frozen cell inventory was chosen with exposure to reference B; this stage contains no independently anchored VMS attribution. The paired synthetic effect is planted by construction.
