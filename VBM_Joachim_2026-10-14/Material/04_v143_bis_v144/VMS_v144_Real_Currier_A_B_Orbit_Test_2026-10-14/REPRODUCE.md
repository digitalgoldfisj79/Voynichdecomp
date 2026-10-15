# Reproduce / bound

1. Download exactly the Zandbergen–Landini source [eva_zl3b.txt](https://github.com/Aspect-Research/voynich-autoexploration/blob/master/data/transcriptions/eva_zl3b.txt) (Git blob SHA `2a4533ab9bdfa85db9bad602d590978953055df1`). Save as `eva_zl3b.txt`. This original transcription is not in the attached audit ZIP.
2. The original real-data scores in `04_RESULTS.json` were computed through the connected GitHub tool and a V8 script, applying the frozen contract. `replicate_orbit_test.py` is a *separate Python reimplementation*, successfully syntax-checked but **not numerically executed on the pinned input in this offline runtime**; it uses Python RNG and will not produce byte-identical permutation nulls. Do not treat it as independent confirmation.
3. Run `python replicate_orbit_test.py eva_zl3b.txt --out verify.json` for ordinary EVA, `--leaf` for leaf holdouts, `--leaf --reverse` for the sensitivity representation, and `--planted` for the 15% calibration. JSON and atomic pickle files are written alongside output.
4. The Python file reproduces basic family-sign rotation results. The additional arbitrary edit-group permutation bound is recorded in the report/results, not implemented in this standalone program.

**No promotion:** the sensitivity test failed and calibration was underpowered for two sections. Exact mathematical equivalence to an autonomous surface transducer remains an identification barrier.
