# Standalone reproduction
Python 3.10+; no third-party packages.
Run `python run_standalone.py` for the held-out 80-permutation orbit test (~1 minute). Run `python autonomous_equivalence_standalone.py` for exact two-corpus ciphertext reconstruction.
Neither script reads files outside this folder. `autonomous_equivalence_standalone.py` reads plaintext reference **only at the end**, in an explicitly separate verification loop: the transduction inputs are solely baseline ciphertext + frozen manifest, as the code shows. The baseline ciphertexts are themselves produced by v142b from German; independent original-language status of the base generator is not claimed.
`00_FROZEN_PROTOCOL.md` and `01_AUTONOMOUS_EQUIVALENCE_PROTOCOL.md` precede the respective test runs. `stage*.pkl` are atomic checkpoints. Check `INTEGRITY.json` for the delivered files' hashes.
