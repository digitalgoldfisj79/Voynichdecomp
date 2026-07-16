#!/bin/sh
set -eu
pip install -q numpy numba rapidfuzz scikit-learn
cd /tmp/v
printf 'V060_VALIDATION_GIT_HEAD %s\n' "$(git rev-parse HEAD)"
exec python experiments/terminal_cipher_v0_6/v060_blind_model_selection.py /tmp/v060_blind_model_selection_result.json
