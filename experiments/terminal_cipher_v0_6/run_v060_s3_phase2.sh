#!/bin/sh
set -eu
pip install -q numpy numba rapidfuzz sentencepiece requests
cd /tmp/v/experiments/terminal_cipher_v0_6
printf 'V060_S3_EVAL_GIT_HEAD %s\n' "$(git -C /tmp/v rev-parse HEAD)"
exec python v060_family_s_neural_postprocess.py two /tmp/v
