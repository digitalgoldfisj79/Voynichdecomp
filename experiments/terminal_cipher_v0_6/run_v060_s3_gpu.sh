#!/bin/sh
set -eu
pip install -q numba rapidfuzz sentencepiece requests
cd /tmp/v/experiments/terminal_cipher_v0_6
printf 'V060_S3_EVAL_GIT_HEAD %s\n' "$(git -C /tmp/v rev-parse HEAD)"
exec torchrun --standalone --nproc_per_node=1 v060_family_s_neural_postprocess.py three /tmp/v
