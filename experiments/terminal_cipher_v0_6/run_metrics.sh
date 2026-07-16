#!/bin/sh
set -eu
pip install -q numpy numba rapidfuzz
printf 'V060_METRICS_GIT_HEAD %s\n' "$(git -C /tmp/v rev-parse HEAD)"
exec python /tmp/v/experiments/terminal_cipher_v0_6/v060_family_p_runtime_audit.py /tmp/v
