#!/bin/sh
set -eu
pip install -q numpy numba rapidfuzz gitpython
cd /tmp/v
printf 'V060_VALIDATION_GIT_HEAD %s\n' "$(git rev-parse HEAD)"
exec python experiments/terminal_cipher_v0_6/v060_family_p_locked_test_termination_fixed.py /tmp/v /tmp/p2-test-full-fixed.json
