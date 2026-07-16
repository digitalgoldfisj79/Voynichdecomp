#!/bin/sh
set -eu
pip install -q numpy scipy
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd /tmp/v
printf 'V070_GIT_HEAD %s\n' "$(git rev-parse HEAD)"
exec python experiments/cipher_generated_mdl_v0_7/v070_entry.py --repo /tmp/v --split engineering --workers 2 --output /tmp/v070_engineering.json
