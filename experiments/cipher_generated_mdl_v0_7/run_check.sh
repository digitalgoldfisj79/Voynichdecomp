#!/bin/sh
set -eu
pip install -q numpy scipy
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd /tmp/v
printf 'V070_GIT_HEAD %s\n' "$(git rev-parse HEAD)"
exec python experiments/cipher_generated_mdl_v0_7/v070_oracle_source_transfer.py --repo /tmp/v --workers 24 --output /tmp/v070_oracle.json
