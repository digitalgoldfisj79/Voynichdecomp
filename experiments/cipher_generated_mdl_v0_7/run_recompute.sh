#!/bin/sh
set -eu
pip install -q numpy scipy
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd /tmp/v
printf 'V070_RECOMPUTE_GIT_HEAD %s\n' "$(git rev-parse HEAD)"
python experiments/cipher_generated_mdl_v0_7/v070_oracle_entry.py --repo /tmp/v --workers 24 --output /tmp/v070_oracle.json
python - <<'PY'
import json
p=json.load(open('/tmp/v070_oracle.json'))
for r in p['results']:
    a=r['accounting']
    print('V070_ORACLE_ROW', json.dumps({
        'trial_type':r['trial_type'],
        'family':r['family'],
        'corpus':r.get('corpus'),
        'replicate':r['replicate'],
        'source_profile':r['selected']['source_profile'],
        'source_order':r['selected']['source_order'],
        'primary_leave_target_out':r['primary_leave_target_out'],
        'heldout_gain':a['heldout_gain_bits_per_token'],
        'total_gain':a['total_gain_bits_per_token'],
        'full_difference':a['full_difference_bits'],
        'conditional_difference':a['conditional_difference_bits'],
        'selected':r['source_message_selected'],
    }, sort_keys=True))
print('V070_ORACLE_RECOMPUTE_SUMMARY', json.dumps(p['summary'], sort_keys=True))
print('V070_ORACLE_RECOMPUTE_SHA256', p['sha256'])
PY
