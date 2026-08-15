import json, os, requests
from huggingface_hub import HfApi

api = HfApi(token=os.environ['HF_TOKEN'])
rows = []
for j in list(api.list_jobs())[:20]:
    rows.append({
        'id': j.id,
        'created_at': str(j.created_at),
        'started_at': str(j.started_at),
        'finished_at': str(j.finished_at),
        'flavor': j.flavor,
        'status': j.status.stage,
        'command': ' '.join(j.command)[:500],
    })
body = {
    'secret': 'frontier-u6-stageb-20260815',
    'id': 'u6-stageb-20260815-pathfix-jobs',
    'payload': json.dumps({'jobs': rows}, sort_keys=True),
    'meta': {'phase': 'execution-status-probe'},
}
r = requests.post('https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814', json=body, timeout=60)
r.raise_for_status()
print(json.dumps(rows[:5], indent=2))
