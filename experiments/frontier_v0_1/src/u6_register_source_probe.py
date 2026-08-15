import ast, json, os, requests
from huggingface_hub import HfApi, hf_hub_download

REPO='Digitalgoldfish79/voynich-dinov3-pipeline'
TOKEN=os.environ['HF_TOKEN']
api=HfApi(token=TOKEN)
info=api.repo_info(REPO,repo_type='dataset')
p=hf_hub_download(REPO,'vdino3/register.py',repo_type='dataset',revision=info.sha,token=TOKEN)
src=open(p,encoding='utf-8').read(); tree=ast.parse(src)
funcs={}
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in {'register_folio','candidate_service_ids'}:
        funcs[n.name]=ast.get_source_segment(src,n)
out={'pipeline_revision':info.sha,'functions':funcs,'target_opened':False,'true_retention_read':False}
r=requests.post('https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814',json={'secret':'frontier-u6-stageb-20260815','id':'u6-stageb-20260815-register-source','payload':json.dumps(out,sort_keys=True),'meta':{'phase':'source-only'}},timeout=120); r.raise_for_status()
print(json.dumps(out,indent=2))
