#!/usr/bin/env python3
"""Stage 4: deterministic blind object triage for the sealed IIIF subset.

No VMS images or similarity are loaded. The VLM sees only an image and opaque ID.
The script prints JSONL; persistence is deliberately separate from inference.
"""
from __future__ import annotations
import hashlib,json,re,requests,torch
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL='Qwen/Qwen2.5-VL-3B-Instruct'
SEED='20260808'
MAX_PER_DEFAULT=10
SOURCES={
'external:bsb_cod_icon_242':'https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00013084/manifest',
'external:bodl_canon_misc_554':'https://iiif.bodleian.ox.ac.uk/iiif/manifest/6ae78449-a166-439c-9c98-ddfc7b6c8bf8.json',
'external:vat_lat_4082':'https://digi.vatlib.it/iiif/MSS_Vat.lat.4082/manifest.json',
'external:pizzigano_bell_1424':'https://cdm16022.contentdm.oclc.org/iiif/info/p16022coll251/8809/manifest.json',
'registry:merlon_beinecke_ms_327_venice':'https://collections.library.yale.edu/manifests/10269817',
'external:walsperger_pal_lat_1362b':'https://digi.vatlib.it/iiif/MSS_Pal.lat.1362.pt.B/manifest.json',
'registry:mn_munchen_bayerische_staatsbibliothek_clm_14622':'https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00109392/manifest',
'external:bsb_clm_14684':'https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00086348/manifest',
'registry:cr_munchen_bayerische_staatsbibliothek_cgm_38_konrad_von_':'https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00043227/manifest',
'external:bnf_nal_1673':'https://gallica.bnf.fr/iiif/ark:/12148/btv1b105380445/manifest.json',
'external:sloane_4016':'https://bl.digirati.io/iiif/ark:/81055/vdc_100165172997.0x000001',
'external:bnf_lat_7342':'https://gallica.bnf.fr/iiif/ark:/12148/btv1b9068035n/manifest.json',
}
CLASSES=['plant','root','flower','zodiac','star_astronomy','bath_human','architecture_cartography','diagram_geometry','other_relevant']

def parse_manifest(m):
 out=[]
 if m.get('sequences'):
  for i,c in enumerate(m['sequences'][0].get('canvases',[]),1):
   rr=((c.get('images') or [{}])[0].get('resource') or {});svc=rr.get('service') or {}
   if isinstance(svc,list):svc=svc[0] if svc else {}
   base=svc.get('@id') or svc.get('id');url=f'{base}/full/1400,/0/default.jpg' if base else rr.get('@id') or rr.get('id')
   out.append({'seq':i,'label':str(c.get('label') or ''),'url':url})
 else:
  for i,c in enumerate(m.get('items',[]),1):
   try:b=c['items'][0]['items'][0]['body']
   except Exception:b={}
   svc=b.get('service') or []
   if isinstance(svc,dict):svc=[svc]
   base=svc[0].get('id') if svc else None;url=f'{base}/full/1400,/0/default.jpg' if base else b.get('id')
   lab=c.get('label')
   if isinstance(lab,dict):lab=next(iter(lab.values()),[''])[0]
   out.append({'seq':i,'label':str(lab or ''),'url':url})
 return [x for x in out if x['url']]

def evenly(xs,n):
 if len(xs)<=n:return xs
 # deterministic endpoints excluded when possible
 idx=[]
 for j in range(n):
  k=round((j+1)*(len(xs)+1)/(n+1))-1;k=max(0,min(len(xs)-1,k));idx.append(k)
 return [xs[k] for k in sorted(set(idx))]

def select(key,pages):
 # remove obvious bindings/color targets
 usable=[p for p in pages if not re.search(r'(deckel|cover|spine|colorchecker|scala|edge|goutti|tranche|mirror|spiegel|pastedown)',p['label'],re.I)]
 if key=='external:bodl_canon_misc_554':
  z=[]
  for p in usable:
   m=re.search(r'(?<!\d)(15[4-9]|16\d|17[0-4])\s*[rv]?',p['label'])
   if m:z.append(p)
  return evenly(z if z else usable,10)
 if key=='external:pizzigano_bell_1424':return evenly(usable,2)
 if key=='external:walsperger_pal_lat_1362b':return evenly(usable,2)
 if key=='external:vat_lat_4082':return evenly(usable,16)
 if key in ('registry:mn_munchen_bayerische_staatsbibliothek_clm_14622','external:bsb_clm_14684','external:bnf_lat_7342'):return evenly(usable,12)
 return evenly(usable,MAX_PER_DEFAULT)

def opaque(key,seq):return 'P'+hashlib.sha256(f'{key}|{seq}|{SEED}'.encode()).hexdigest()[:15]
def parse_json_answer(s):
 s=s.strip();s=re.sub(r'^```(?:json)?\s*','',s);s=re.sub(r'\s*```$','',s)
 return json.loads(s)

def main():
 proc=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=1000*28*28)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto')
 session=requests.Session();session.headers.update({'User-Agent':'VoynichCorridorBlindTriage/0.1'})
 print('STAGE4_CONFIG='+json.dumps({'model':MODEL,'sources':len(SOURCES),'classes':CLASSES,'seed':SEED}),flush=True)
 for key,mu in SOURCES.items():
  try:m=session.get(mu,timeout=60).json(); pages=parse_manifest(m); chosen=select(key,pages)
  except Exception as e:
   print('STAGE4_SOURCE_ERROR='+json.dumps({'candidate_key':key,'error':type(e).__name__+': '+str(e)[:200]}),flush=True);continue
  print('STAGE4_SOURCE='+json.dumps({'candidate_key':key,'manifest_pages':len(pages),'selected':len(chosen)}),flush=True)
  for p in chosen:
   oid=opaque(key,p['seq'])
   try:
    rr=session.get(p['url'],timeout=60);rr.raise_for_status();im=Image.open(BytesIO(rr.content)).convert('RGB')
    if max(im.size)>1200:im.thumbnail((1200,1200))
    prompt='''Blind manuscript image triage. You are not given manuscript identity, geography, date, or any Voynich reference and must not infer them. Identify only clearly visible regions in these frozen classes: plant, root, flower, zodiac, star_astronomy, bath_human, architecture_cartography, diagram_geometry, other_relevant. Return at most FOUR objects. Use tight non-overlapping boxes. Exclude ordinary text, decorated initials, borders and generic ornament unless structurally part of a relevant diagram. Return JSON only: {"objects":[{"object_class":one_of_the_classes,"bbox_1000":[x0,y0,x1,y1],"description":"neutral morphology only","geometry_features":{}}]}. Every coordinate must be integer 0..1000, x0<x1,y0<y1. If no relevant illustration is visible return {"objects":[]}.'''
    msgs=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
    txt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);imgs,vids=process_vision_info(msgs)
    inp=proc(text=[txt],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():out=model.generate(**inp,max_new_tokens=700,do_sample=False)
    trim=[o[len(i):] for i,o in zip(inp.input_ids,out)];ans=proc.batch_decode(trim,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    try:data=parse_json_answer(ans)
    except Exception:data={'objects':[],'parse_error':ans[:500]}
    valid=[]
    for o in data.get('objects',[]):
     b=o.get('bbox_1000',[]);cls=o.get('object_class')
     if cls not in CLASSES or len(b)!=4:continue
     try:x0,y0,x1,y1=map(int,b)
     except Exception:continue
     if not(0<=x0<x1<=1000 and 0<=y0<y1<=1000):continue
     if (x1-x0)*(y1-y0)<400:continue
     valid.append({'object_class':cls,'bbox_1000':[x0,y0,x1,y1],'description':str(o.get('description',''))[:700],'geometry_features':o.get('geometry_features') or {}})
    print('STAGE4_PAGE='+json.dumps({'candidate_key':key,'seq':p['seq'],'folio':p['label'],'opaque_id':oid,'image_url':p['url'],'objects':valid},ensure_ascii=False),flush=True)
   except Exception as e:
    print('STAGE4_PAGE_ERROR='+json.dumps({'candidate_key':key,'seq':p['seq'],'opaque_id':oid,'error':type(e).__name__+': '+str(e)[:200]}),flush=True)
if __name__=='__main__':main()
