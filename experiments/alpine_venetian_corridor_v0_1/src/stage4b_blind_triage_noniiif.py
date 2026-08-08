#!/usr/bin/env python3
"""Stage 4b: deterministic blind triage of sealed non-IIIF/direct-image witnesses.

Frozen before any VMS similarity inspection. Source resolution may use institutional
or public-domain surrogates of the already sealed manuscripts, but cannot add a
manuscript to the cohort. Model sees image only, never geography/corridor status.
"""
from __future__ import annotations
import hashlib,json,re,requests,torch
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL='Qwen/Qwen2.5-VL-3B-Instruct'; SEED='20260808'
CLASSES=['plant','root','flower','zodiac','star_astronomy','bath_human','architecture_cartography','diagram_geometry','other_relevant']
MAP_KEYS={'external:andrea_bianco_atlas_1436','external:pizzigani_parm_1612_1367','registry:zeg_de_virga_world_map','external:fra_mauro_map_1450','external:pizzigano_ambrosiana_1373','external:pizzigano_bell_1424'}

def evenly(xs,n):
 if len(xs)<=n:return xs
 return [xs[max(0,min(len(xs)-1,round((j+1)*(len(xs)+1)/(n+1))-1))] for j in range(n)]
def opaque(k,i):return 'B'+hashlib.sha256(f'{k}|{i}|{SEED}'.encode()).hexdigest()[:15]

def commons_category(session,cat,n=10):
 r=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'categorymembers','gcmtitle':cat,'gcmtype':'file','gcmlimit':'50','prop':'imageinfo','iiprop':'url','iiurlwidth':'1800'},timeout=60);r.raise_for_status();d=r.json();a=[]
 for p in d.get('query',{}).get('pages',{}).values():
  ii=(p.get('imageinfo') or [{}])[0];u=ii.get('thumburl') or ii.get('url')
  if u:a.append({'label':p.get('title',''),'url':u})
 return evenly(sorted(a,key=lambda x:x['label']),n)
def commons_file(session,title,width=2200):
 r=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':title,'prop':'imageinfo','iiprop':'url','iiurlwidth':str(width)},timeout=60);r.raise_for_status();p=next(iter(r.json().get('query',{}).get('pages',{}).values()),{});ii=(p.get('imageinfo') or [{}])[0];u=ii.get('thumburl') or ii.get('url');return [{'label':title,'url':u}] if u else []
def resolve(session):
 out={}
 out['registry:cr_british_library_egerton_ms_2020_erbario_carrarese']=commons_category(session,'Category:Carrara Herbal (c.1400) - BL Egerton MS 2020',10)
 # Andrea Bianco: current Commons page exposes a Ptolemaic sheet; retain only actual embedded manuscript image.
 try:
  r=session.get('https://commons.wikimedia.org/w/api.php',params={'action':'parse','format':'json','page':"Atlante di Andrea Bianco dell' anno 1436",'prop':'images'},timeout=60);r.raise_for_status();names=r.json().get('parse',{}).get('images',[]);arr=[]
  for name in names:
   if not re.search(r'(?i)\.(jpe?g|png|tif?f)$',name):continue
   arr+=commons_file(session,'File:'+name,1800)
  out['external:andrea_bianco_atlas_1436']=arr[:10]
 except Exception:out['external:andrea_bianco_atlas_1436']=[]
 out['external:pizzigani_parm_1612_1367']=commons_file(session,'File:Pizigani 1367 Chart 10MB.jpg')
 out['registry:zeg_de_virga_world_map']=commons_file(session,'File:DeVirgaWorldMap.jpg')
 out['external:fra_mauro_map_1450']=commons_file(session,'File:Fra Mauro Map.jpg')
 # Casanatense 4182: prefer numbered tavole, avoiding covers/guards.
 try:
  r=session.get('https://casanatense.contentdm.oclc.org/digital/api/search/collection/miniature/searchterm/4182/field/all/mode/all/conn/and/maxRecords/100',timeout=60);r.raise_for_status();arr=[]
  for it in r.json().get('items',[]):
   title=it.get('title') or '';ptr=(it.get('itemLink') or '').rstrip('/').split('/')[-1]
   if ptr and re.search(r'(?i)\btav\.',title):arr.append({'label':title,'url':f'https://casanatense.contentdm.oclc.org/iiif/2/miniature:{ptr}/full/1800,/0/default.jpg'})
  out['external:casanatense_4182']=evenly(arr,10)
 except Exception:out['external:casanatense_4182']=[]
 # Ambrosiana Google Arts institutional exhibit: embedded CDN images only.
 try:
  r=session.get('https://artsandculture.google.com/story/hAUxVcTL66hQKw',timeout=60);r.raise_for_status();urls=[]
  for u in re.findall(r'https://lh3\.googleusercontent\.com/[^"\'<> ]+',r.text):
   u=u.replace('\\u003d','=').replace('\\u0026','&')
   if u not in urls:urls.append(u)
  out['external:pizzigano_ambrosiana_1373']=[{'label':f'image_{i+1}','url':u} for i,u in enumerate(urls[:2])]
 except Exception:out['external:pizzigano_ambrosiana_1373']=[]
 # 1424 chart: re-triage under the same map-family prompt because generic Stage 4 returned zero objects.
 try:
  r=session.get('https://cdm16022.contentdm.oclc.org/iiif/info/p16022coll251/8809/manifest.json',timeout=60);r.raise_for_status();m=r.json();c=m.get('sequences',[{}])[0].get('canvases',[{}])[0];res=((c.get('images') or [{}])[0].get('resource') or {});svc=res.get('service') or {};base=svc.get('@id') or svc.get('id');out['external:pizzigano_bell_1424']=[{'label':str(c.get('label') or 'chart'),'url':base+'/full/2200,/0/default.jpg'}] if base else []
 except Exception:out['external:pizzigano_bell_1424']=[]
 return out

def main():
 proc=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=1100*28*28);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto')
 s=requests.Session();s.headers.update({'User-Agent':'VoynichCorridorBlindTriage4b/0.1'});sources=resolve(s)
 print('STAGE4B_CONFIG='+json.dumps({'model':MODEL,'seed':SEED,'classes':CLASSES,'sources':{k:len(v) for k,v in sources.items()},'map_family_prompt':sorted(MAP_KEYS)}),flush=True)
 for key,imgs0 in sources.items():
  print('STAGE4B_SOURCE='+json.dumps({'candidate_key':key,'selected':len(imgs0)}),flush=True)
  for i,p in enumerate(imgs0,1):
   oid=opaque(key,i)
   try:
    rr=s.get(p['url'],timeout=90);rr.raise_for_status();im=Image.open(BytesIO(rr.content)).convert('RGB');
    if max(im.size)>1400:im.thumbnail((1400,1400))
    if key in MAP_KEYS:
     prompt='''Blind manuscript-map triage. You are not given identity, place, date or any Voynich reference. Locate at most FOUR tight, non-overlapping substantive regions belonging only to architecture_cartography or diagram_geometry. Prefer distinct towers/cities/enclosures/fortified structures/map-like architectural motifs and distinct radial/circular/wind-rose/network diagrams. Do NOT use the whole map if smaller coherent regions exist. Exclude text-only labels, borders, decorative compass flourishes without structure, and blank sea/land. Return JSON only: {"objects":[{"object_class":"architecture_cartography|diagram_geometry","bbox_1000":[x0,y0,x1,y1],"description":"neutral morphology only","geometry_features":{}}]}. Coordinates integer 0..1000, x0<x1,y0<y1.'''
    else:
     prompt='''Blind manuscript image triage. You are not given manuscript identity, geography, date, or any Voynich reference and must not infer them. Identify only clearly visible regions in these frozen classes: plant, root, flower, zodiac, star_astronomy, bath_human, architecture_cartography, diagram_geometry, other_relevant. Return at most FOUR objects. Use tight non-overlapping boxes. Exclude ordinary text, decorated initials, borders and generic ornament unless structurally part of a relevant diagram. Return JSON only: {"objects":[{"object_class":one_of_the_classes,"bbox_1000":[x0,y0,x1,y1],"description":"neutral morphology only","geometry_features":{}}]}. Coordinates integer 0..1000, x0<x1,y0<y1. If no relevant illustration is visible return {"objects":[]}.'''
    msgs=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}];txt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);vi,vv=process_vision_info(msgs);inp=proc(text=[txt],images=vi,videos=vv,padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():gen=model.generate(**inp,max_new_tokens=700,do_sample=False)
    trim=[o[len(x):] for x,o in zip(inp.input_ids,gen)];ans=proc.batch_decode(trim,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();ans=re.sub(r'^```(?:json)?\s*','',ans);ans=re.sub(r'\s*```$','',ans);data=json.loads(ans);objs=data if isinstance(data,list) else data.get('objects',[]);valid=[]
    if isinstance(objs,dict):objs=[objs]
    for o in objs if isinstance(objs,list) else []:
     if not isinstance(o,dict):continue
     cls=o.get('object_class');b=o.get('bbox_1000',[])
     if cls not in CLASSES or len(b)!=4:continue
     try:x0,y0,x1,y1=map(int,b)
     except:continue
     if not(0<=x0<x1<=1000 and 0<=y0<y1<=1000):continue
     if (x1-x0)*(y1-y0)<400:continue
     gf=o.get('geometry_features') if isinstance(o.get('geometry_features'),dict) else {}
     valid.append({'object_class':cls,'bbox_1000':[x0,y0,x1,y1],'description':str(o.get('description',''))[:500],'geometry_features':gf})
    print('STAGE4B_PAGE='+json.dumps({'candidate_key':key,'seq':i,'folio':p['label'],'opaque_id':oid,'image_url':p['url'],'objects':valid},ensure_ascii=False),flush=True)
   except Exception as e:print('STAGE4B_PAGE_ERROR='+json.dumps({'candidate_key':key,'seq':i,'opaque_id':oid,'error':type(e).__name__+': '+str(e)[:220]}),flush=True)
if __name__=='__main__':main()
