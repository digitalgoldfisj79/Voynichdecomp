#!/usr/bin/env python3
from __future__ import annotations
import os,re,io,json,time,hashlib,gc,statistics
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,requests,torch
from PIL import Image,ImageOps,ImageFilter
from transformers import AutoImageProcessor,AutoModel

RUN_ID=os.environ.get('RUN_ID','Q9Q10-SCALE-CMP-v0.2-20260810')
SUPABASE_URL=os.environ['SUPABASE_URL'].rstrip('/'); SUPABASE_KEY=os.environ['SUPABASE_KEY']
DINO_MODEL='facebook/dinov3-vits16-pretrain-lvd1689m'; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; DTYPE=torch.float16 if DEVICE=='cuda' else torch.float32
DATE_MIN=1250; DATE_MAX=1500; TOPK=12
CLASSES=['astro_diagram','sun_moon','sun','moon','star','sphere_heavens','computus_table','astrology_diagram','zodiac_wheel','zodiac_aries','zodiac_taurus','zodiac_pisces']
TARGETS={
'f67r1':('https://collections.library.yale.edu/iiif/2/1006194/0,0,2560,3738/900,/0/default.jpg','faced Moon with lateral crescents inside a twelve-point coloured star; twelve paired radial writing-and-star units; three circular text rings; inward radial organization; no human figures'),
'f67r2':('https://collections.library.yale.edu/iiif/2/1006194/2412,0,2560,3738/900,/0/default.jpg','central eight-point form with cloud-like surround; twelve outer compartments each containing a small Moon face and text; unique red writing and ruled red line; compartmental/circular organization'),
'f67v1':('https://collections.library.yale.edu/iiif/2/1006195/2449,0,2610,3753/900,/0/default.jpg','faced Sun with wavy rays; star field divided by seventeen outward radial text lines; one to three stars per text-defined area; twelve perimeter labels'),
'f67v2':('https://collections.library.yale.edu/iiif/2/1006195/0,0,2610,3753/900,/0/default.jpg','green square or aperture around a small spiral-rayed centre; four axial plus four diagonal line systems; four corner roundels containing linked faces; branching termini; mixed inward/outward text'),
'f68r1':('https://collections.library.yale.edu/iiif/2/1006196/0,0,2735,3828/900,/0/default.jpg','small faced Sun above faced Moon with twenty-nine scattered individually labelled stars in an open field; no segmented wheel'),
'f68r2':('https://collections.library.yale.edu/iiif/2/1006196/2575,0,2800,3828/900,/0/default.jpg','faced Moon above faced Sun; large outer ring of unlabelled stars enclosing mostly labelled stars; two circular texts; open luminary/star field'),
'f68r3':('https://collections.library.yale.edu/iiif/2/1006196/5215,0,2778,3828/900,/0/default.jpg','faced Moon at centre of eight-spoke rota; alternating dense small-star and sparse larger-labelled-star sectors; separate seven-star group connected by curved line'),
'f68v1':('https://collections.library.yale.edu/iiif/2/1006197/5330,0,2805,3843/900,/0/default.jpg','faced Sun inside star-shaped field with sixteen outer sectors alternating eight text-bearing and eight star-bearing sectors; two outer circular texts'),
'f68v2':('https://collections.library.yale.edu/iiif/2/1006197/2670,0,2820,3843/900,/0/default.jpg','blue eight-point form with yellow hub; eight alternating star/text sectors; four isolated larger stars alternating with denser small-star fields; outward text'),
'f68v3':('https://collections.library.yale.edu/iiif/2/1006197/0,0,2830,3843/900,/0/default.jpg','T-O-like divided central disk; blue annulus with yellow stars; wavy or nebuly boundary; eight visibly spiralling text bands between centre and outer ring; concentric cosmography'),
'f69r':('https://collections.library.yale.edu/iiif/2/1006198/0,0,2793,3763/900,/0/default.jpg','uneven six-point star centre; twenty-two outward text rays; eight blue axial rays and four green diagonal rays; outer ring of sixteen text units with blue/green ovals; multiaxial rota'),
'f69v':('https://collections.library.yale.edu/iiif/2/1006199/0,0,3020,3876/900,/0/default.jpg','white star on blue centre; twenty-eight pipe-like radial bodies ending in oval openings; twenty-eight inward radial words; three circular text arcs sharing a common-looking gap'),
'f70r1':('https://collections.library.yale.edu/iiif/2/1006199/2860,0,3070,3876/900,/0/default.jpg','white six-point star in small circle; blue-speckled field; fifteen inward radial texts; nine convex stamp-like outer forms plus six inner texts; four circular texts'),
'f70r2':('https://collections.library.yale.edu/iiif/2/1006199/5770,0,3116,3876/900,/0/default.jpg','faced Sun inside blue eight-point form; eight uninscribed rays; five circular text rings; offset prose paragraph; no radial text loci'),
'f70v1':('https://collections.library.yale.edu/iiif/2/1006201/0,0,2979,3700/900,/0/default.jpg','Aries sheep/goat eating from plant at centre; fifteen human figures in tubs arranged in two rings, each holding or attached to a star; two circular texts'),
'f70v2':('https://collections.library.yale.edu/iiif/2/1006200/0,0,3945,3772/900,/0/default.jpg','two Pisces fish with nearby stars and connecting lines at centre; two concentric rings with twenty-nine human/tub/star units; thirty labels total; three circular texts')}

def normfolio(s):
 s=(s or '').lower().replace('folio','').replace('fol.','').replace('fol','').replace('f.','').replace('recto','r').replace('verso','v'); s=re.sub(r'\s+','',s); s=s.lstrip('0') or s; return re.sub(r'[^0-9a-z]','',s)
def nlabel(x):
 if isinstance(x,dict):
  vals=x.get('en') or x.get('none') or next(iter(x.values()),[]); return ' '.join(map(str,vals)) if isinstance(vals,list) else str(vals)
 return str(x or '')
def folio_tokens(s):
 if not s:return []
 out=[]
 for b in re.split(r'[,;]',s):
  b=b.strip(); out.extend([x.strip() for x in b.split('-',1)] if '-' in b and re.search(r'\d',b) else [b])
 seen=[]
 for x in out:
  n=normfolio(x)
  if n and n not in seen:seen.append(n)
 return seen[:4]
def canvas_url(c):
 try:
  if 'images' in c:
   body=c['images'][0]['resource']; svc=body.get('service'); svc=svc[0] if isinstance(svc,list) and svc else svc; sid=(svc or {}).get('@id') or (svc or {}).get('id'); return sid.rstrip('/')+'/full/900,/0/default.jpg' if sid else body.get('@id') or body.get('id')
  body=c['items'][0]['items'][0]['body']; svc=body.get('service'); svc=svc[0] if isinstance(svc,list) and svc else svc; sid=(svc or {}).get('id') or (svc or {}).get('@id'); return sid.rstrip('/')+'/full/900,/0/default.jpg' if sid else body.get('id') or body.get('@id')
 except:return None

def query_rows():
 base=f'{SUPABASE_URL}/rest/v1/comparanda_illuminations'; h={'apikey':SUPABASE_KEY,'Authorization':f'Bearer {SUPABASE_KEY}'}
 sel='id,vms_class_v2,title,cote,work,folio,datation,date_start,date_end,holding_region,subject,iconclass_label,manifest,record_url,thumb,thumb_usable'; rows=[]
 for off in (0,1000):
  p={'select':sel,'date_start':f'gte.{DATE_MIN}','date_end':f'lte.{DATE_MAX}','vms_class_v2':'in.('+','.join(CLASSES)+')','order':'id.asc','limit':1000,'offset':off}
  r=requests.get(base,headers=h,params=p,timeout=20); r.raise_for_status(); z=r.json(); rows+=z
  if len(z)<1000:break
 return rows

def fetch_img(u,timeout=9):
 r=requests.get(u,headers={'User-Agent':'q9q10-scale-comparanda/0.2'},timeout=timeout); r.raise_for_status();
 if len(r.content)<1500:raise ValueError('small')
 im=Image.open(io.BytesIO(r.content)).convert('RGB')
 if min(im.size)<80:raise ValueError('dims')
 return im

def center_crop(im,frac=.60):
 w,h=im.size; cw=int(w*frac); ch=int(h*frac); x=(w-cw)//2;y=(h-ch)//2; return im.crop((x,y,x+cw,y+ch))
def candidate_views(im):
 w,h=im.size; out=[im,center_crop(im,.60)]; cw=max(80,int(w*.55)); ch=max(80,int(h*.55)); xs=[0,(w-cw)//2,w-cw]; ys=[0,(h-ch)//2,h-ch]; seen=set()
 for y in ys:
  for x in xs:
   b=(x,y,x+cw,y+ch)
   if b not in seen:out.append(im.crop(b));seen.add(b)
 return out
def gray(im):return ImageOps.autocontrast(ImageOps.grayscale(im)).convert('RGB')
def edge(im):
 g=ImageOps.autocontrast(ImageOps.grayscale(im)); e=g.filter(ImageFilter.FIND_EDGES); e=ImageOps.autocontrast(ImageOps.invert(e)); return e.convert('RGB')
def embed(model,proc,ims,batch=64):
 outs=[]
 with torch.inference_mode():
  for i in range(0,len(ims),batch):
   x=proc(images=ims[i:i+batch],return_tensors='pt');x={k:v.to(DEVICE) for k,v in x.items()}; y=model(**x).last_hidden_state[:,0].float(); y=torch.nn.functional.normalize(y,dim=1);outs.append(y.cpu().numpy())
 return np.concatenate(outs)
def rrf(a,b,k=60):
 ra=np.empty(len(a),int); rb=np.empty(len(b),int);ra[np.argsort(-a)]=np.arange(1,len(a)+1);rb[np.argsort(-b)]=np.arange(1,len(b)+1);return 1/(k+ra)+1/(k+rb)
def mskey(rows,url):
 for r in rows:
  m=(r.get('manifest') or '').strip()
  if m.startswith('http'):return 'm:'+re.sub(r'[?#].*','',m).lower()
 for r in rows:
  c=(r.get('cote') or '').strip().lower()
  if c:return 'c:'+re.sub(r'\s+',' ',c)
 return 'u:'+url

def acquire(rows):
 mus=sorted({(r.get('manifest') or '').strip() for r in rows if (r.get('manifest') or '').startswith('http') and 'handschriftenportal.de/workspace' not in (r.get('manifest') or '') and not(r.get('thumb_usable') is True and (r.get('thumb') or '').startswith('http'))})
 mc={}; mf=Counter()
 def mfet(u):
  try:r=requests.get(u,headers={'User-Agent':'q9q10-scale-comparanda/0.2'},timeout=6);r.raise_for_status();return u,r.json(),None
  except Exception as e:return u,None,type(e).__name__
 with ThreadPoolExecutor(max_workers=40) as ex:
  for k,f in enumerate(as_completed([ex.submit(mfet,u) for u in mus]),1):
   u,m,e=f.result();mc[u]=m if m is not None else None;mf[e]+=bool(e)
 def resolve(r):
  t=(r.get('thumb') or '').strip()
  if r.get('thumb_usable') is True and t.startswith('http'):return t,'thumb'
  m=mc.get((r.get('manifest') or '').strip())
  if not m:return None,'manifest_unavailable'
  cs=((m.get('sequences') or [{}])[0].get('canvases') or []) if 'sequences' in m else (m.get('items') or []); toks=folio_tokens(r.get('folio'))
  for tok in toks:
   for c in cs:
    lab=normfolio(nlabel(c.get('label')))
    if tok and (tok==lab or tok in lab or (lab and lab in tok)):
     u=canvas_url(c)
     if u:return u,'manifest_folio'
  if len(cs)==1:
   u=canvas_url(cs[0]);return (u,'manifest_single') if u else (None,'manifest_shape')
  return None,'folio_unresolved'
 groups=defaultdict(list);paths=Counter();un=Counter()
 for r in rows:
  u,p=resolve(r);paths[p]+=1
  if u:groups[u].append(r)
  else:un[p]+=1
 print('ACQ_URLS',len(groups),'ROW_RESOLUTION',dict(paths),'UNRESOLVED',dict(un),'MANIFEST_FAIL',dict(mf),flush=True)
 entities=[];df=Counter()
 def dl(u,rs):
  try:return u,rs,fetch_img(u),None
  except Exception as e:return u,rs,None,type(e).__name__
 items=list(groups.items())
 with ThreadPoolExecutor(max_workers=32) as ex:
  fs=[ex.submit(dl,u,rs) for u,rs in items]
  for k,f in enumerate(as_completed(fs),1):
   u,rs,im,e=f.result()
   if im is None:df[e]+=1
   else:
    entities.append({'url':u,'rows':rs,'classes':sorted({x.get('vms_class_v2') for x in rs if x.get('vms_class_v2')}),'mskey':mskey(rs,u),'im':im})
   if k%100==0:print('DOWNLOAD',k,'OF',len(items),'OK',len(entities),'FAIL',sum(df.values()),flush=True)
 print('ACQ_ENTITIES',len(entities),'UNIQUE_URLS',len(items),'DOWNLOAD_FAIL',dict(df),flush=True)
 return entities,{'manifest_total':len(mus),'manifest_ok':sum(v is not None for v in mc.values()),'resolved_unique_urls':len(items),'entities':len(entities),'paths':dict(paths),'unresolved':dict(un),'manifest_fail':dict(mf),'download_fail':dict(df)}

def rank_diverse(scores,entities,exclude_ms=None,limit=20):
 order=np.argsort(-scores);out=[];seen=set()
 for j in order:
  m=entities[j]['mskey']
  if m==exclude_ms or m in seen:continue
  seen.add(m);out.append(int(j))
  if len(out)>=limit:break
 return out

def main():
 st=time.time();print('RUN_START',RUN_ID,flush=True);rows=query_rows();print('ROWS',len(rows),Counter(r.get('vms_class_v2') for r in rows),flush=True)
 entities,acq=acquire(rows)
 proc=AutoImageProcessor.from_pretrained(DINO_MODEL,token=True);model=AutoModel.from_pretrained(DINO_MODEL,token=True,dtype=DTYPE).to(DEVICE).eval()
 tnames=list(TARGETS); tg={};te={}
 for n in tnames:
  im=fetch_img(TARGETS[n][0],12);vs=[im,center_crop(im,.60)];tg[n]=embed(model,proc,[gray(x) for x in vs],16);te[n]=embed(model,proc,[edge(x) for x in vs],16)
 EG=[];EE=[]
 for base in range(0,len(entities),24):
  sub=entities[base:base+24];views=[];owners=[]
  for oi,e in enumerate(sub):
   vs=candidate_views(e['im']);e['nviews']=len(vs);views+=vs;owners += [oi]*len(vs)
  ag=embed(model,proc,[gray(x) for x in views]);ae=embed(model,proc,[edge(x) for x in views]);by=defaultdict(list)
  for j,o in enumerate(owners):by[o].append(j)
  for oi,e in enumerate(sub):EG.append(ag[by[oi]]);EE.append(ae[by[oi]])
  print('EMBED',min(base+24,len(entities)),'OF',len(entities),flush=True)
 for e in entities:e.pop('im',None)
 cal={}; overall=True
 for cls in ['zodiac_aries','zodiac_pisces','astro_diagram']:
  qidx=sorted([i for i,e in enumerate(entities) if cls in e['classes']],key=lambda i:hashlib.sha256(entities[i]['url'].encode()).hexdigest())
  chosen=[];seen=set()
  for i in qidx:
   if entities[i]['mskey'] not in seen:chosen.append(i);seen.add(entities[i]['mskey'])
   if len(chosen)==8:break
  enrich=[];qout=[]
  for qi in chosen:
   sg=np.array([float((EG[j] @ EG[qi][:2].T).max()) for j in range(len(entities))]);se=np.array([float((EE[j] @ EE[qi][:2].T).max()) for j in range(len(entities))]);f=rrf(sg,se);top=rank_diverse(f,entities,entities[qi]['mskey'],20)
   elig_ms={e['mskey'] for e in entities if e['mskey']!=entities[qi]['mskey']};class_ms={e['mskey'] for e in entities if e['mskey']!=entities[qi]['mskey'] and cls in e['classes']};basep=len(class_ms)/max(1,len(elig_ms));share=sum(cls in entities[j]['classes'] for j in top)/max(1,len(top));ef=share/basep if basep else 0;enrich.append(ef);qout.append({'query_id':entities[qi]['rows'][0].get('id'),'query_ms':entities[qi]['mskey'],'top20_same':sum(cls in entities[j]['classes'] for j in top),'top20_n':len(top),'baseline':basep,'enrichment':ef})
  med=float(statistics.median(enrich)) if enrich else 0;above=sum(x>1 for x in enrich);passed=(len(chosen)==8 and med>=2 and above>=6);overall &= passed;cal[cls]={'n_queries':len(chosen),'median_enrichment':med,'queries_above_1':above,'pass':passed,'queries':qout}
 print('CALIBRATION_JSON='+json.dumps({'overall_pass':overall,'classes':cal},ensure_ascii=False),flush=True)
 tops={};compact=[]
 for n in tnames:
  sg=np.array([float((EG[j] @ tg[n].T).max()) for j in range(len(entities))]);se=np.array([float((EE[j] @ te[n].T).max()) for j in range(len(entities))]);f=rrf(sg,se);top=rank_diverse(f,entities,None,TOPK);lst=[]
  for rank,j in enumerate(top,1):
   r=entities[j]['rows'][0];z={'target':n,'rank':rank,'id':r.get('id'),'classes':entities[j]['classes'],'title':r.get('title'),'cote':r.get('cote'),'folio':r.get('folio'),'datation':r.get('datation'),'date_start':r.get('date_start'),'holding_region':r.get('holding_region'),'subject':r.get('subject'),'image_url':entities[j]['url'],'record_url':r.get('record_url'),'gray_score':float(sg[j]),'edge_score':float(se[j]),'rrf':float(f[j]),'mskey':entities[j]['mskey']};lst.append(z)
   if rank<=5:compact.append(z)
  tops[n]=lst
 print('TARGET_TOP_JSON='+json.dumps(compact,ensure_ascii=False),flush=True)
 meta={'run_id':RUN_ID,'row_count':len(rows),'acquisition':acq,'unique_entities':len(entities),'calibration_pass':overall,'calibration':cal,'model':DINO_MODEL,'elapsed_pre_vlm':round(time.time()-st,1)}
 print('META_JSON='+json.dumps(meta,ensure_ascii=False),flush=True)
 if not overall:
  print('GATE_STOP calibration_failed',flush=True);print('RUN_COMPLETE',RUN_ID,flush=True);return
 del model,proc;gc.collect();torch.cuda.empty_cache()
 from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
 qm='Qwen/Qwen2.5-VL-3B-Instruct';q=Qwen2_5_VLForConditionalGeneration.from_pretrained(qm,dtype=DTYPE,device_map='auto').eval();qp=AutoProcessor.from_pretrained(qm);out=[]
 def iv(x):
  try:return int(x)
  except:return 0
 for n in tnames:
  for z in tops[n][:3]:
   try:
    a=fetch_img(TARGETS[n][0],12);b=fetch_img(z['image_url'],12);prompt='Compare visible morphology only. Do not infer titles, date, shelfmark, subject or Voynich theory. Frozen target morphology: '+TARGETS[n][1]+'. Return only JSON integers layout_0_3, center_0_2, partition_count_0_2, line_morphology_0_2, object_class_0_2, text_placement_0_2; boolean fatal_mismatch; rationale <=35 words.';msgs=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]}];txt=qp.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);inp=qp(text=[txt],images=[a,b],padding=True,return_tensors='pt').to(q.device);gen=q.generate(**inp,max_new_tokens=170,do_sample=False);ans=qp.batch_decode(gen[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0];m=re.search(r'\{.*\}',ans,re.S);p=json.loads(m.group(0)) if m else {'raw':ans};score=sum(iv(p.get(k,0)) for k in ['layout_0_3','center_0_2','partition_count_0_2','line_morphology_0_2','object_class_0_2','text_placement_0_2']);fatal=bool(p.get('fatal_mismatch',True));o={**z,'blind':p,'blind_score':score,'screen_pass':score>=8 and not fatal,'screen_strong':score>=10 and not fatal};out.append(o);print('ADJ',n,z['rank'],z['id'],score,fatal,flush=True)
   except Exception as e:out.append({**z,'error':repr(e)});print('ADJ_FAIL',n,z['rank'],type(e).__name__,flush=True)
 print('ADJUDICATION_JSON='+json.dumps(out,ensure_ascii=False),flush=True);print('FINAL_JSON='+json.dumps({'run_id':RUN_ID,'calibration_pass':overall,'adjudicated':len(out),'passes':sum(x.get('screen_pass',False) for x in out),'strong':sum(x.get('screen_strong',False) for x in out),'elapsed':round(time.time()-st,1)},ensure_ascii=False),flush=True);print('RUN_COMPLETE',RUN_ID,flush=True)
if __name__=='__main__':main()
