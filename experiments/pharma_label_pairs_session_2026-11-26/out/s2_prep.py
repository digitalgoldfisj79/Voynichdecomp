# Stage 2: features for shortlisted pairs, final selection, blind sheets. No label data is read here.
import json, pickle, random, os, subprocess, hashlib, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
random.seed(20261126)
R={r['obj_id'].replace('voynich_pharma:',''):r for r in pickle.load(open('ckpt/s1_final.pkl','rb'))}
REF={f"{o['folio_seq']}:{o['obj_index']}":o for o in json.load(open('data/ref_objects.json'))}
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
pairs=[]
for l in open('data/s2_sql_top.txt'):
    c,s=l.strip().split('|')
    for t in s.split(';'):
        a,b,tc,rc,ms=t.split(','); pairs.append(dict(cls=c,a=a,b=b,tcos=float(tc),rc=float(rc),msim=float(ms),src='shortlist'))
def crop(k):
    fn=f'crops/{k.replace(":","_")}.png'
    if os.path.exists(fn): return Image.open(fn)
    o=REF[k]; r=R[k]; iid=r['yale_image_id']; s=W[iid]/1000
    x,y,w,h=r['yale_x'],r['yale_y'],r['yale_w'],r['yale_h']
    scale=min(1.0,500/max(w,h)); ow=max(1,int(w*scale))
    raw=subprocess.run(['curl','-s','--max-time','60',f'https://collections.library.yale.edu/iiif/2/{iid}/{x},{y},{w},{h}/{ow},/0/default.jpg'],capture_output=True).stdout
    im=Image.open(io.BytesIO(raw)).convert('RGB')
    mraw=subprocess.run(['curl','-s','--max-time','60',o['mask_url']],capture_output=True).stdout
    m=Image.open(io.BytesIO(mraw)).split()[-1]
    b=o['bbox']; mc=m.crop((b['x'],b['y'],b['x']+b['w'],b['y']+b['h'])).filter(ImageFilter.MaxFilter(5)).resize(im.size,Image.NEAREST)
    white=Image.new('RGB',im.size,(255,255,255)); out=Image.composite(im,white,mc)
    out.save(fn); return out
def feats(k):
    im=np.asarray(crop(k)).astype(float); msk=(im.sum(2)<750)
    col=im[msk].mean(0) if msk.sum()>20 else np.array([200,200,200])
    r=R[k]; return dict(ar=np.log(r['yale_w']/r['yale_h']),col=col)
# random pairs (calibration), drawn before features, from linked root/leaf objects on different fragments
linked=[k for k,r in R.items() if r['fragment_final'] and r['obj_class'] in ('root','leaf') and '/' not in str(r['fragment_final'])]
short={(p['a'],p['b']) for p in pairs}|{(p['b'],p['a']) for p in pairs}
rnd=[]
while len(rnd)<12:
    a,b=random.sample(linked,2)
    if R[a]['obj_class']!=R[b]['obj_class'] or R[a]['fragment_final']==R[b]['fragment_final'] or (a,b) in short: continue
    rnd.append(dict(cls=R[a]['obj_class'],a=a,b=b,src='random'))
F={}
for p in pairs+rnd:
    for k in (p['a'],p['b']):
        if k not in F: F[k]=feats(k)
for p in pairs:
    p['ard']=abs(F[p['a']]['ar']-F[p['b']]['ar']); p['cold']=float(np.linalg.norm(F[p['a']]['col']-F[p['b']]['col']))
# percentile ranks within class (higher = more alike)
sel=[]
for c in ('root','leaf'):
    P=[p for p in pairs if p['cls']==c]; n=len(P)
    for key,sign in (('tcos',1),('rc',1),('ard',-1),('cold',-1)):
        order=sorted(P,key=lambda p:sign*p[key])
        for i,p in enumerate(order): p['pr_'+key]=i/(n-1)
    for p in P: p['score']=np.mean([p['pr_tcos'],p['pr_rc'],p['pr_ard'],p['pr_cold']])
    P.sort(key=lambda p:-p['score'])
    seen=set()
    for p in P:
        fp=tuple(sorted([R[p['a']]['fragment_final'],R[p['b']]['fragment_final']]))
        if fp in seen: continue
        seen.add(fp); sel.append(p)
        if sum(1 for q in sel if q['cls']==c)>=50: break
allp=sel+rnd; random.shuffle(allp)
codes=set()
for p in allp:
    while True:
        cd=''.join(random.choice('BCDFGHJKLMNPQRSTVWXZ') for _ in range(4))
        if cd not in codes: codes.add(cd); p['code']=cd; break
key=[dict(code=p['code'],a=p['a'],b=p['b'],cls=p['cls'],src=p['src'],fa=R[p['a']]['fragment_final'],fb=R[p['b']]['fragment_final'],
          score=round(p.get('score',float('nan')),3)) for p in allp]
json.dump(key,open('ckpt/s2_key_SEALED.json','w'))
print('pairs viewed',len(allp),'shortlist',len(sel),'random',len(rnd),'objects cropped',len(F))
# sheets: 8 pairs per sheet, code only
try: FT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26)
except: FT=None
def fit(im,S=330):
    im=im.copy(); im.thumbnail((S,S)); bg=Image.new('RGB',(S,S),(255,255,255)); bg.paste(im,((S-im.width)//2,(S-im.height)//2)); return bg
for si in range(0,len(allp),8):
    sh=Image.new('RGB',(1400,4*380),(235,235,235)); d=ImageDraw.Draw(sh)
    for j,p in enumerate(allp[si:si+8]):
        X=(j%2)*700; Y=(j//2)*380
        sh.paste(fit(crop(p['a'])),(X+10,Y+40)); sh.paste(fit(crop(p['b'])),(X+350,Y+40))
        d.text((X+10,Y+5),p['code'],fill=(0,0,0),font=FT); d.line([(X+695,Y),(X+695,Y+380)],fill=(0,0,0),width=3)
    sh.save(f'sheets/sheet_{si//8:02d}.jpg',quality=88)
print('sheets',(len(allp)+7)//8)
