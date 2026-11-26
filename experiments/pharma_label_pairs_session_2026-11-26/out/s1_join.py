import json, csv, math, pickle, sys, os
MARGIN=int(sys.argv[1]) if len(sys.argv)>1 else 300
REACH=int(sys.argv[2]) if len(sys.argv)>2 else 800
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
SEQ2IMG={161:1037112,162:1006233,163:1006234,164:1006235,175:1006246,176:1006247,177:1006248,178:1006249,179:1006250,180:1006251,181:1006252,182:1006253}
def page_of(seq,cx):
    if seq==161: return 'f88r'
    if seq==162: return 'f88v' if cx<310 else ('f89r1' if cx<580 else 'f89r2')
    if seq==163: return 'f89v2' if cx>=40 else 'OFFPAGE'
    if seq==164: return 'f89v1' if cx<385 else 'f90r_HERBAL'
    if seq==175: return 'f99r'
    if seq==176: return 'f99v'
    if seq==177: return 'f100r'
    if seq==178: return 'f100v' if cx<390 else 'f101r'
    if seq==179: return 'f101v'   # f101v2 panel image
    if seq==180: return 'f101v' if cx<350 else ('f102r1' if cx<640 else 'f102r2')
    if seq==181: return 'f102v2'
    if seq==182: return 'f102v1'
PAGEW={1006233:9078/3,1006235:7796*0.385,1006249:7486*0.39,1006251:8176*0.35}
objs=json.load(open('data/ref_objects.json'))
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
labels=[r for r in conc if r['locus_type']=='Lf' and r['box_x']]
out=[]
for o in objs:
    if o['obj_class']=='container': continue
    iid=SEQ2IMG[o['folio_seq']]; s=W[iid]/1000; b=o['bbox']
    x0,y0=b['x']*s,b['y']*s; x1,y1=x0+b['w']*s,y0+b['h']*s
    pg=page_of(o['folio_seq'],b['x']+b['w']/2)
    pw=PAGEW.get(iid,W[iid])
    if iid==1006233: pw=W[iid]/3
    cands=[]
    for r in labels:
        if r['page']!=pg or int(r['yale_image_id'])!=iid: continue
        cx=int(r['box_x'])+int(r['box_w'])/2; cy=int(r['box_y'])+int(r['box_h'])/2
        if not (y0-MARGIN<=cy<=y1+MARGIN): continue
        dx=max(x0-cx,0,cx-x1); dy=max(y0-cy,0,cy-y1)
        cands.append((math.hypot(dx,dy), r['zl_locus'], r['fragment_num'] or r['fragment_alt'], int(r['vnu_row'] or 0), (x0<=cx<=x1 and y0<=cy<=y1)))
    cands.sort()
    inside=sum(1 for c in cands if c[4])
    rec=dict(obj_id=o['reference_id'],folio_seq=o['folio_seq'],obj_index=o['obj_index'],obj_class=o['obj_class'],page=pg,yale_image_id=iid,
             ox=round(x0),oy=round(y0),ow=round(x1-x0),oh=round(y1-y0),n_cands=len(cands),labels_inside=inside)
    span = inside>=2 or (x1-x0)>0.45*pw
    if not cands or cands[0][0]>REACH:
        rec.update(fragment=None,zl_locus=None,vnu_row=None,dist=round(cands[0][0]) if cands else None,gap=None,conf='NONE')
    else:
        d=cands[0][0]; gap=(cands[1][0]-d) if len(cands)>1 else 9999
        conf='HIGH' if (d<=150 and gap>=150) else ('MED' if d<=400 else 'LOW')
        rec.update(fragment=cands[0][2],zl_locus=cands[0][1],vnu_row=cands[0][3],dist=round(d),gap=round(gap),conf=conf,
                   runner_up=cands[1][2] if len(cands)>1 else None)
    rec['span']=span; rec['ambig']= rec.get('fragment') is not None and '|' in str(rec['fragment'])
    out.append(rec)
os.makedirs('ckpt',exist_ok=True)
pickle.dump(out,open(f'ckpt/s1_join_m{MARGIN}_r{REACH}.pkl','wb'))
from collections import Counter
print(MARGIN,REACH,len(out),Counter(r['page'] for r in out if r['page'] in('OFFPAGE','f90r_HERBAL','f101r')),Counter(r['conf'] for r in out))
