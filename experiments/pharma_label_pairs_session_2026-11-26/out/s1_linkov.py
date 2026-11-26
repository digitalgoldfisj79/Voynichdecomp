import pickle, csv
from PIL import Image, ImageDraw, ImageFont
J=pickle.load(open('ckpt/s1_join_m300_r800.pkl','rb'))
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
try: F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
except: F=None
CC={'HIGH':(0,160,0),'MED':(0,90,255),'LOW':(255,120,0),'NONE':(255,0,0)}
pages={}
for r in J: pages.setdefault((r['yale_image_id'],r['page']),[]).append(r)
for (iid,pg),rs in pages.items():
    if pg in ('f90r_HERBAL','OFFPAGE'): continue
    im=Image.open(f'img/{iid}_2000.jpg').convert('RGB'); d=ImageDraw.Draw(im); k=2000/W[iid]
    lab={}
    for c in conc:
        if c['page']==pg and c['yale_image_id']==str(iid) and c['box_x'] and c['locus_type']=='Lf':
            x,y,w,h=[int(c[q])*k for q in ('box_x','box_y','box_w','box_h')]
            lab[c['zl_locus']]=(x+w/2,y+h/2); d.rectangle([x,y,x+w,y+h],outline=(0,0,0),width=3)
            d.text((x,y-24),'F'+(c['fragment_num'] or c['fragment_alt']),fill=(0,0,0),font=F)
    xs=[]
    for r in rs:
        x0,y0,x1,y1=r['ox']*k,r['oy']*k,(r['ox']+r['ow'])*k,(r['oy']+r['oh'])*k; xs+= [x0,x1]
        col=CC[r['conf']]; d.rectangle([x0,y0,x1,y1],outline=col,width=3)
        tag=f"{r['obj_index']}{r['obj_class'][0]}"+('*' if r['span'] else '')
        d.text((x0+4,y0+4),tag,fill=col,font=F)
        if r['zl_locus'] in lab:
            lx,ly=lab[r['zl_locus']]; d.line([( (x0+x1)/2,(y0+y1)/2),(lx,ly)],fill=col,width=3)
    for c in conc:
        if c['page']==pg and c['yale_image_id']==str(iid) and c['box_x']:
            xs+=[int(c['box_x'])*k,(int(c['box_x'])+int(c['box_w']))*k]
    a,b=max(0,min(xs)-60),min(im.width,max(xs)+60)
    im.crop((int(a),0,int(b),im.height)).save(f'ov/link_{pg}_{iid}.jpg',quality=85)
    print(pg,iid,int(a),int(b))
