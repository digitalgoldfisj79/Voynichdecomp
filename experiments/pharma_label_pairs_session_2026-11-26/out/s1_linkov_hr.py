# hi-res link overlays per page: fetch the page region from Yale IIIF at 1500 px wide
import pickle, csv, sys, os, subprocess
from PIL import Image, ImageDraw, ImageFont
J=pickle.load(open('ckpt/s1_join_m300_r800.pkl','rb'))
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
CC={'HIGH':(0,160,0),'MED':(0,90,255),'LOW':(255,120,0),'NONE':(255,0,0)}
pg,iid=sys.argv[1],int(sys.argv[2]); X0,X1=int(sys.argv[3]),int(sys.argv[4])  # Yale px x-range
Y0,Y1=(int(sys.argv[5]),int(sys.argv[6])) if len(sys.argv)>6 else (0,None)
fn=f'img/reg_{iid}_{X0}_{X1}.jpg'
if not os.path.exists(fn):
    H=3900
    subprocess.run(['curl','-s','--max-time','120','-o',fn,f'https://collections.library.yale.edu/iiif/2/{iid}/{X0},0,{X1-X0},{H}/1500,/0/default.jpg'],check=True)
im=Image.open(fn).convert('RGB'); d=ImageDraw.Draw(im); k=1500/(X1-X0)
T=lambda x,y:((x-X0)*k,y*k)
lab={}
for c in conc:
    if c['page']==pg and c['yale_image_id']==str(iid) and c['box_x'] and c['locus_type']=='Lf':
        x,y,w,h=[int(c[q]) for q in ('box_x','box_y','box_w','box_h')]
        a=T(x,y); b=T(x+w,y+h); lab[c['zl_locus']]=((a[0]+b[0])/2,(a[1]+b[1])/2)
        d.rectangle([a,b],outline=(0,0,0),width=2); d.text((a[0],a[1]-22),'F'+(c['fragment_num'] or c['fragment_alt']),fill=(0,0,0),font=F)
for r in J:
    if r['page']!=pg or r['yale_image_id']!=iid: continue
    a=T(r['ox'],r['oy']); b=T(r['ox']+r['ow'],r['oy']+r['oh']); col=CC[r['conf']]
    d.rectangle([a,b],outline=col,width=2); d.text((a[0]+3,a[1]+3),f"{r['obj_index']}{r['obj_class'][0]}"+('*' if r['span'] else ''),fill=col,font=F)
    if r['zl_locus'] in lab: d.line([((a[0]+b[0])/2,(a[1]+b[1])/2),lab[r['zl_locus']]],fill=col,width=2)
if Y1: im=im.crop((0,int(Y0*k),im.width,int(Y1*k)))
out=f'ov/hr_{pg}_{iid}_{Y0}.jpg'; im.save(out,quality=88); print(out,im.size)
