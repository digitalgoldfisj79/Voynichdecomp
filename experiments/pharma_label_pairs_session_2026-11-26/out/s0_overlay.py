# overlay object boxes (by class) and fragment-label boxes (fragment number only) on page images
import json, csv, sys
from PIL import Image, ImageDraw
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
SEQ2IMG={161:1037112,162:1006233,163:1006234,164:1006235,175:1006246,176:1006247,177:1006248,178:1006249,179:1006250,180:1006251,181:1006252,182:1006253}
objs=json.load(open('data/ref_objects.json'))
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
COL={'root':(200,0,0),'leaf':(0,150,0),'stem':(0,0,200),'flower':(200,0,200),'whole_plant':(255,140,0),'container':(120,120,120)}
for seq,iid in SEQ2IMG.items():
    im=Image.open(f'img/{iid}_2000.jpg').convert('RGB'); d=ImageDraw.Draw(im)
    k=2000/1000  # bbox units -> 2000px image
    for o in objs:
        if o['folio_seq']!=seq: continue
        b=o['bbox']; x,y,w,h=[b[c]*k for c in 'xywh']
        d.rectangle([x,y,x+w,y+h],outline=COL[o['obj_class']],width=3); d.text((x+3,y+3),f"{o['obj_index']}",fill=COL[o['obj_class']])
    kk=2000/W[iid]
    for r in conc:
        if r['yale_image_id']!=str(iid) or not r['box_x']: continue
        x,y,w,h=[int(r[c])*kk for c in ('box_x','box_y','box_w','box_h')]
        tag=('C' if r['locus_type']=='Lc' else 'F'+(r['fragment_num'] or r['fragment_alt']))+'/r'+(r['vnu_row'] or '-')
        d.rectangle([x,y,x+w,y+h],outline=(0,0,0),width=3); d.text((x,y-12),tag,fill=(0,0,0))
    im.save(f'ov/ov_{seq}_{iid}.jpg',quality=85)
print('done')
