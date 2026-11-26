# whole-fragment composites: union of the fragment's object masks applied to the Yale region (no labels outside masks)
import json, io, os, subprocess
from PIL import Image, ImageFilter, ImageChops
S=json.load(open('ckpt/s2b_setup.json'))
REF={f"{o['folio_seq']}:{o['obj_index']}":o for o in json.load(open('data/ref_objects.json'))}
W={int(l.split()[0]):int(l.split()[1]) for l in open('yale_dims.txt')}
SEQ2IMG={161:1037112,162:1006233,163:1006234,164:1006235,175:1006246,176:1006247,177:1006248,178:1006249,179:1006250,180:1006251,181:1006252,182:1006253}
def get(url,fn):
    if not os.path.exists(fn):
        open(fn,'wb').write(subprocess.run(['curl','-s','--max-time','90',url],capture_output=True).stdout)
    return Image.open(fn)
for f,objs in S['frag_objs'].items():
    out=f'frag/F{f}.png'
    if os.path.exists(out): continue
    seq=int(objs[0].split(':')[0]); iid=SEQ2IMG[seq]; s=W[iid]/1000
    bb=[REF[o]['bbox'] for o in objs]
    x0=min(b['x'] for b in bb); y0=min(b['y'] for b in bb); x1=max(b['x']+b['w'] for b in bb); y1=max(b['y']+b['h'] for b in bb)
    X,Y,Wd,Hd=int(x0*s),int(y0*s),int((x1-x0)*s),int((y1-y0)*s)
    sc=min(1.0,600/max(Wd,Hd)); ow=max(1,int(Wd*sc))
    im=get(f'https://collections.library.yale.edu/iiif/2/{iid}/{X},{Y},{Wd},{Hd}/{ow},/0/default.jpg',f'frag/raw_{f}.jpg').convert('RGB')
    m=Image.new('L',(x1-x0,y1-y0),0)
    for o in objs:
        mk=get(REF[o]['mask_url'],'masks/'+o.replace(':','_')+'.png').split()[-1]
        m=ImageChops.lighter(m,mk.crop((x0,y0,x1,y1)))
    m=m.filter(ImageFilter.MaxFilter(5)).resize(im.size,Image.NEAREST)
    Image.composite(im,Image.new('RGB',im.size,(255,255,255)),m).save(out)
print(len(os.listdir('frag')))
