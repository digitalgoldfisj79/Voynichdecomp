import json, random
from PIL import Image, ImageDraw, ImageFont
S=json.load(open('ckpt/s2b_setup.json')); codes=json.load(open('ckpt/s2b_fragcodes_SEALED.json')); inv={v:k for k,v in codes.items()}
nom=[tuple(l.split()) for l in open('out/s2b_nominations.txt') if l.strip()]
pairs=[dict(a=inv[a],b=inv[b],src='nominated') for a,b in nom]
pairs+=[dict(a=a,b=b,src='random') for a,b in S['random_pairs']]
# a random pair that is also nominated stays once as nominated
seen=set(); P=[]
for p in pairs:
    k=tuple(sorted((p['a'],p['b']),key=int))
    if k in seen: continue
    seen.add(k); P.append(p)
rng=random.Random(2627); rng.shuffle(P); used=set()
for p in P:
    while True:
        c=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(4))
        if c not in used: used.add(c); p['code']=c; break
    if rng.random()<0.5: p['a'],p['b']=p['b'],p['a']
json.dump(P,open('ckpt/s2b_pairkey_SEALED.json','w'))
FT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26); T=440
def fit(f):
    im=Image.open(f'frag/F{f}.png'); im.thumbnail((T,T)); bg=Image.new('RGB',(T,T),(255,255,255)); bg.paste(im,((T-im.width)//2,(T-im.height)//2)); return bg
for i in range(0,len(P),4):
    sh=Image.new('RGB',(2*(2*T+30),2*(T+40)),(225,225,225)); d=ImageDraw.Draw(sh)
    for j,p in enumerate(P[i:i+4]):
        X=(j%2)*(2*T+30); Y=(j//2)*(T+40)
        sh.paste(fit(p['a']),(X+5,Y+36)); sh.paste(fit(p['b']),(X+T+15,Y+36)); d.text((X+5,Y+4),p['code'],fill=(0,0,0),font=FT)
        d.line([(X+2*T+25,Y),(X+2*T+25,Y+T+40)],fill=(0,0,0),width=4)
    sh.save(f'sheets2b/grade_{i//4:02d}.jpg',quality=88)
print(len(P),sum(p['src']=='nominated' for p in P),sum(p['src']=='random' for p in P),(len(P)+3)//4,'sheets')
