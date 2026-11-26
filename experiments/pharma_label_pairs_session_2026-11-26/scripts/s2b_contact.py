import json, random
from PIL import Image, ImageDraw, ImageFont
S=json.load(open('ckpt/s2b_setup.json')); rng=random.Random(2611)
FT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
codes={}; used=set()
for f in S['group']:
    while True:
        c=''.join(rng.choice('BCDFGHJKLMNPRSTVWXZ') for _ in range(3))
        if c not in used: used.add(c); codes[f]=c; break
json.dump(codes,open('ckpt/s2b_fragcodes_SEALED.json','w'))
T=230; n=0
for g in ('R','L','RL','X'):
    fs=[f for f in S['group'] if S['group'][f]==g]; rng.shuffle(fs)
    for k in range(0,len(fs),30):
        sh=Image.new('RGB',(6*(T+10),5*(T+34)),(225,225,225)); d=ImageDraw.Draw(sh)
        for j,f in enumerate(fs[k:k+30]):
            im=Image.open(f'frag/F{f}.png'); im.thumbnail((T,T)); X=(j%6)*(T+10)+5; Y=(j//6)*(T+34)+30
            bg=Image.new('RGB',(T,T),(255,255,255)); bg.paste(im,((T-im.width)//2,(T-im.height)//2)); sh.paste(bg,(X,Y))
            d.text((X,Y-27),codes[f],fill=(0,0,0),font=FT)
        sh.save(f'sheets2b/contact_{g}_{k//30}.jpg',quality=88); n+=1
print(n)
