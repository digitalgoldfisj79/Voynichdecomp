from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF

SEED=20260814
MAX_WINDOWS=4
CANDIDATES=256
WIN_H,WIN_W=96,320
OUT=224
INK_MIN,INK_MAX=0.02,0.35
MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big') & 0x7fffffffffffffff


def writer_bucket(w):
    h=hashlib.sha256(('u6v02-writer|'+w).encode()).hexdigest()
    return int(h[:8],16)%10


def split_name(w):
    b=writer_bucket(w)
    return 'train' if b<=5 else 'calibration' if b<=7 else 'locked'


def image_paths(root):
    exts={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.pgm'}
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts])


def writer_id(path):
    base=path.name
    return base.split('-',1)[0] if '-' in base else None


def read_gray(path):
    arr=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if arr is None: raise ValueError(f'cannot decode {path}')
    return arr.astype(np.float32)/255.0


def border_mean(g):
    h,w=g.shape;dy=max(1,int(round(.05*h)));dx=max(1,int(round(.05*w)))
    vals=np.concatenate([g[:dy,:].ravel(),g[-dy:,:].ravel(),g[:, :dx].ravel(),g[:, -dx:].ravel()])
    return float(vals.mean())


def normalize_page(path):
    g=read_gray(path);h,w=g.shape
    y=int(round(.02*h));x=int(round(.02*w))
    if h-2*y>=10 and w-2*x>=10:g=g[y:h-y,x:w-x]
    H,W=g.shape
    cy0,cy1=int(.2*H),int(.8*H);cx0,cx1=int(.2*W),int(.8*W)
    bm=border_mean(g);cm=float(g[cy0:cy1,cx0:cx1].mean()) if cy1>cy0 and cx1>cx0 else float(g.mean())
    ink=(1.0-g) if bm>=cm else g
    bw=(ink>.5).astype(np.uint8)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(bw,8)
    heights=[]
    if n>1:
        for st in stats[1:]:
            x0,y0,ww,hh,area=map(int,st)
            if area>=3 and hh>=2:heights.append(hh)
    if len(heights)>=20:
        med=float(np.median(heights));scale=18.0/med if med>0 else 1.0;fallback=False
    else:
        scale=1200.0/max(1,ink.shape[0]);fallback=True
    raw_scale=scale;scale=min(4.0,max(.25,scale));capped=(scale!=raw_scale)
    nw=max(1,int(round(ink.shape[1]*scale)));nh=max(1,int(round(ink.shape[0]*scale)))
    ink=cv2.resize(ink,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    return np.clip(ink,0,1).astype(np.float32),{'components':len(heights),'fallback_scale':fallback,'scale_capped':capped,'scale':scale,'shape':[nh,nw]}


def choose_windows(path,ink):
    h,w=ink.shape
    if h<WIN_H or w<WIN_W:return []
    rng=random.Random(stable_seed('u6v02-window',path.name))
    out=[];seen=set()
    for _ in range(CANDIDATES):
        y=rng.randrange(0,h-WIN_H+1);x=rng.randrange(0,w-WIN_W+1)
        if (y,x) in seen:continue
        seen.add((y,x));frac=float((ink[y:y+WIN_H,x:x+WIN_W]>.5).mean())
        if INK_MIN<=frac<=INK_MAX:
            out.append((y,x,frac))
            if len(out)>=MAX_WINDOWS:break
    return out


@dataclass
class PageSpec:
    path:str
    writer:str
    split:str
    windows:list


@lru_cache(maxsize=64)
def cached_page(path):
    return normalize_page(Path(path))[0]


def crop_tensor(spec:PageSpec,win,augment=False,rng=None):
    ink=cached_page(spec.path);y,x,_=win
    c=ink[y:y+WIN_H,x:x+WIN_W]
    im=Image.fromarray(np.uint8(np.clip(c,0,1)*255),mode='L').resize((OUT,OUT),Image.Resampling.BILINEAR)
    z=torch.from_numpy(np.asarray(im,dtype=np.float32)/255.0).unsqueeze(0)
    if augment:
        assert rng is not None
        if rng.random()<0.25:
            if rng.random()<0.5:z=F.max_pool2d(z.unsqueeze(0),3,1,1).squeeze(0)
            else:z=-F.max_pool2d((-z).unsqueeze(0),3,1,1).squeeze(0)
        angle=rng.uniform(-2,2);dx=rng.randint(-4,4);dy=rng.randint(-4,4)
        z=TF.affine(z,angle=angle,translate=[dx,dy],scale=1.0,shear=[0.0,0.0],fill=0.0)
    rgb=(1.0-z).repeat(3,1,1)
    return (rgb-MEAN)/STD


class WindowDataset(Dataset):
    def __init__(self,pages,writer_to_i,epoch):
        self.items=[(p,w) for p in pages for w in p.windows]
        self.w2i=writer_to_i;self.epoch=epoch
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        p,w=self.items[i];rng=random.Random(stable_seed('u6v02-aug',self.epoch,i,p.path))
        return crop_tensor(p,w,True,rng),self.w2i[p.writer]


class WriterNet(nn.Module):
    def __init__(self,nwriters):
        super().__init__();base=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base.fc=nn.Identity();self.base=base;self.embed=nn.Linear(512,128);self.cls=nn.Linear(128,nwriters)
    def forward(self,x):
        h=self.base(x);e=F.normalize(self.embed(h),dim=1);return e,self.cls(e)


def page_embeddings(model,pages,device,blank=False):
    model.eval();out={}
    with torch.no_grad():
        blank_t=None
        if blank:
            z=torch.zeros(1,OUT,OUT);rgb=(1-z).repeat(3,1,1);blank_t=((rgb-MEAN)/STD).to(device)
        for p in pages:
            if blank:
                X=torch.stack([blank_t for _ in p.windows])
            else:X=torch.stack([crop_tensor(p,w,False) for w in p.windows]).to(device)
            e,_=model(X);m=F.normalize(e.mean(0,keepdim=True),dim=1)[0].cpu().numpy();out[p.path]=m
    return out


def pair_panel(pages,emb):
    by=defaultdict(list)
    for p in pages:by[p.writer].append(p)
    positives=[]
    for w,ps in by.items():
        for a,b in combinations(ps,2):positives.append((a,b))
    if not positives:return None
    allpages=list(pages);neg=[]
    for idx,(a,b) in enumerate(positives):
        rng=random.Random(stable_seed('u6v02-neg',a.path,b.path,idx))
        target=(len(a.windows)+len(b.windows))/2
        cand=[p for p in allpages if p.writer!=a.writer and abs(len(p.windows)-target)<=2]
        if not cand:cand=[p for p in allpages if p.writer!=a.writer]
        if not cand:continue
        cand.sort(key=lambda p:(abs(len(p.windows)-target),stable_seed('u6v02-order',p.path)))
        c=cand[rng.randrange(min(len(cand),20))];neg.append((a,c))
    n=min(len(positives),len(neg));positives=positives[:n];neg=neg[:n]
    y=[1]*n+[0]*n;s=[]
    for a,b in positives+neg:s.append(float(np.dot(emb[a.path],emb[b.path])))
    return {'auc':float(roc_auc_score(y,s)),'positive_pairs':n,'negative_pairs':n,'positive_scores':s[:n],'negative_scores':s[n:]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    paths=image_paths(a.data);audit={'images_discovered':len(paths),'decode_failures':0,'no_writer_id':0,'no_windows':0,'scale_fallback_pages':0,'scale_capped_pages':0}
    specs=[]
    for j,p in enumerate(paths):
        w=writer_id(p)
        if not w:audit['no_writer_id']+=1;continue
        try:ink,meta=normalize_page(p)
        except Exception:audit['decode_failures']+=1;continue
        audit['scale_fallback_pages']+=int(meta['fallback_scale']);audit['scale_capped_pages']+=int(meta['scale_capped'])
        ws=choose_windows(p,ink)
        if not ws:audit['no_windows']+=1;continue
        specs.append(PageSpec(str(p),w,split_name(w),ws))
        if (j+1)%200==0:print('INDEX_PROGRESS',j+1,'/',len(paths),flush=True)
    split_pages={s:[p for p in specs if p.split==s] for s in ('train','calibration','locked')}
    split_stats={s:{'writers':len({p.writer for p in ps}),'pages':len(ps),'windows':sum(len(p.windows) for p in ps)} for s,ps in split_pages.items()}
    result={'schema':'frontier-u6-v0.2-external','target_opened':False,'voynich_read':False,'audit':audit,'split_stats':split_stats,'formal_verdict':None}
    for s,st in split_stats.items():
        if st['writers']<30 or st['pages']<60:
            result['formal_verdict']='FAIL_EXTERNAL_DATA_GATE';result['failed_split']=s
            (a.out/'U6_EXTERNAL_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True));print('U6_FINAL',json.dumps(result,sort_keys=True));return
    train=split_pages['train'];cal=split_pages['calibration'];locked=split_pages['locked']
    writers=sorted({p.writer for p in train});w2i={w:i for i,w in enumerate(writers)}
    device=torch.device('cpu');model=WriterNet(len(writers)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    history=[];best_auc=-1;best_state=None;best_epoch=None
    for epoch in range(1,6):
        model.train();ds=WindowDataset(train,w2i,epoch);g=torch.Generator().manual_seed(SEED+epoch)
        dl=DataLoader(ds,batch_size=64,shuffle=True,num_workers=2,generator=g,persistent_workers=False)
        loss_sum=0.0;n=0
        for bi,(x,y) in enumerate(dl):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);e,logit=model(x);loss=F.cross_entropy(logit,y);loss.backward();opt.step();loss_sum+=float(loss)*len(y);n+=len(y)
            if (bi+1)%50==0:print('TRAIN',epoch,bi+1,'/',len(dl),'loss',loss_sum/max(1,n),flush=True)
        emb=page_embeddings(model,cal,device,False);panel=pair_panel(cal,emb)
        auc=float(panel['auc']) if panel else float('nan');history.append({'epoch':epoch,'train_loss':loss_sum/max(1,n),'calibration_auc':auc,'calibration_pairs':panel['positive_pairs'] if panel else 0})
        print('EPOCH_RESULT',json.dumps(history[-1]),flush=True)
        if math.isfinite(auc) and auc>best_auc+1e-12:
            best_auc=auc;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    if best_state is None:
        result.update({'formal_verdict':'FAIL_EXTERNAL_DATA_GATE','reason':'no calibration pair AUC','history':history});(a.out/'U6_EXTERNAL_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True));return
    model.load_state_dict(best_state)
    locked_emb=page_embeddings(model,locked,device,False);locked_panel=pair_panel(locked,locked_emb)
    blank_emb=page_embeddings(model,locked,device,True);blank_panel=pair_panel(locked,blank_emb)
    locked_auc=float(locked_panel['auc']) if locked_panel else float('nan');blank_auc=float(blank_panel['auc']) if blank_panel else float('nan')
    if not math.isfinite(locked_auc):verdict='FAIL_EXTERNAL_DATA_GATE'
    elif locked_auc<0.80:verdict='FAIL_EXTERNAL_AUC'
    elif not math.isfinite(blank_auc) or blank_auc>0.60:verdict='FAIL_EXTERNAL_NUISANCE'
    else:verdict='PASS_EXTERNAL'
    result.update({'formal_verdict':verdict,'selected_epoch':best_epoch,'calibration_auc':best_auc,'locked_auc':locked_auc,'background_only_auc':blank_auc,'history':history,'locked_panel':locked_panel,'background_panel':blank_panel,'target_may_open_stage_b':verdict=='PASS_EXTERNAL'})
    (a.out/'U6_EXTERNAL_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'writers':writers,'selected_epoch':best_epoch,'result_summary':{'locked_auc':locked_auc,'background_only_auc':blank_auc,'verdict':verdict}},a.out/'U6_EXTERNAL_ENCODER.pt')
    md=['# U6-v0.2 external writer instrument','',f'Formal verdict: **{verdict}**','',f'- selected epoch: {best_epoch}',f'- calibration AUC: {best_auc:.4f}',f'- locked unseen-writer AUC: {locked_auc:.4f} (gate >=0.80)',f'- background-only AUC: {blank_auc:.4f} (gate <=0.60)','',f'- train writers/pages/windows: {split_stats["train"]}',f'- calibration writers/pages/windows: {split_stats["calibration"]}',f'- locked writers/pages/windows: {split_stats["locked"]}','','Voynich was not read in this stage.']
    (a.out/'U6_EXTERNAL_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('U6_FINAL',json.dumps({'formal_verdict':verdict,'selected_epoch':best_epoch,'calibration_auc':best_auc,'locked_auc':locked_auc,'background_only_auc':blank_auc,'split_stats':split_stats},sort_keys=True),flush=True)

if __name__=='__main__':main()
