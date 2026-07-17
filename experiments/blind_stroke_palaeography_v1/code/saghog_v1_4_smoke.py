#!/usr/bin/env python3
from __future__ import annotations

import base64, hashlib, io, json, math, os, random, re, sys, tarfile, time, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import solve
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_sauvola

SEED = 20260717
UPSTREAM_COMMIT = "123cf0f306f105a46edbe8def06f49b54e64832e"
ARCHIVE_URL = "https://zenodo.org/api/records/1324999/files/icdar17-historicalwi-training-color.zip/content"
ARCHIVE_MD5 = "e5ba2c7049bfb1453946233f681e4d53"
WORK = Path("/tmp/saghog_v14_smoke")
OUT = WORK / "output"


def log(event: str, **kw: Any) -> None:
    print(event + " " + json.dumps(kw, sort_keys=True), flush=True)


def digest(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, md5: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log("DOWNLOAD_BEGIN", url=url, dest=str(dest))
        with urllib.request.urlopen(url, timeout=180) as r, dest.open("wb") as w:
            while True:
                b = r.read(8 << 20)
                if not b:
                    break
                w.write(b)
        log("DOWNLOAD_END", bytes=dest.stat().st_size, sha256=digest(dest))
    if md5 and digest(dest, "md5") != md5:
        raise RuntimeError("archive MD5 mismatch")


def load_upstream():
    raw = urllib.request.urlopen(
        f"https://codeload.github.com/marco-peer/icdar24/zip/{UPSTREAM_COMMIT}", timeout=120
    ).read()
    log("UPSTREAM", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), commit=UPSTREAM_COMMIT)
    root = WORK / "upstream"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(root)
    src = next(root.glob("icdar24-*"))
    sys.path.insert(0, str(src))
    from mae.hog_openmim import MaskedAutoencoderViT
    from finetune.netvlad import Wrapper
    return MaskedAutoencoderViT, Wrapper


def extract_archive(archive: Path) -> Path:
    root = WORK / "historical_wi"
    if not root.exists():
        root.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                p = (root / info.filename).resolve()
                if not str(p).startswith(str(root.resolve())):
                    raise RuntimeError("unsafe zip path")
                z.extract(info, root)
    return root


def parse_pages(root: Path) -> dict[str, list[tuple[str, Path]]]:
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    pat = re.compile(r"^(\d+)-\d+-IMG_MAX[_-]?(\d+)$", re.I)
    for p in sorted(root.rglob("*.jpg")):
        m = pat.match(p.stem)
        if m:
            out[m.group(1)].append((m.group(2), p))
    out = {w: sorted(v) for w, v in out.items() if len({p for p, _ in v}) >= 3}
    if len(out) < 40:
        raise RuntimeError(f"too few writers: {len(out)}")
    return out


def split_writers(writers: list[str]) -> dict[str, list[str]]:
    rng = random.Random(SEED)
    writers = sorted(writers)
    rng.shuffle(writers)
    chosen = writers[:40]
    return {"train": sorted(chosen[:24]), "val": sorted(chosen[24:32]), "test": sorted(chosen[32:40])}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return rgb


def ink_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thr = threshold_sauvola(gray, window_size=31, k=0.2)
    ink = gray < thr
    return gray, ink.astype(np.uint8)


def crop32(arr: np.ndarray, cx: float, cy: float, fill: int = 255) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0 = int(round(cx)) - 16, int(round(cy)) - 16
    x1, y1 = x0 + 32, y0 + 32
    if arr.ndim == 3:
        out = np.full((32, 32, arr.shape[2]), fill, arr.dtype)
    else:
        out = np.full((32, 32), fill, arr.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = arr[sy0:sy1, sx0:sx1]
    return out


def page_patches(path: Path, max_patches: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = read_rgb(path)
    gray, ink = ink_mask(rgb)
    sift = cv2.SIFT_create(nfeatures=max_patches * 8, contrastThreshold=0.01, edgeThreshold=12)
    kps, desc = sift.detectAndCompute(gray, (ink * 255).astype(np.uint8))
    items: list[tuple[float, Any, np.ndarray]] = []
    if desc is not None:
        for kp, d in zip(kps, desc):
            items.append((float(kp.response), kp, d.astype(np.float32)))
    items.sort(key=lambda z: (-z[0], z[1].pt[1], z[1].pt[0]))
    patches, targets, descriptors = [], [], []
    for _, kp, d in items:
        p = crop32(rgb, kp.pt[0], kp.pt[1], 255)
        m = crop32(ink, kp.pt[0], kp.pt[1], 0)
        occ = float(m.mean())
        if 0.025 <= occ <= 0.60:
            patches.append(p)
            targets.append(m)
            descriptors.append(d)
        if len(patches) >= max_patches:
            break
    if len(patches) < max_patches // 2:
        ys, xs = np.where(ink > 0)
        if len(xs):
            order = np.linspace(0, len(xs) - 1, max_patches * 2).astype(int)
            for i in order:
                p = crop32(rgb, xs[i], ys[i], 255)
                m = crop32(ink, xs[i], ys[i], 0)
                occ = float(m.mean())
                if 0.025 <= occ <= 0.60:
                    patches.append(p)
                    targets.append(m)
                    descriptors.append(np.zeros(128, np.float32))
                if len(patches) >= max_patches:
                    break
    if not patches:
        raise RuntimeError(f"no patches for {path}")
    return np.stack(patches), np.stack(targets), np.stack(descriptors)


def nuisance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray, ink = ink_mask(rgb)
    h, w = gray.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = ink == 0
    acq = [math.log1p(w), math.log1p(h), w / max(h, 1)]
    for c in range(3):
        vals = lab[..., c][bg] if bg.any() else lab[..., c].ravel()
        acq += [float(vals.mean()/255), float(vals.std()/255)]
    for yy in range(4):
        for xx in range(4):
            y0, y1 = yy*h//4, (yy+1)*h//4
            x0, x1 = xx*w//4, (xx+1)*w//4
            acq.append(float(ink[y0:y1, x0:x1].mean()))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float) if n > 1 else np.array([0.])
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    widths = 2 * dist[ink > 0]
    if not len(widths): widths = np.array([0.])
    inkf = [float(ink.mean()), math.log1p(max(0, n-1)), float(np.mean(areas)/(h*w)), float(np.std(areas)/(h*w)),
            float(np.mean(widths)), float(np.std(widths))]
    inkf += np.quantile(widths, [0.1,0.25,0.5,0.75,0.9]).astype(float).tolist()
    return np.asarray(acq,np.float32), np.asarray(inkf,np.float32)


def to_tensor(patches: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(patches).permute(0,3,1,2).float().div_(255)
    return x.to(device, non_blocking=True)


def balanced_indices(labels: np.ndarray, classes_per_batch: int, m: int, rng: np.random.Generator) -> np.ndarray:
    by = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    cs = rng.choice(list(by), size=min(classes_per_batch, len(by)), replace=False)
    idx = [rng.choice(by[c], size=m, replace=len(by[c]) < m) for c in cs]
    return np.concatenate(idx)


def powernorm(x: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    x = np.sign(x) * np.abs(x) ** alpha
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def retrieval(x: np.ndarray, writers: list[str]) -> dict[str,float]:
    x = x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    sim = x @ x.T
    aps=[]; top1=top5=0
    for i,w in enumerate(writers):
        valid=np.arange(len(writers))!=i
        rel=np.array([ww==w for ww in writers]) & valid
        order=np.argsort(-sim[i],kind="stable"); order=order[valid[order]]
        hits=rel[order].astype(int); pos=np.flatnonzero(hits)+1
        if not len(pos): continue
        aps.append(float(np.mean(np.arange(1,len(pos)+1)/pos)))
        top1 += int(hits[:1].any()); top5 += int(hits[:5].any())
    n=max(len(aps),1)
    return {"map":float(np.mean(aps)) if aps else 0.,"top1":top1/n,"top5":top5/n,"eligible":len(aps)}


def fit_residual(train_x: np.ndarray, test_x: np.ndarray, train_n: np.ndarray, test_n: np.ndarray, alpha: float=10.) -> np.ndarray:
    nm=train_n.mean(0); ns=train_n.std(0); ns[ns<1e-6]=1
    ntr=(train_n-nm)/ns; nte=(test_n-nm)/ns
    xm=train_x.mean(0); xs=train_x.std(0); xs[xs<1e-6]=1
    xtr=(train_x-xm)/xs; xte=(test_x-xm)/xs
    ntr=np.column_stack([np.ones(len(ntr)),ntr]); nte=np.column_stack([np.ones(len(nte)),nte])
    reg=np.eye(ntr.shape[1],dtype=np.float32)*alpha; reg[0,0]=0
    beta=solve(ntr.T@ntr+reg,ntr.T@xtr,assume_a='pos')
    return xte-nte@beta


def permutation_p(x: np.ndarray, writers: list[str], observed: float, n: int=19) -> dict[str,float]:
    rng=np.random.default_rng(SEED+91); labels=np.array(writers,object); null=[]
    for _ in range(n):
        pl=labels.copy(); rng.shuffle(pl); null.append(retrieval(x,pl.tolist())["map"])
    p=(1+sum(v>=observed for v in null))/(n+1)
    return {"p":p,"n":n,"null_mean":float(np.mean(null)),"null_sd":float(np.std(null))}


def k_smoke(x: np.ndarray, writers: list[str]) -> dict[str,float]:
    rng=np.random.default_rng(SEED+177); by=defaultdict(list)
    for i,w in enumerate(writers): by[w].append(i)
    eligible=sorted(by); exact=within=total=0
    for true_k in range(2,min(8,len(eligible))+1):
        for _ in range(2):
            chosen=rng.choice(eligible,size=true_k,replace=False); idx=sum((by[w] for w in chosen),[]); xx=x[idx]
            best=(-1e9,None)
            for k in range(2,min(10,len(xx)-1)+1):
                lab=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(xx)
                score=silhouette_score(xx,lab)-0.01*k
                if score>best[0]: best=(score,k)
            exact += int(best[1]==true_k); within += int(abs(best[1]-true_k)<=1); total += 1
    return {"panels":total,"exact_rate":exact/max(total,1),"within_one_rate":within/max(total,1)}


def main() -> int:
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    MaskedAutoencoderViT, Wrapper = load_upstream()
    archive=WORK/'archives'/'historical_wi_color.zip'; download(ARCHIVE_URL,archive,ARCHIVE_MD5)
    pages=parse_pages(extract_archive(archive)); splits=split_writers(list(pages))
    manifest={"seed":SEED,"splits":splits,"upstream_commit":UPSTREAM_COMMIT,"archive_md5":ARCHIVE_MD5}
    (OUT/'writer_split.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    log('SPLIT',**{k:len(v) for k,v in splits.items()})

    patch_x=[]; patch_t=[]; patch_d=[]; patch_w=[]; page_records=[]
    for part, ws in splits.items():
        for w in ws:
            for pid,path in pages[w][:3]:
                px,pt,pd=page_patches(path,64)
                start=len(patch_x); patch_x.extend(px); patch_t.extend(pt); patch_d.extend(pd); patch_w.extend([w]*len(px))
                acq,inkf=nuisance(read_rgb(path))
                page_records.append({"part":part,"writer":w,"page":pid,"start":start,"end":len(patch_x),"acq":acq,"ink":inkf})
    patch_x=np.stack(patch_x); patch_t=np.stack(patch_t); patch_d=np.stack(patch_d); patch_w=np.array(patch_w)
    log('PATCH_AUDIT',patches=len(patch_x),pages=len(page_records),writers=len(set(patch_w)),bytes=int(patch_x.nbytes))

    device='cuda'; torch.manual_seed(SEED); np.random.seed(SEED); rng=np.random.default_rng(SEED)
    model=MaskedAutoencoderViT(img_size=32,patch_size=4,embed_dim=512,hog_pool=4,hog_bins=9,depth=8,decoder_depth=1,in_chans=3,global_pool=False,norm_pix_loss=False,target_in_chans=1).to(device)
    train_idx=np.flatnonzero(np.isin(patch_w,splits['train']))
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.05)
    model.train(); losses=[]
    for step in range(160):
        ii=rng.choice(train_idx,size=128,replace=len(train_idx)<128)
        x=to_tensor(patch_x[ii],device); target=torch.from_numpy(patch_t[ii]).float().unsqueeze(1).to(device)
        opt.zero_grad(set_to_none=True); loss,_,_=model(x,hog_target_imgs=target,mask_ratio=.75)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite MAE loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.02); opt.step(); losses.append(float(loss.detach()))
        if step%40==39: log('MAE_TRAIN',step=step+1,loss=float(np.mean(losses[-40:])))

    train_desc=patch_d[train_idx]; valid=np.linalg.norm(train_desc,axis=1)>0
    pseudo=MiniBatchKMeans(n_clusters=128,random_state=SEED,batch_size=4096,n_init=3,max_iter=200).fit_predict(train_desc[valid])
    ft_idx=train_idx[valid]; pseudo=np.asarray(pseudo)
    args={'model_options':{'in_dim':-1},'netvlad':{'num_clusters':100,'random':True},'netvlad_pooling':False}
    wrapped=Wrapper(model,args).to(device)
    with torch.inference_mode():
        sample=ft_idx[:min(10000,len(ft_idx))]; feats=[]
        for s in range(0,len(sample),256): feats.append(wrapped.forward_features(to_tensor(patch_x[sample[s:s+256]],device)).float().cpu().numpy())
    centers=MiniBatchKMeans(n_clusters=100,random_state=SEED,batch_size=4096,n_init=3,max_iter=200).fit(np.concatenate(feats)).cluster_centers_
    wrapped.nv._init_params(torch.from_numpy(centers).float().to(device))
    from pytorch_metric_learning.losses import MultiSimilarityLoss
    from pytorch_metric_learning.miners import MultiSimilarityMiner
    lossfn=MultiSimilarityLoss(alpha=2,beta=40,base=0.2); miner=MultiSimilarityMiner(epsilon=0.1)
    opt=torch.optim.AdamW(wrapped.parameters(),lr=1e-4,weight_decay=.01)
    wrapped.train(); ftl=[]
    for step in range(120):
        loc=balanced_indices(pseudo,classes_per_batch=16,m=4,rng=rng); ii=ft_idx[loc]; yy=torch.from_numpy(pseudo[loc]).long().to(device)
        x=to_tensor(patch_x[ii],device); opt.zero_grad(set_to_none=True); emb=wrapped(x); hard=miner(emb,yy); loss=lossfn(emb,yy,hard)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite metric loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.0); opt.step(); ftl.append(float(loss.detach()))
        if step%30==29: log('METRIC_TRAIN',step=step+1,loss=float(np.mean(ftl[-30:])))

    ckpt=OUT/'saghog_smoke.pt'; torch.save({'model_state_dict':wrapped.state_dict(),'manifest':manifest},ckpt)
    state=torch.load(ckpt,map_location='cpu'); wrapped.load_state_dict(state['model_state_dict']); wrapped.eval()
    page_vec=[]; page_writer=[]; page_part=[]; acqs=[]; inks=[]
    with torch.inference_mode():
        for r in page_records:
            local=[]
            for s in range(r['start'],r['end'],128): local.append(wrapped(to_tensor(patch_x[s:min(s+128,r['end'])],device)).float().cpu().numpy())
            v=powernorm(np.sum(np.concatenate(local),axis=0,keepdims=True))[0]
            page_vec.append(v); page_writer.append(r['writer']); page_part.append(r['part']); acqs.append(r['acq']); inks.append(r['ink'])
    page_vec=np.stack(page_vec); acqs=np.stack(acqs); inks=np.stack(inks); combined=np.concatenate([acqs,inks],axis=1)
    tr=np.array([p=='train' for p in page_part]); va=np.array([p=='val' for p in page_part]); te=np.array([p=='test' for p in page_part])
    pca=PCA(n_components=64,whiten=True,random_state=SEED).fit(page_vec[tr]); ztr=pca.transform(page_vec[tr]); zva=pca.transform(page_vec[va]); zte=pca.transform(page_vec[te])
    candidates={'raw':zte,'resid_acquisition':fit_residual(ztr,zte,acqs[tr],acqs[te]),'resid_ink':fit_residual(ztr,zte,inks[tr],inks[te]),'resid_combined':fit_residual(ztr,zte,combined[tr],combined[te])}
    val_candidates={'raw':zva,'resid_acquisition':fit_residual(ztr,zva,acqs[tr],acqs[va]),'resid_ink':fit_residual(ztr,zva,inks[tr],inks[va]),'resid_combined':fit_residual(ztr,zva,combined[tr],combined[va])}
    vw=[page_writer[i] for i in np.flatnonzero(va)]; tw=[page_writer[i] for i in np.flatnonzero(te)]
    val_metrics={k:retrieval(v,vw) for k,v in val_candidates.items()}; selected=max(val_metrics,key=lambda k:val_metrics[k]['map'])
    test_metrics={k:retrieval(v,tw) for k,v in candidates.items()}; observed=test_metrics[selected]['map']; perm=permutation_p(candidates[selected],tw,observed); kcal=k_smoke(candidates[selected],tw)
    nuisance_metrics={'acquisition':retrieval(acqs[te],tw),'ink':retrieval(inks[te],tw),'combined':retrieval(combined[te],tw)}
    np.savez_compressed(OUT/'exact_features.npz',train=ztr,val=zva,test=zte,test_selected=candidates[selected],val_writers=np.array(vw),test_writers=np.array(tw),acquisition_test=acqs[te],ink_test=inks[te])
    result={'schema':'blind-pal-saghog-v1.4-smoke','seed':SEED,'selected_by_validation':selected,'validation_metrics':val_metrics,'test_metrics':test_metrics,'nuisance_metrics':nuisance_metrics,'permutation':perm,'k_smoke':kcal,'mae_loss_last40':float(np.mean(losses[-40:])),'metric_loss_last30':float(np.mean(ftl[-30:])),'counts':{'patches':len(patch_x),'pages':len(page_records)},'davis_labels_loaded':False,'voynich_opened':False,'files':{}}
    for p in sorted(OUT.iterdir()):
        if p.is_file(): result['files'][p.name]={'bytes':p.stat().st_size,'sha256':digest(p)}
    (OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    result['files']['result.json']={'bytes':(OUT/'result.json').stat().st_size,'sha256':digest(OUT/'result.json')}
    print('SAGHOG_V14_SMOKE_RESULT '+json.dumps(result,sort_keys=True),flush=True)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for name in ['result.json','writer_split.json','exact_features.npz']:
            tar.add(OUT/name,arcname=name)
    enc=base64.b64encode(buf.getvalue()).decode(); sha=hashlib.sha256(buf.getvalue()).hexdigest()
    log('SAGHOG_BUNDLE_BEGIN',bytes=len(buf.getvalue()),sha256=sha,chunks=(len(enc)+2999)//3000)
    for i in range(0,len(enc),3000): print(f'SAGHOG_BUNDLE_CHUNK {i//3000:05d} {enc[i:i+3000]}',flush=True)
    print('SAGHOG_BUNDLE_END '+sha,flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
