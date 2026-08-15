from __future__ import annotations
import argparse, hashlib, json, math, os, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

warnings.filterwarnings('ignore')
SEED = 20260814
OUT = 224
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
EXPECTED_ENCODER_SHA256 = '54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0'
WORD_KEYSET_SHA256 = 'c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84'
CALIB_FOLIOS = ['f10r', 'f10v', 'f11r', 'f11v', 'f13r', 'f13v', 'f14r', 'f14v', 'f15r', 'f15v', 'f16r', 'f16v', 'f17r', 'f17v', 'f18r', 'f18v', 'f19r', 'f19v', 'f1v', 'f20r', 'f20v', 'f21r', 'f21v', 'f22r', 'f22v', 'f23r', 'f23v', 'f24r', 'f24v', 'f25r', 'f25v', 'f26r', 'f26v', 'f27r', 'f27v', 'f28r', 'f28v', 'f29r', 'f29v', 'f2r', 'f2v', 'f30r', 'f30v', 'f31r', 'f31v', 'f32r', 'f32v', 'f33r', 'f33v', 'f34r', 'f34v', 'f35r', 'f35v', 'f36r', 'f36v', 'f37r', 'f37v', 'f38r', 'f38v', 'f39r', 'f39v', 'f3r', 'f3v', 'f40r', 'f40v', 'f41r', 'f42r', 'f42v', 'f43r', 'f43v', 'f44r', 'f44v', 'f45r', 'f45v', 'f46r', 'f46v', 'f47r', 'f47v', 'f48r', 'f48v', 'f49r', 'f49v', 'f4r', 'f4v', 'f50r', 'f50v', 'f51r', 'f51v', 'f52r', 'f52v', 'f53r', 'f53v', 'f54r', 'f54v', 'f55r', 'f55v', 'f56r', 'f56v', 'f5r', 'f6r', 'f6v', 'f7r', 'f7v', 'f8r', 'f8v', 'f9r', 'f9v']
EXPECTED_PAIR_SKELETON_SHA256 = '7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4'
EXPECTED = {'word_rows':9620,'word_folios':107,'pair_rows':485,'pair_folios':103,'pair_bifolia':28,'mid_valid':331,'suffix_valid':327}
BASE_RATE = {'midfix':0.32340862422997946,'suffix':0.5314757481940144}
NULL_SOURCES = ['iid_null','page_only','hand_only','abstract_text_only','background_page_pc']
PHYSICAL_SOURCES = ['immediate_visual','line_reset_visual','broad_visual']
BETAS = [0.20,0.30,0.40,0.50,0.70]
CAT = ['family','hand','section']
CONT = ['same_line','prev_len','cur_len','prev_unit_len','cur_unit_len','rel_pos','gap_px','width_prev','width_cur','height_prev','height_cur','x_prev','x_cur','y_prev','y_cur']


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
    tmp.replace(path)


class WriterNet(nn.Module):
    def __init__(self,nwriters:int):
        super().__init__()
        base=resnet18(weights=None)
        base.fc=nn.Identity()
        self.base=base
        self.embed=nn.Linear(512,128)
        self.cls=nn.Linear(128,nwriters)
    def forward(self,x):
        h=self.base(x)
        e=F.normalize(self.embed(h),dim=1)
        return e,self.cls(e)


def image_to_tensor(path:Path)->torch.Tensor:
    g=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.0
    if g.ndim!=2 or min(g.shape)<2: raise ValueError(f'bad image {path}: {g.shape}')
    h,w=g.shape; dy=max(1,int(round(.05*h))); dx=max(1,int(round(.05*w)))
    border=np.concatenate([g[:dy,:].ravel(),g[-dy:,:].ravel(),g[:,:dx].ravel(),g[:,-dx:].ravel()])
    cy0,cy1=int(.2*h),max(int(.8*h),int(.2*h)+1); cx0,cx1=int(.2*w),max(int(.8*w),int(.2*w)+1)
    bm=float(border.mean()); cm=float(g[cy0:cy1,cx0:cx1].mean())
    ink=(1.0-g) if bm>=cm else g
    im=Image.fromarray(np.uint8(np.clip(ink,0,1)*255),mode='L').resize((OUT,OUT),Image.Resampling.BILINEAR)
    z=torch.from_numpy(np.asarray(im,dtype=np.float32)/255.0).unsqueeze(0)
    rgb=(1.0-z).repeat(3,1,1)
    return (rgb-MEAN)/STD


def discover_word_paths(root:Path, required_ids:set[str]):
    found=defaultdict(list); scanned=0
    for p in root.rglob('*_norm.png'):
        scanned+=1
        stem=p.stem
        if not stem.endswith('_norm'): continue
        i=stem[:-5]
        if i in required_ids: found[i].append(p)
    missing=sorted(required_ids-set(found))
    dup={k:[str(x) for x in v] for k,v in found.items() if len(v)!=1}
    if missing or dup:
        raise RuntimeError(f'crop path gate failed: missing={len(missing)} duplicate_ids={len(dup)} first_missing={missing[:10]} first_dups={list(dup.items())[:3]}')
    return {k:v[0] for k,v in found.items()},scanned


def embed_words(model, W:pd.DataFrame, id2path:dict[str,Path], device, batch_size:int=128):
    model.eval(); arr=[]; decode_fail=[]
    with torch.no_grad():
        for s in range(0,len(W),batch_size):
            chunk=W.iloc[s:s+batch_size]
            xs=[]
            for r in chunk.itertuples():
                try: xs.append(image_to_tensor(id2path[str(r.id)]))
                except Exception as e: decode_fail.append((str(r.id),repr(e))); xs.append(torch.zeros(3,OUT,OUT))
            if decode_fail: break
            X=torch.stack(xs).to(device)
            e,_=model(X); arr.append(e.detach().cpu().numpy().astype(np.float32))
            if (s//batch_size+1)%10==0: print('EMBED_PROGRESS',min(s+batch_size,len(W)),'/',len(W),flush=True)
    if decode_fail: raise RuntimeError(f'decode failures: {decode_fail[:5]}')
    X=np.concatenate(arr,axis=0)
    n=np.linalg.norm(X,axis=1)
    if not np.isfinite(X).all() or np.max(np.abs(n-1))>1e-4: raise RuntimeError('embedding finite/unit-norm gate failed')
    return X


def solve_intercept(eta,p0):
    lo,hi=-20.,20.
    for _ in range(60):
        mid=(lo+hi)/2
        if expit(mid+eta).mean()>p0: hi=mid
        else: lo=mid
    return (lo+hi)/2


def simulate_labels(df,source,beta,rng,p0):
    n=len(df)
    if source=='iid_null': eta=np.zeros(n)
    elif source=='page_only':
        vals={f:v for f,v in zip(sorted(df.folio.unique()),rng.normal(size=df.folio.nunique()))}; eta=np.array([vals[f] for f in df.folio])*0.9
    elif source=='hand_only':
        vals={h:v for h,v in zip(sorted(df.hand.astype(str).unique()),rng.normal(size=df.hand.astype(str).nunique()))}; eta=np.array([vals[str(h)] for h in df.hand])*0.9
    elif source=='abstract_text_only':
        z=(df.cur_len-df.cur_len.mean())/(df.cur_len.std()+1e-12); eta=0.55*z.to_numpy()+0.35*df.same_line.astype(float).to_numpy()+0.30*df.hidden_family.to_numpy()
    elif source=='background_page_pc': eta=0.9*df.hidden_page_pc1.to_numpy()
    elif source in ('immediate_visual','broad_visual'): eta=beta*df.hidden_local_visual.to_numpy()
    elif source=='line_reset_visual': eta=beta*df.hidden_local_visual.to_numpy()*df.same_line.astype(float).to_numpy()
    else: raise ValueError(source)
    b=solve_intercept(eta,p0); return rng.binomial(1,expit(b+eta),size=n).astype(int)


def build_design(df, visual_score):
    out={}
    for f in range(5):
        te=df.fold.to_numpy(int)==f; tr=~te
        pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=3),CAT),('cont',StandardScaler(),CONT)],remainder='drop')
        X0tr=pre.fit_transform(df.loc[tr,CAT+CONT]); X0te=pre.transform(df.loc[te,CAT+CONT])
        vs=np.asarray(visual_score[df.index.to_numpy()],float)
        m=vs[tr].mean(); s=vs[tr].std()+1e-12
        ztr=((vs[tr]-m)/s).reshape(-1,1); zte=((vs[te]-m)/s).reshape(-1,1)
        X1tr=hstack([X0tr,csr_matrix(ztr)],format='csr'); X1te=hstack([X0te,csr_matrix(zte)],format='csr')
        out[f]=(tr,te,X0tr,X0te,X1tr,X1te)
    return out


def eval_labels(df,design,y):
    gains=[]; coefs=[]; n_test=0
    for f in range(5):
        tr,te,X0tr,X0te,X1tr,X1te=design[f]
        if y[tr].min()==y[tr].max() or y[te].min()==y[te].max(): gains.append(float('-inf')); coefs.append(float('nan')); continue
        m0=LogisticRegression(C=1.0,solver='liblinear',max_iter=400).fit(X0tr,y[tr])
        m1=LogisticRegression(C=1.0,solver='liblinear',max_iter=400).fit(X1tr,y[tr])
        p0=m0.predict_proba(X0te)[:,1]; p1=m1.predict_proba(X1te)[:,1]
        gains.append(float(log_loss(y[te],p0,normalize=False)-log_loss(y[te],p1,normalize=False)))
        coefs.append(float(m1.coef_[0,-1])); n_test+=int(te.sum())
    total=sum(g for g in gains if np.isfinite(g)); cost=.5*math.log(max(n_test,2)); adj=total-cost
    return {'raw_total_gain_nats':total,'complexity_cost_nats':cost,'adjusted_total_gain_nats':adj,'adjusted_gain_nats_per_event':adj/max(n_test,1),'positive_folds':int(sum(g>0 for g in gains)),'fold_gains_nats':gains,'mean_visual_coef':float(np.nanmean(coefs)),'n':n_test}


def run_component(H, comp, visual_score, dev_reps, confirm_reps):
    valid='mid_valid' if comp=='midfix' else 'suffix_valid'; p0=BASE_RATE[comp]
    df=H[H[valid].astype(bool)].copy().reset_index(drop=True)
    orig=H[H[valid].astype(bool)].index.to_numpy(); df.index=orig
    design=build_design(df,visual_score)
    dev=[]
    for si,src in enumerate(NULL_SOURCES):
        for r in range(dev_reps):
            rng=np.random.default_rng(100000+10000*si+r+(0 if comp=='midfix' else 500000)); y=simulate_labels(df,src,0.0,rng,p0); z=eval_labels(df,design,y); z.update(source=src,beta=0.0,rep=r); dev.append(z)
    for si,src in enumerate(PHYSICAL_SOURCES):
        for bi,beta in enumerate(BETAS):
            for r in range(dev_reps):
                rng=np.random.default_rng(200000+100000*si+10000*bi+r+(0 if comp=='midfix' else 500000)); y=simulate_labels(df,src,beta,rng,p0); z=eval_labels(df,design,y); z.update(source=src,beta=beta,rep=r); dev.append(z)
    DEV=pd.DataFrame(dev)
    vals=np.sort(DEV[DEV.source.isin(NULL_SOURCES)].adjusted_gain_nats_per_event.unique()); candidates=np.r_[vals-1e-12,vals+1e-12]
    tau=None
    for t in np.sort(candidates):
        rates=[]
        for src in NULL_SOURCES:
            q=DEV[DEV.source==src]; rates.append(((q.adjusted_gain_nats_per_event>t)&(q.positive_folds>=4)).mean())
        if max(rates)<=.05+1e-12: tau=float(t); break
    if tau is None: tau=float(DEV.adjusted_gain_nats_per_event.max()+1e-9)
    conf=[]; specs=[(src,0.0) for src in NULL_SOURCES]+[(src,b) for src in PHYSICAL_SOURCES for b in BETAS]
    for si,(src,beta) in enumerate(specs):
        for rr in range(confirm_reps):
            rng=np.random.default_rng(900000+100000*si+rr+(0 if comp=='midfix' else 500000)); y=simulate_labels(df,src,beta,rng,p0); z=eval_labels(df,design,y); z.update(source=src,beta=beta,rep=rr); conf.append(z)
    CONF=pd.DataFrame(conf); CONF['call']=(CONF.adjusted_gain_nats_per_event>tau)&(CONF.positive_folds>=4)
    summary=[]
    for (src,b),g in CONF.groupby(['source','beta']):
        summary.append({'source':src,'beta':float(b),'n_reps':len(g),'call_rate':float(g.call.mean()),'positive_coef_rate':float((g.mean_visual_coef>0).mean()),'gain_mean':float(g.adjusted_gain_nats_per_event.mean()),'gain_sd':float(g.adjusted_gain_nats_per_event.std())})
    S=pd.DataFrame(summary)
    null_rates={src:float(S[(S.source==src)&(S.beta==0.0)].iloc[0].call_rate) for src in NULL_SOURCES}
    power50={src:float(S[(S.source==src)&(np.isclose(S.beta,0.50))].iloc[0].call_rate) for src in PHYSICAL_SOURCES}
    fpr_pass=all(v<=.05+1e-12 for v in null_rates.values()); power_pass=all(v>=.80-1e-12 for v in power50.values())
    return {'component':comp,'n_events':len(df),'base_rate':p0,'threshold_gain_nats_per_event':tau,'null_fpr':null_rates,'physical_power_beta_0_50':power50,'fpr_pass':fpr_pass,'power_pass':power_pass,'pass':bool(fpr_pass and power_pass),'summary':summary},DEV,CONF


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--encoder',type=Path,required=True); ap.add_argument('--pair-skeleton',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dev-reps',type=int,default=60); ap.add_argument('--confirm-reps',type=int,default=100); ap.add_argument('--result-put-url',default=os.getenv('RESULT_PUT_URL')); ap.add_argument('--result-key',default=os.getenv('RESULT_KEY'))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    audit={'schema':'frontier-u6-stageb-v0.2','target_opened':False,'true_retention_read':False,'visual_instrument':'U6-v0.2 writer-sensitive 128-D','page_subspace_subtraction':False,'dev_reps':a.dev_reps,'confirm_reps':a.confirm_reps}
    hashes={'encoder':sha256(a.encoder),'pair_skeleton':sha256(a.pair_skeleton)}; audit['input_sha256']=hashes
    if hashes['encoder']!=EXPECTED_ENCODER_SHA256 or hashes['pair_skeleton']!=EXPECTED_PAIR_SKELETON_SHA256: raise RuntimeError(f'input hash gate failed: {hashes}')
    H=pd.read_csv(a.pair_skeleton,dtype={'folio':str,'bifolium':str,'family':str,'hand':str,'section':str})
    manifest=a.data/'results/corpus_crop_manifest.jsonl'; rows=[]
    with manifest.open('r',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('kind')=='word' and r.get('view')=='norm' and str(r.get('folio')) in CALIB_FOLIOS:
                rows.append({'id':str(r['id']),'folio':str(r['folio']),'word_index':int(r['word_index'])})
    W=pd.DataFrame(rows).sort_values(['folio','word_index','id'],kind='stable').drop_duplicates(['folio','word_index'],keep='first').reset_index(drop=True)
    keytext=''.join(f'{f}|{int(i)}\n' for f,i in sorted(zip(W.folio,W.word_index)))
    keyhash=hashlib.sha256(keytext.encode()).hexdigest(); audit['word_keyset_sha256']=keyhash
    if keyhash!=WORD_KEYSET_SHA256: raise RuntimeError(f'word keyset gate failed: rows={len(W)} folios={W.folio.nunique()} hash={keyhash}')
    forbidden={'mid_retain','suffix_retain','exact_retain','prev_midfix','cur_midfix','prev_suffix','cur_suffix'}
    if forbidden & set(H.columns): raise RuntimeError('target firewall failed: true-retention field present')
    counts={'word_rows':len(W),'word_folios':W.folio.nunique(),'pair_rows':len(H),'pair_folios':H.folio.nunique(),'pair_bifolia':H.bifolium.nunique(),'mid_valid':int(H.mid_valid.sum()),'suffix_valid':int(H.suffix_valid.sum())}; audit['counts']=counts
    if counts!=EXPECTED: raise RuntimeError(f'population gate failed: {counts}')
    for c in CONT:
        H[c]=pd.to_numeric(H[c],errors='coerce'); H[c]=H[c].fillna(H[c].median())
    required=set(W.id.astype(str)); id2path,scanned=discover_word_paths(a.data,required); audit['norm_png_scanned']=scanned; audit['required_crop_paths']=len(id2path)
    ck=torch.load(a.encoder,map_location='cpu',weights_only=False); writers=ck['writers']; model=WriterNet(len(writers)); model.load_state_dict(ck['state_dict'],strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); audit['device']=str(device); audit['encoder_train_writers']=len(writers)
    Xw=embed_words(model,W,id2path,device)
    np.savez_compressed(a.out/'U6_STAGEB_WORD_EMBEDDINGS.npz',embeddings=Xw,ids=W.id.astype(str).to_numpy())
    key2row={(str(r.folio),int(r.word_index)):i for i,r in enumerate(W.itertuples())}
    ip=[];ic=[]
    for r in H.itertuples():
        x=key2row.get((str(r.folio),int(r.source_word_index_prev))); y=key2row.get((str(r.folio),int(r.source_word_index_cur)))
        if x is None or y is None: raise RuntimeError(f'pair join failure {r.folio} {r.source_word_index_prev} {r.source_word_index_cur}')
        ip.append(x);ic.append(y)
    H['emb_prev_idx']=ip; H['emb_cur_idx']=ic
    rawcos=(Xw[H.emb_prev_idx.to_numpy()]*Xw[H.emb_cur_idx.to_numpy()]).sum(1).astype(np.float64)
    H['raw_cos']=rawcos; fm=H.groupby('folio').raw_cos.transform('mean'); fs=H.groupby('folio').raw_cos.transform('std').replace(0,np.nan)
    H['hidden_local_visual']=((H.raw_cos-fm)/fs).fillna(0.0).clip(-4,4)
    cents=[]; fols=[]
    for fol,g in W.groupby('folio'):
        cents.append(Xw[g.index.to_numpy()].mean(0));fols.append(fol)
    C=np.stack(cents); Z=C-C.mean(0); _,_,Vt=np.linalg.svd(Z,full_matrices=False); pc1=Z@Vt[0]; pc1=(pc1-pc1.mean())/(pc1.std()+1e-12); pcmap=dict(zip(fols,pc1)); H['hidden_page_pc1']=H.folio.map(pcmap).astype(float)
    fams=sorted(H.family.unique()); vals=np.random.default_rng(SEED).normal(size=len(fams)); H['hidden_family']=H.family.map(dict(zip(fams,vals))).astype(float)
    audit['visual_score']={'n':len(rawcos),'mean':float(rawcos.mean()),'sd':float(rawcos.std()),'min':float(rawcos.min()),'max':float(rawcos.max())}
    results={}; allpass=True
    for comp in ('midfix','suffix'):
        print('CALIBRATING',comp,flush=True)
        res,dev,conf=run_component(H,comp,rawcos,a.dev_reps,a.confirm_reps); results[comp]=res; allpass &= res['pass']
        dev.to_csv(a.out/f'U6_STAGEB_DEV_{comp}.csv',index=False); conf.to_csv(a.out/f'U6_STAGEB_CONFIRM_{comp}.csv',index=False)
    verdict='PASS_VTPS_CALIBRATION' if allpass else 'FAIL_VTPS_CALIBRATION'
    out={'schema':'frontier-u6-stageb-v0.2','formal_verdict':verdict,'interpretation':'QUALIFIED_FOR_SEPARATE_TARGET_OPENING' if allpass else 'ABSTAIN_UNRESOLVED','target_opened':False,'true_retention_read':False,'target_may_open_later':bool(allpass),'audit':audit,'components':results}
    atomic_json(a.out/'U6_STAGEB_RESULT.json',out)
    md=['# U6-v0.2 Stage-B synthetic VTPS qualification','',f'Formal verdict: **{verdict}**','',f'- target opened: **NO**',f'- true retention labels read: **NO**',f'- page-subspace subtraction: **NO**',f'- calibration population: {counts}',f'- pair visual cosine mean/sd: {rawcos.mean():.4f} / {rawcos.std():.4f}']
    for comp in ('midfix','suffix'):
        r=results[comp]; md += ['',f'## {comp}',f'- threshold: {r["threshold_gain_nats_per_event"]:.8f}',f'- null FPRs: {r["null_fpr"]}',f'- physical power at beta=0.50: {r["physical_power_beta_0_50"]}',f'- gate: {"PASS" if r["pass"] else "FAIL"}']
    (a.out/'U6_STAGEB_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    if a.result_put_url and a.result_key:
        import requests
        payload=(a.out/'U6_STAGEB_RESULT.json').read_bytes(); rr=requests.post(a.result_put_url,headers={'apikey':a.result_key,'Authorization':'Bearer '+a.result_key,'Content-Type':'application/json','x-upsert':'true'},data=payload,timeout=120)
        print('RESULT_UPLOAD',rr.status_code,flush=True)
    print('STAGEB_FINAL',json.dumps({'formal_verdict':verdict,'components':{k:{'null_fpr':v['null_fpr'],'power50':v['physical_power_beta_0_50'],'pass':v['pass']} for k,v in results.items()}},sort_keys=True),flush=True)

if __name__=='__main__': main()
