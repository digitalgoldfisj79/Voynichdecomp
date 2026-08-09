#!/usr/bin/env python3
import os,json,math,hashlib,urllib.request
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
import torch

# Reuse frozen image-segmentation and BnF/language primitives without invoking their mains.
U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7b97e064c1098d63158a9a406780999aca91103d/experiments/bnf_m19_image_bridge_v1_2/run_arm_b.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
seg={'__name__':'v15seg'};exec(compile(src,'run_arm_b.py','exec'),seg)
b=seg['b']; K=19; LAM=.12; DREAL=768
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DTYPE=torch.float32
print('DEVICE',DEVICE,flush=True)

# ---------- sealed dense image loader ----------
def load_dense():
    sel=[];folios=set();p=os.path.join(b['DATA'],'corpus_crop_manifest.jsonl')
    with open(p) as h:
        for rowi,line in enumerate(h):
            r=json.loads(line)
            if r.get('kind')=='ccmerge' and r.get('view')=='norm' and not r.get('low_conf',False):
                # Only image/provenance fields survive this projection.
                sel.append((rowi,r['id'],r['folio'],int(r['word_index']),int(r['slot']),int(r['n_slots'])));folios.add(r['folio'])
    idx=np.array([q[0] for q in sel],np.int64);z=np.load(os.path.join(b['DATA'],'corpus_embeddings_full_dense.npz'),allow_pickle=False);ids=z['ids'];checks=np.linspace(0,len(sel)-1,min(1000,len(sel)),dtype=int)
    for j in checks:
        q=sel[j]
        if ids[q[0]]!=q[1]+'::norm':raise RuntimeError(('dense order',j))
    X=np.asarray(z['vectors'][idx],np.float32);del ids,z;X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    rec={'folio':np.array([q[2] for q in sel],dtype=object),'word':np.array([q[3] for q in sel],np.int32),'slot':np.array([q[4] for q in sel],np.int16),'nslots':np.array([q[5] for q in sel],np.int16)}
    folios=sorted(folios,key=lambda f:hashlib.sha256(('M19IMAGEv12split::'+f).encode()).digest());nt=round(.5*len(folios));nh=round(.2*len(folios));T=folios[:nt];H=folios[nt:nt+nh];C=folios[nt+nh:]
    tv=sorted(T,key=lambda f:hashlib.sha256(('M19IMAGEv12vis::'+f).encode()).digest());cut=round(.8*len(tv));split={'T':set(T),'H':set(H),'C':set(C),'Tf':set(tv[:cut]),'Tv':set(tv[cut:])}
    print('IMAGE_CENSUS',json.dumps({'components':len(sel),'folios':len(folios),'T':len(T),'H':len(H),'C':len(C),'Tfit':len(split['Tf']),'Tvis':len(split['Tv'])},separators=(',',':')),flush=True)
    return X,rec,split

def word_indices(rec,F):return seg['word_index_lists'](rec,F)
def fit_boundary(X,rec,F):return seg['fit_segmental'](X,word_indices(rec,F),K,LAM,408)
def segment_panel(X,rec,F,cent):return seg['segment_words'](X,word_indices(rec,F),cent,LAM)
def segment_vectors(segwords):
    words=[]
    for key,inds,ss in segwords:
        if ss:words.append(np.stack([x[2] for x in ss]).astype(np.float32))
    return words

def flatten(words):return np.concatenate(words,0) if words else np.zeros((0,DREAL),np.float32)

def stable_sample(X,n,tag):
    if len(X)<=n:return X
    rng=np.random.default_rng(b['seed']('v15sample',tag));return X[np.sort(rng.choice(len(X),n,replace=False))]

# ---------- image-only sigma calibration ----------
def gaussian_mix_ll(X,cent,w,sigma):
    # Exact spherical Gaussian mixture log-likelihood per observation.
    d=X.shape[1];x2=(X*X).sum(1)[:,None];m2=(cent*cent).sum(1)[None,:];dist=np.maximum(0,x2+m2-2*X@cent.T)
    const=-.5*d*math.log(2*math.pi*sigma*sigma); return float(np.mean(logsumexp(np.log(np.maximum(w,1e-15))[None,:]+const-dist/(2*sigma*sigma),axis=1)))
def calibrate_sigma(X,rec,split):
    c=fit_boundary(X,rec,split['Tf']);wf=segment_vectors(segment_panel(X,rec,split['Tf'],c));wv=segment_vectors(segment_panel(X,rec,split['Tv'],c));F=stable_sample(flatten(wf),80000,'sigfit');V=stable_sample(flatten(wv),30000,'sigvis')
    km=MiniBatchKMeans(n_clusters=K,random_state=408,batch_size=4096,n_init=5,max_iter=220,reassignment_ratio=.003).fit(F);lab=km.labels_;cent=km.cluster_centers_.astype(np.float32);res=((F-cent[lab])**2).sum(1);sigma0=float(math.sqrt(float(np.mean(res))/F.shape[1]));cnt=np.bincount(lab,minlength=K).astype(float);w=cnt/cnt.sum();rows=[]
    for scale in [.75,1.,1.5,2.,3.]:
        sig=sigma0*scale;ll=gaussian_mix_ll(V,cent,w,sig);rows.append({'scale':scale,'sigma':sig,'tvis_mix_ll':ll});print('SIGMA',json.dumps(rows[-1],separators=(',',':')),flush=True)
    mx=max(r['tvis_mix_ll'] for r in rows);near=[r for r in rows if mx-r['tvis_mix_ll']<=.001];near.sort(key=lambda r:-r['sigma']);ch=near[0];print('SIGMA_CHOICE',json.dumps({'sigma0':sigma0,**ch},separators=(',',':')),flush=True);return ch['sigma'],rows

# ---------- HMM utilities ----------
def probs(comp):
    lt,ls,le=comp;return np.asarray(lt,np.float32),np.asarray(ls,np.float32),np.asarray(le,np.float32)
def np_hard_labels(words,cent):
    out=[]
    for W in words:
        dist=(W*W).sum(1)[:,None]+(cent*cent).sum(1)[None,:]-2*W@cent.T;out.append(list(np.argmin(dist,axis=1).astype(int)))
    return out

def group_tensors(words,device=DEVICE):
    d=defaultdict(list)
    for W in words:d[len(W)].append(W)
    return {L:torch.tensor(np.stack(v),dtype=DTYPE,device=device) for L,v in d.items() if L>0}
def torch_comp(comp,device=DEVICE):
    lt,ls,le=probs(comp);return (torch.tensor(lt,device=device),torch.tensor(ls,device=device),torch.tensor(le,device=device))

def emit_log(X,mu,sigma,include_const=True):
    d=X.shape[-1];x2=(X*X).sum(-1,keepdim=True);m2=(mu*mu).sum(-1).view(*([1]*(X.ndim-1)),-1);cross=torch.matmul(X,mu.t());z=-(torch.clamp(x2+m2-2*cross,min=0))/(2*sigma*sigma)
    if include_const:z=z+(-.5*d*math.log(2*math.pi*sigma*sigma))
    return z

def hmm_fb_group(X,mu,sigma,tc,need_gamma):
    lt,ls,le=tc;E=emit_log(X,mu,sigma,True);B,L,V=E.shape;alph=[];a=ls[None,:]+E[:,0,:];alph.append(a)
    for t in range(1,L):a=E[:,t,:]+torch.logsumexp(a[:,:,None]+lt[None,:,:],dim=1);alph.append(a)
    ll=torch.logsumexp(a+le[None,:],dim=1)
    if not need_gamma:return ll,None
    beta=le[None,:].expand(B,V);gam=[None]*L;gam[L-1]=torch.softmax(alph[L-1]+beta,dim=1)
    for t in range(L-2,-1,-1):
        beta=torch.logsumexp(lt[None,:,:]+E[:,t+1,:][:,None,:]+beta[:,None,:],dim=2);gam[t]=torch.softmax(alph[t]+beta,dim=1)
    return ll,gam

def length_ll_for_L(L,tc):
    lt,ls,le=tc;a=ls
    for _ in range(1,L):a=torch.logsumexp(a[:,None]+lt,dim=0)
    return torch.logsumexp(a+le,dim=0)

def score_groups(groups,mu,sigma,comp):
    tc=torch_comp(comp);tot=0.;n=0;llen=0.
    with torch.no_grad():
        for L,X in groups.items():
            ll,_=hmm_fb_group(X,mu,sigma,tc,False);tot+=float(ll.sum().cpu());n+=X.shape[0]*L;llen+=float(length_ll_for_L(L,tc).cpu())*X.shape[0]
    const=-.5*next(iter(groups.values())).shape[-1]*math.log(2*math.pi*sigma*sigma);joint=tot/max(1,n);length=llen/max(1,n);gain=joint-length-const;return {'joint':joint,'length_only':length,'visual_gain':gain,'segments':n}

def em_fit(groups,mu0,sigma,comp,iters=8):
    mu=torch.tensor(mu0,dtype=DTYPE,device=DEVICE);tc=torch_comp(comp);counts=None;trainll=None
    for it in range(iters):
        sums=torch.zeros_like(mu);counts=torch.zeros(K,dtype=DTYPE,device=DEVICE);tot=0.;n=0
        with torch.no_grad():
            for L,X in groups.items():
                ll,gam=hmm_fb_group(X,mu,sigma,tc,True);tot+=float(ll.sum().cpu());n+=X.shape[0]*L
                for t,g in enumerate(gam):sums+=g.t()@X[:,t,:];counts+=g.sum(0)
            mu=sums/torch.clamp(counts[:,None],min=1e-6);trainll=tot/max(1,n)
        print('EM_ITER',it+1,round(trainll,6),flush=True)
    return mu,trainll,counts

def init_means(words,comp,tag,rs):
    X=flatten(words);sample=stable_sample(X,80000,('init',tag,rs));km=MiniBatchKMeans(n_clusters=K,random_state=rs,batch_size=4096,n_init=4,max_iter=200,reassignment_ratio=.003).fit(sample);cent=km.cluster_centers_.astype(np.float32);labs=np_hard_labels(words,cent);S=b['sym_stats'](labs,K);_,m=b['optimize'](S,comp,K,('v15init',tag,rs),6000,3);mu=np.zeros_like(cent)
    for c,v in enumerate(m):mu[int(v)]=cent[c]
    return mu

def fit_language(words,groups,sigma,comp,tag):
    fits=[]
    for rs in [408,409]:
        mu0=init_means(words,comp,(tag,rs),rs);mu,ll,cnt=em_fit(groups,mu0,sigma,comp,8);fits.append((mu,ll,cnt))
    best=0 if fits[0][1]>=fits[1][1] else 1;mu,cnt=fits[best][0],fits[best][2];m1=fits[0][0];m2=fits[1][0];cos=torch.sum(m1*m2,1)/(torch.linalg.norm(m1,dim=1)*torch.linalg.norm(m2,dim=1)+1e-12);w=(fits[0][2]+fits[1][2])/2;agr=float((cos*w).sum().cpu()/torch.clamp(w.sum(),min=1));return mu,fits[best][1],cnt,agr

# ---------- synthetic continuous controls ----------
def split_plain(pool,tag):
    span=b['choose_span'](pool,84000,tag);return b['split_text_letters'](span,45000)
def synth_words(text,lang,D,sigma_real,part):
    rngp=np.random.default_rng(b['seed']('v15proto',lang));P=rngp.normal(size=(K,D)).astype(np.float32);P/=np.linalg.norm(P,axis=1,keepdims=True);sig=sigma_real*math.sqrt(DREAL/D);rv=np.random.default_rng(b['seed']('v15vals',lang,part));rn=np.random.default_rng(b['seed']('v15noise',lang,part));out=[]
    for w in text.split():
        V=[]
        for c in w:
            vi=b['V2I'][int(rv.choice(b['LETTER_VALS'][b['A2I'][c]]))];x=P[vi]+rn.normal(0,sig,size=D).astype(np.float32);x/=max(np.linalg.norm(x),1e-12);V.append(x)
        if V:out.append(np.stack(V))
    return out,P,sig

def mean_recovery(mu,P):
    m=mu.detach().cpu().numpy();c=np.sum(m*P,1)/(np.linalg.norm(m,axis=1)*np.linalg.norm(P,axis=1)+1e-12);return float(np.mean(c))
def qualify(lms,pools,comps,sigma_real):
    rows=[]
    for la in b['QUAL']:
        trtxt,hotxt=split_plain(pools[la],('v15qual',la));tw,P,sig=synth_words(trtxt,la,64,sigma_real,'tr');hw,P2,sig2=synth_words(hotxt,la,64,sigma_real,'ho');assert np.allclose(P,P2) and abs(sig-sig2)<1e-12;tg=group_tensors(tw);hg=group_tensors(hw);rank=[];correct=None
        for cand in b['LANGS']:
            mu,trll,cnt,agr=fit_language(tw,tg,sig,comps[cand],('qual',la,cand));sc=score_groups(hg,mu,sig,comps[cand]);rank.append((cand,sc,agr,mu,cnt))
            if cand==la:correct=(mu,agr,cnt)
        rank.sort(key=lambda z:z[1]['visual_gain'],reverse=True);top=rank[0];margin=top[1]['visual_gain']-rank[1][1]['visual_gain'];rec=mean_recovery(correct[0],P);r={'lang':la,'top':top[0],'margin':margin,'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'mean_recovery':rec,'agreement':correct[1],'ranking':[(x[0],x[1]['visual_gain']) for x in rank]};rows.append(r);print('QUAL',json.dumps(r,separators=(',',':')),flush=True)
        del tg,hg
        if DEVICE=='cuda':torch.cuda.empty_cache()
    gate={'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'median_recovery':float(np.median([r['mean_recovery'] for r in rows])),'min_recovery':min(r['mean_recovery'] for r in rows),'min_agreement':min(r['agreement'] for r in rows)};gate['pass']=gate['correct']==6 and gate['min_margin']>=.03 and gate['median_recovery']>=.90 and gate['min_recovery']>=.80 and gate['min_agreement']>=.90;print('QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);return rows,gate

# ---------- main ----------
def c_buckets(C):
    out=[set() for _ in range(4)]
    for f in C:out[hashlib.sha256(('M19IMAGEv12bucket::'+f).encode()).digest()[0]%4].add(f)
    return out

def score_all(groups,fitted,sigma,comps):
    rows=[]
    for la,(mu,agr,cnt) in fitted.items():
        sc=score_groups(groups,mu,sigma,comps[la]);rows.append({'lang':la,**sc,'agreement':agr})
    return rows

def main():
    lms,pools,lmmeta=b['load_lms']();comps={la:b['induced'](lms[la]) for la in b['LANGS']};X,rec,split=load_dense();sigma,sigrows=calibrate_sigma(X,rec,split)
    qrows,qgate=qualify(lms,pools,comps,sigma);out={'protocol':'v1.5','sigma':sigma,'sigma_rows':sigrows,'qualification':qrows,'qualification_gate':qgate,'lm_meta':lmmeta}
    if not qgate['pass']:
        out['verdict']='CONTINUOUS IMAGE INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    # Frozen boundary model on all T12; no hard labels enter continuous model.
    bc=fit_boundary(X,rec,split['T']);Tw=segment_vectors(segment_panel(X,rec,split['T'],bc));Hw=segment_vectors(segment_panel(X,rec,split['H'],bc));Tg=group_tensors(Tw);Hg=group_tensors(Hw);print('STREAM',json.dumps({'T_words':len(Tw),'T_segments':sum(map(len,Tw)),'H_words':len(Hw),'H_segments':sum(map(len,Hw)),'sigma':sigma},separators=(',',':')),flush=True)
    fitted={};hrows=[]
    for la in b['LANGS']:
        mu,trll,cnt,agr=fit_language(Tw,Tg,sigma,comps[la],('VMS',la));sc=score_groups(Hg,mu,sigma,comps[la]);r={'lang':la,'train_joint':trll,**sc,'agreement':agr,'min_state_eff':float(cnt.min().cpu())};hrows.append(r);fitted[la]=(mu,agr,cnt);print('H12_LANG',json.dumps(r,separators=(',',':')),flush=True)
    rg=sorted(hrows,key=lambda r:r['visual_gain'],reverse=True);rj=sorted(hrows,key=lambda r:r['joint'],reverse=True);top=rg[0];gm=rg[0]['visual_gain']-rg[1]['visual_gain'];jm=rj[0]['joint']-rj[1]['joint'];primary=(rg[0]['lang']==rj[0]['lang'] and gm>=.03 and jm>=.03 and top['agreement']>=.90 and top['min_state_eff']>=100)
    signal={'gain_top':rg[0]['lang'],'gain_second':rg[1]['lang'],'gain_margin':gm,'joint_top':rj[0]['lang'],'joint_second':rj[1]['lang'],'joint_margin':jm,'agreement':top['agreement'],'min_state_eff':top['min_state_eff'],'primary':primary};out['H12']=hrows;out['signal']=signal;print('H12_SIGNAL',json.dumps(signal,separators=(',',':')),flush=True)
    if not primary:
        out['verdict']='NO CONTINUOUS IMAGE-M19 SIGNAL';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    # C12 unlocked only here. Winning language's T12 means remain fixed; all language scores use their T12-fitted means for fair rank comparison.
    Cw=segment_vectors(segment_panel(X,rec,split['C'],bc));Cg=group_tensors(Cw);crows=score_all(Cg,fitted,sigma,comps);cg=sorted(crows,key=lambda r:r['visual_gain'],reverse=True);cj=sorted(crows,key=lambda r:r['joint'],reverse=True);cand=top['lang'];gmargin=cg[0]['visual_gain']-cg[1]['visual_gain'] if cg[0]['lang']==cand else None;jmargin=cj[0]['joint']-cj[1]['joint'] if cj[0]['lang']==cand else None;buckets=[]
    for bi,B in enumerate(c_buckets(split['C'])):
        Bw=segment_vectors(segment_panel(X,rec,B,bc));Bg=group_tensors(Bw);rr=score_all(Bg,fitted,sigma,comps);rr.sort(key=lambda r:r['visual_gain'],reverse=True);cs=next(r['visual_gain'] for r in rr if r['lang']==cand);bo=max(r['visual_gain'] for r in rr if r['lang']!=cand);buckets.append({'bucket':bi,'folios':len(B),'segments':sum(map(len,Bw)),'candidate_margin':cs-bo,'gain_ranking':[(r['lang'],r['visual_gain']) for r in rr]})
    confirmed=(cg[0]['lang']==cand and cj[0]['lang']==cand and gmargin is not None and jmargin is not None and gmargin>=.03 and jmargin>=.03 and all(x['candidate_margin']>0 for x in buckets));out['C12']={'gain_ranking':[(r['lang'],r['visual_gain']) for r in cg],'joint_ranking':[(r['lang'],r['joint']) for r in cj],'candidate':cand,'gain_margin':gmargin,'joint_margin':jmargin,'buckets':buckets,'confirmed':confirmed};print('C12',json.dumps(out['C12'],separators=(',',':')),flush=True);out['verdict']=('CONFIRMED CONTINUOUS IMAGE-M19 SIGNAL '+cand) if confirmed else 'H12 CONTINUOUS IMAGE-M19 CANDIDATE / C12 FAILED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
