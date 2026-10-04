#!/usr/bin/env python3
import collections,hashlib,json,math,os,re,urllib.request
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/420d4363973174760a78000f1049339dbd26fa46/research/hf_emergent_occupancy_fold.py"
ns={"__name__":"occupancy_base"}
exec(compile(urllib.request.urlopen(BASE,timeout=60).read().decode(),BASE,"exec"),ns)
rows,folds=ns["load_rows"]()

PARA_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/de1d843a04f27862036e3551be291083ed96a1af/research/data/paragraph_ranges_20261004.tsv"
ptxt=urllib.request.urlopen(PARA_URL,timeout=60).read().decode()
PR=collections.defaultdict(list)
for ln in ptxt.splitlines():
    if not ln.strip():continue
    f,p,a,b=ln.split("\t");PR[f].append((int(a),int(b),int(p)))

MERGES=['ch','dy','ai','ok','ol','che','sh','in','ee','aiin','ot','qok','ar','al','or','she','ain','daiin','eey','chedy','edy','ey','qot','eedy','ody','hy','ct','ck','chy','chey','chol','qo']
CLASSES=[
['o','a','qo','c','b','j','u'],
['ai','i'],
['aiin','dy','ain','daiin','eey','chedy','edy','ey','eedy','ody','hy','m','chy','chey','in','n'],
['ar','al','or','s','r','chol','g'],
['ch'],
['che','e','ee','h','q','z','v'],
['ct','ck'],
['d','x'],
['p','f'],
['k','qok','ok','ot','t','qot'],
['y','ol','l'],
['sh','she']]
ST={x:i for i,a in enumerate(CLASSES) for x in a};K=12
def segment(t):
    z=list(t)
    for mg in MERGES:
        out=[];i=0
        while i<len(z):
            if i+1<len(z) and z[i]+z[i+1]==mg:out.append(mg);i+=2
            else:out.append(z[i]);i+=1
        z=out
    if any(x not in ST for x in z):raise RuntimeError(("unmapped",t,z,[x for x in z if x not in ST]))
    return z
def para_id(f,line):
    for a,b,p in PR.get(f,()):
        if a<=line<=b:return p
    # rare P lines outside paragraph source: isolate rather than leak neighboring state
    return 10000+line
def house(t):
    t=(t or '').lower()
    if t.startswith(('ch','sh')):return 'H3'
    if t.startswith(('q','t')):return 'H1'
    if t.startswith(('d','o')):return 'H2'
    if t.startswith(('s','y')):return 'H4'
    return 'O'

# annotate chunks and K12 start classes
for r in rows:
    z=segment(r['token']);r['pieces']=z;r['start']=ST[z[0]];r['final_piece']=z[-1];r['fold']=folds[r['bifolium']]

page=collections.defaultdict(lambda:np.zeros(K,float));para=collections.defaultdict(lambda:np.zeros(K,float));linehist=collections.defaultdict(list)
LINES=collections.OrderedDict();missing_para=0
for r in rows:
    lk=(r['folio'],r['line']); pk=r['folio']; q=(pk,para_id(pk,r['line'])); y=r['start']
    rec=LINES.setdefault(lk,{'folio':pk,'line':r['line'],'bif':r['bifolium'],'fold':r['fold'],'opener_token':None,'events':[]})
    if r['pos']==0: rec['opener_token']=r['token']
    else:
        prev=linehist[lk][-1]
        rc=np.zeros(K,float)
        for c in [x[0] for x in linehist[lk][-6:]]:rc[c]+=1
        rec['events'].append({'y':y,'section':r['section'],'piece':prev[1],'page':page[pk].copy(),'para':para[q].copy(),'recent':rc})
    page[pk][y]+=1;para[q][y]+=1;linehist[lk].append((y,r['final_piece']))
LINES=[dict(v,house=house(v['opener_token'])) for v in LINES.values() if len(v['events'])>=3]
ALPHA=30.0

def fit_struct(train_lines):
    ev=[x for l in train_lines for x in l['events']]
    M=collections.defaultdict(collections.Counter);G=collections.Counter()
    for x in ev:M[(x['section'],x['piece'])][x['y']]+=1;G[x['y']]+=1
    ng=sum(G.values())
    def base(x):
        cc=M.get((x['section'],x['piece']),{});n=sum(cc.values())
        return np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(ng+.5*K)))/(n+ALPHA) for c in range(K)],float)
    P=np.stack([base(x) for x in ev]);Y=np.array([x['y'] for x in ev],int)
    Pg=np.stack([np.log1p(x['page']) for x in ev]);Pa=np.stack([np.log1p(x['para']) for x in ev]);Rc=np.stack([x['recent'] for x in ev])
    def fg(th):
        sc=np.log(np.maximum(P,1e-15))+Pg*th[:K]+Pa*th[K:2*K]+Rc*th[-1]
        mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        loss=-np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300)).sum()+.5*np.dot(th,th)
        D=Q;D[np.arange(len(Y)),Y]-=1
        gp=(D*Pg).sum(0)+th[:K];ga=(D*Pa).sum(0)+th[K:2*K];gr=float((D*Rc).sum())+th[-1]
        return float(loss),np.r_[gp,ga,gr]
    rr=minimize(lambda th:fg(th),np.zeros(2*K+1),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11})
    return M,G,rr.x

def attach(lines,model):
    M,G,th=model;ng=sum(G.values());out=[]
    for l in lines:
        P=[];Y=[]
        for x in l['events']:
            cc=M.get((x['section'],x['piece']),{});n=sum(cc.values())
            b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(ng+.5*K)))/(n+ALPHA) for c in range(K)],float)
            sc=np.log(np.maximum(b,1e-15))+np.log1p(x['page'])*th[:K]+np.log1p(x['para'])*th[K:2*K]+x['recent']*th[-1]
            sc-=sc.max();p=np.exp(sc);p/=p.sum();P.append(p);Y.append(x['y'])
        q=dict(l);q['P']=np.stack(P);q['Y']=np.array(Y,int);out.append(q)
    return out

def flatten(ls):
    P=np.vstack([l['P'] for l in ls]);Y=np.concatenate([l['Y'] for l in ls]);li=np.concatenate([np.full(len(l['Y']),i,int) for i,l in enumerate(ls)])
    return P,Y,li

def fit_mix(ls,S,l2,seed=20261004,maxem=60):
    P,Y,li=flatten(ls);L=len(ls);rng=np.random.default_rng(seed+S+int(l2*100));B=rng.normal(0,.02,(S,K));B-=B.mean(1,keepdims=True);pi=np.ones(S)/S;last=-1e99
    for it in range(maxem):
        lineLL=np.zeros((L,S))
        for z in range(S):
            sc=np.log(np.maximum(P,1e-15))+B[z];mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True);lp=np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))
            lineLL[:,z]=np.bincount(li,weights=lp,minlength=L)
        A=lineLL+np.log(pi+1e-15);mx=A.max(1,keepdims=True);W=np.exp(A-mx);W/=W.sum(1,keepdims=True);ll=float(np.sum(mx[:,0]+np.log(np.exp(A-mx).sum(1))))
        pi=(W.sum(0)+1)/(L+S)
        for z in range(S):
            ew=W[li,z]
            def fg(b):
                bb=b-b.mean();sc=np.log(np.maximum(P,1e-15))+bb;mm=sc.max(1,keepdims=True);Q=np.exp(sc-mm);Q/=Q.sum(1,keepdims=True)
                loss=-float(np.sum(ew*np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))))+.5*l2*np.dot(bb,bb)
                D=Q;D[np.arange(len(Y)),Y]-=1;g=(D*ew[:,None]).sum(0)+l2*bb;g-=g.mean()
                return loss,g
            rr=minimize(lambda b:fg(b),B[z],jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-9});B[z]=rr.x-rr.x.mean()
        if abs(ll-last)<1e-5:break
        last=ll
    # final posterior -> opener priors
    lineLL=np.zeros((L,S))
    for z in range(S):
        sc=np.log(np.maximum(P,1e-15))+B[z];mm=sc.max(1,keepdims=True);Q=np.exp(sc-mm);Q/=Q.sum(1,keepdims=True);lp=np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300));lineLL[:,z]=np.bincount(li,weights=lp,minlength=L)
    A=lineLL+np.log(pi+1e-15);mm=A.max(1,keepdims=True);W=np.exp(A-mm);W/=W.sum(1,keepdims=True)
    hp={}
    for h in ('H1','H2','H3','H4','O'):
        ix=[i for i,l in enumerate(ls) if l['house']==h];hp[h]=(W[ix].sum(0)+2*pi)/(len(ix)+2) if ix else pi.copy()
    return {'B':B,'pi':pi,'hp':hp,'iter':it+1,'ll':ll}

def score(ls,md,use_house=False):
    B,pi,hp=md['B'],md['pi'],md['hp'];bb=mm=0.;n=0;blk=collections.defaultdict(lambda:[0.,0.,0])
    for l in ls:
        P,Y=l['P'],l['Y'];q=2;prior=hp[l['house']] if use_house else pi;A=np.log(prior+1e-15)
        for z in range(len(pi)):
            sc=np.log(np.maximum(P[:q],1e-15))+B[z];mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True);A[z]+=np.log(np.maximum(Q[np.arange(q),Y[:q]],1e-300)).sum()
        mx=A.max();post=np.exp(A-mx);post/=post.sum()
        for i in range(q,len(Y)):
            pb=P[i,Y[i]];pm=0.
            for z in range(len(pi)):
                sc=np.log(np.maximum(P[i],1e-15))+B[z];sc-=sc.max();Q=np.exp(sc);Q/=Q.sum();pm+=post[z]*Q[Y[i]]
            b=-math.log2(max(pb,1e-300));m=-math.log2(max(pm,1e-300));bb+=b;mm+=m;n+=1;v=blk[l['bif']];v[0]+=b;v[1]+=m;v[2]+=1
    gs=[(a-b)/c for a,b,c in blk.values() if c];sd=np.std(gs,ddof=1) if len(gs)>1 else 0
    return {'n':n,'base_bits':bb/n,'model_bits':mm/n,'gain_bits':(bb-mm)/n,'blocks_n':len(gs),'blocks_positive':int(sum(x>0 for x in gs)),'block_mean':float(np.mean(gs)),'block_sd':float(sd),'block_mean_over_sd':float(np.mean(gs)/sd) if sd>0 else None}

D=[l for l in LINES if l['fold'] in (2,3)];V=[l for l in LINES if l['fold']==4];TR=[l for l in LINES if l['fold'] in (2,3,4)];TE=[l for l in LINES if l['fold'] in (0,1)]
baseD=fit_struct(D);Dp=attach(D,baseD);Vp=attach(V,baseD)
G_D=Dp;G_V=Vp

def val_task(arg):
    S,l2=arg;md=fit_mix(G_D,S,l2);return {'K':S,'l2':l2,'global':score(G_V,md,False),'house':score(G_V,md,True),'iter':md['iter']}

if __name__=="__main__":
    grid=[(S,l2) for S in (1,2,3,4,6,8) for l2 in (1.,3.,10.,30.)]
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex: val=list(ex.map(val_task,grid,chunksize=1))
    best=max(val,key=lambda r:max(r['global']['gain_bits'],r['house']['gain_bits']));variant='house' if best['house']['gain_bits']>best['global']['gain_bits'] else 'global'
    baseF=fit_struct(TR);TRp=attach(TR,baseF);TEp=attach(TE,baseF);md=fit_mix(TRp,best['K'],best['l2']);fg=score(TEp,md,False);fh=score(TEp,md,True)
    ladder=[]
    for S in (1,2,3,4,6,8):
        x=fit_mix(TRp,S,best['l2']);ladder.append({'K':S,'global':score(TEp,x,False),'house':score(TEp,x,True)})
    out={'corpus_n':len(rows),'lines_eligible':len(LINES),'discovery_lines':len(D),'validation_lines':len(V),'train_lines':len(TR),'test_lines':len(TE),'fold_sha':ns['canon_sha'](folds),'validation_grid':val,'selected':{'K':best['K'],'l2':best['l2'],'variant':variant,'validation_gain_bits':best[variant]['gain_bits']},'final':{'global':fg,'house':fh,'selected':fh if variant=='house' else fg},'final_ladder_descriptive':ladder,'model':{'B':md['B'].tolist(),'pi':md['pi'].tolist(),'house_priors':{k:v.tolist() for k,v in md['hp'].items()}},'baseline_theta_final':baseF[2].tolist()}
    print("LATENT_LINE_STATE_JSON="+json.dumps(out,separators=(",",":")),flush=True)
