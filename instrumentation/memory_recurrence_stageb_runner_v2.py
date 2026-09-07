#!/usr/bin/env python3
import argparse, collections, hashlib, importlib.util, json, math, os, time
from pathlib import Path
import numpy as np

EXPECTED_MANIFEST_SHA='ccfd588efdf5c223ccd3f53465a463c96e55163abb9287c4453b7023a90f4b76'
SOURCE_LAM={0:.03,1:.03,2:.03,3:.02,4:.03}
LAM_GRID=(0.0,0.005,0.01,0.02,0.03,0.04,0.06,0.08)
MU=.0009
SOURCES=('working_set_only','line_reset_r64')
REPS=('R0_canonical_tokens','R1_repeated_glyph_collapse')
FEATURES=['gain_line_reset_vs_zero_bits_per_token','gain_page_r64_vs_zero_bits_per_token','line_reset_advantage_vs_page_r64_bits_per_token','exact_same_line_lag2_5_excess','exact_cross_line_lag2_5_excess','exact_same_minus_cross_lag2_5','exact_same_line_lag6_16_excess','exact_cross_line_lag6_16_excess','exact_same_minus_cross_lag6_16','exact_same_line_lag17_64_excess','exact_cross_line_lag17_64_excess','exact_same_minus_cross_lag17_64','cachehit_same_line_lag2_64_excess','cachehit_cross_line_lag2_64_excess','cachehit_same_minus_cross_lag2_64']
BUCKETS=((2,5),(6,16),(17,64))
SPLIT_NS={'development':520,'validation':522,'power':523}
SOURCE_CODE={'working_set_only':1,'line_reset_r64':2}

def load_t0b():
    p=os.environ.get('T0B_RUNNER','/workspace/t0b/t0b_runner.py'); sp=importlib.util.spec_from_file_location('t0b_runner',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def load_v1():
    p=Path(__file__).resolve().parent/'memory_recurrence_runner_v1.py'; sp=importlib.util.spec_from_file_location('v1_runner',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def verify_manifest(path):
    o=json.loads(Path(path).read_text()); b=json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode(); h=hashlib.sha256(b).hexdigest()
    if h!=EXPECTED_MANIFEST_SHA: raise SystemExit(f'MANIFEST_HASH_MISMATCH {h}')
    return h

def seed(split,source,trial,fold,slot=0): return SPLIT_NS[split]*1_000_000 + SOURCE_CODE[source]*100_000 + trial*1_000 + fold*100 + slot

def generate_source(t0b,source,orig_tr,orig_te,fold,split,trial,strength=None):
    src=t0b.RegionalModel(orig_tr,fold,lam=(0.0 if source=='working_set_only' else (SOURCE_LAM[fold] if strength is None else float(strength))))
    def one(sk,slot):
        if source=='working_set_only': return src.generate(sk,seed(split,source,trial,fold,slot),'full')
        return t0b.generate_line_reset(src,sk,np.random.default_rng(seed(split,source,trial,fold,slot)))
    return one(orig_tr,11),one(orig_te,29)

def line_state_class(t0b):
    class LineResetState(t0b.RegionalState):
        def boundary(self,r):
            old=(self.page,self.line); super().boundary(r)
            if (r['page'],r['line'])!=old: self.page_seq=[]
    return LineResetState

def score_line_model(t0b,model,rows):
    st=line_state_class(t0b)(model.base.train_vocab); bits=0.0
    for r in rows:
        bits-=math.log2(model.prob(st,r,r['token'],'full')); st.update(r,r['token'])
    return bits/len(rows)

def choose_line_lambda(t0b,train,outer_fold):
    bifs=sorted(set(r['bifolium'] for r in train)); vals={x:[] for x in LAM_GRID}
    for q in range(4):
        vb=set(b for i,b in enumerate(bifs) if i%4==q); tr=[r for r in train if r['bifolium'] not in vb]; va=[r for r in train if r['bifolium'] in vb]
        if not tr or not va: continue
        m=t0b.RegionalModel(tr,outer_fold,lam=0.0,mu=MU)
        for x in LAM_GRID:
            m.lam=float(x); vals[x].append(score_line_model(t0b,m,va))
    means={x:(float(np.mean(v)) if v else 1e9) for x,v in vals.items()}; best=min(LAM_GRID,key=lambda x:(means[x],x)); return float(best),means

def smoothed(h,n): return (float(h)+.5)/(float(n)+1.0)
def logex(h,n,nh,nn): return math.log2(smoothed(h,n)/smoothed(nh,nn))

def line_recurrence_features(rows,split,source,trial,fold,rep,v1,t0b):
    rr=v1.transform_rows(rows,rep,t0b.s2); toks=sorted({r['token'] for r in rr}); tid={t:i for i,t in enumerate(toks)}; pages=collections.OrderedDict()
    for r in rr:
        pages.setdefault(r['page'],{'x':[],'line':[]}); pages[r['page']]['x'].append(tid[r['token']]); pages[r['page']]['line'].append(r['line'])
    obs=np.zeros((3,2,2),float); nul=np.zeros((3,2,2),float); cache_obs=np.zeros((2,2),float); cache_nul=np.zeros((2,2),float)
    for pi,v in enumerate(pages.values()):
        x=np.asarray(v['x'],int); ln=np.asarray(v['line']); n=len(x)
        if n<3: continue
        rng=np.random.default_rng(seed(split,source,trial,fold,50+pi)); P=np.stack([rng.permutation(n) for _ in range(64)]); X=x[P]
        for bi,(lo,hi) in enumerate(BUCKETS):
            for d in range(lo,min(hi,n-1)+1):
                same=ln[:-d]==ln[d:]
                for ci,mask in enumerate((same,~same)):
                    if not np.any(mask): continue
                    a=x[:-d][mask]; b=x[d:][mask]; obs[bi,ci,0]+=int(np.sum(a==b)); obs[bi,ci,1]+=int(mask.sum())
                    aa=X[:,:-d][:,mask]; bb=X[:,d:][:,mask]; nul[bi,ci,0]+=int(np.sum(aa==bb)); nul[bi,ci,1]+=int(aa.size)
        for j in range(2,n):
            cand=np.arange(max(0,j-64),j-1)
            for ci,mask in enumerate((ln[cand]==ln[j],ln[cand]!=ln[j])):
                idx=cand[mask]
                if idx.size==0: continue
                cache_obs[ci,1]+=1; cache_obs[ci,0]+=int(np.any(x[idx]==x[j])); cache_nul[ci,1]+=64; cache_nul[ci,0]+=int(np.sum(np.any(X[:,idx]==X[:,j,None],axis=1)))
    out=[]
    for bi in range(3):
        ex=[]
        for ci in range(2): ex.append(logex(obs[bi,ci,0],obs[bi,ci,1],nul[bi,ci,0]/64.0,nul[bi,ci,1]/64.0))
        out.extend([ex[0],ex[1],ex[0]-ex[1]])
    cx=[]
    for ci in range(2): cx.append(logex(cache_obs[ci,0],cache_obs[ci,1],cache_nul[ci,0]/64.0,cache_nul[ci,1]/64.0))
    out.extend([cx[0],cx[1],cx[0]-cx[1]]); return out

def run_fold(t0b,v1,source,split,trial,fold,strength=None):
    orig_tr,orig_te=t0b.w.fold_data(fold); syn_tr,syn_te=generate_source(t0b,source,orig_tr,orig_te,fold,split,trial,strength)
    zero=t0b.RegionalModel(syn_tr,fold,lam=0.0,mu=MU); zero_bits=zero.score(syn_te,'full')
    page_lam,_=t0b.choose_lambda(syn_tr,fold); page=t0b.RegionalModel(syn_tr,fold,lam=page_lam,mu=MU); page_bits=page.score(syn_te,'full')
    line_lam,_=choose_line_lambda(t0b,syn_tr,fold); line=t0b.RegionalModel(syn_tr,fold,lam=line_lam,mu=MU); line_bits=score_line_model(t0b,line,syn_te)
    ll=[zero_bits-line_bits,zero_bits-page_bits,page_bits-line_bits]; reps={}
    for rep in REPS:
        f=ll+line_recurrence_features(syn_te,split,source,trial,fold,rep,v1,t0b); assert len(f)==15 and np.all(np.isfinite(f)); reps[rep]=[float(z) for z in f]
    return {'fold':fold,'selected_page_lambda':page_lam,'selected_line_lambda':line_lam,'features':reps}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('split',choices=SPLIT_NS); ap.add_argument('source',choices=SOURCES); ap.add_argument('start',type=int); ap.add_argument('stop',type=int); ap.add_argument('--strength',type=float,default=None); ap.add_argument('--manifest',required=True); a=ap.parse_args()
    mh=verify_manifest(a.manifest); t0b=load_t0b(); v1=load_v1(); assert len(t0b.w.ROWS)==34087
    print('P1V2QA='+json.dumps({'manifest_sha256':mh,'rows':34087,'source':a.source,'split':a.split,'features':FEATURES,'target_sealed':True},separators=(',',':')),flush=True)
    for t in range(a.start,a.stop):
        st=time.time(); folds=[run_fold(t0b,v1,a.source,a.split,t,f,a.strength) for f in range(5)]; o={'assay_id':'memory_recurrence_source_id_v2','split':a.split,'source':a.source,'trial':t,'strength':a.strength,'feature_names':FEATURES,'folds':folds,'elapsed_sec':time.time()-st,'target_opened':False}; print('P1V2TRIAL='+json.dumps(o,separators=(',',':')),flush=True)
if __name__=='__main__': main()
