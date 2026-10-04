import os,json,math,collections,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE_COMMIT="9ea05f51aef9b2ab00e443bdd0a0e0a295a42ba8"
ROOT=f"https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BASE_COMMIT}/research"
os.environ["TARGET_URL"]=ROOT+"/data/selector_target_bundle_20261004.gz.b64"
os.environ["CAUSAL_URL"]=ROOT+"/data/selector_causal_baseline_20261004.xz.b64"
os.environ["SOURCE_URL"]=ROOT+"/data/selector_source_controls_20261004.json"

# Reuse the frozen bundles, metrics, and pre-registered broad control generators.
src=urllib.request.urlopen(ROOT+"/selector_mechanism_tournament_renderer_20261004.py",timeout=60).read().decode()
_ns={"__name__":"selector_tournament_base"}
exec(compile(src,"selector_mechanism_tournament_renderer_20261004.py","exec"),_ns)

K=_ns["K"]; METS=_ns["METS"]; TARGET_VEC=_ns["TARGET_VEC"]; HALVES=_ns["HALVES"]
CAUSAL=_ns["CAUSAL"]; metrics=_ns["metrics"]
sample_source=_ns["sample_source"]; table_core=_ns["table"]; source_table_core=_ns["hybrid"]

# Frozen from the independent clean 60/40 reinforced-availability selector.
ALPHA=30.0
THETA_PAGE=np.array([
0.3437162263574351,0.2537411498430932,0.3207633023520617,0.31227310588302554,
0.4907827653485044,0.5232580702790207,0.34612681856793276,0.9031121641334551,
0.32199281125042034,0.6190083392438619,0.3399427373676645,1.1119277161237104
],float)
THETA_PARA=np.array([
0.1688071877608053,0.04523238539311162,0.028524860013724937,0.1322435921987762,
-0.18705809522210926,0.2606257361488568,0.25561668314686503,0.2897810358341678,
0.29020526000134056,0.31500410181665006,0.21889503280717712,0.08568000296553302
],float)
THETA_RECENT=0.0530951464493163

G=np.array(CAUSAL["global"],float)
PG=(G+.5)/(G.sum()+.5*K)
CM={(int(s),int(p)):np.array(c,float) for s,p,c in CAUSAL["counts"]}
RECORDS=CAUSAL["records"]
N_LATENT=sum(1 for r in RECORDS if int(r[5])>0)

def _latent_ok(latent):
    return sum(map(len,latent))==N_LATENT

def render(latent,seed,beta=1.5):
    rng=np.random.default_rng(seed^0x6d2b79f5)
    use_core=latent is not None
    if use_core and not _latent_ok(latent):
        raise RuntimeError(("latent length",sum(map(len,latent)),N_LATENT))
    flat=[x for s in latent for x in s] if use_core else []
    li=0
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    line=collections.defaultdict(list)
    out=collections.OrderedDict()
    for fol,ln,pa,sec,piece,pos,yobs in RECORDS:
        fk=int(fol); lk=(fk,int(ln)); pk=(fk,int(pa)); pos=int(pos)
        if pos==0:
            y=int(yobs)  # separate LINE_ENTRY process: observed state only
        else:
            c=CM.get((int(sec),int(piece)))
            base=(c+ALPHA*PG)/(c.sum()+ALPHA) if c is not None else PG
            rc=np.zeros(K,float)
            for q in line[lk][-6:]: rc[int(q)]+=1
            sc=np.log(np.maximum(base,1e-15))
            sc += THETA_PAGE*np.log1p(page[fk])
            sc += THETA_PARA*np.log1p(para[pk])
            sc += THETA_RECENT*rc
            if use_core:
                z=int(flat[li]); li+=1
                sc[z]+=float(beta)
            sc-=sc.max(); pr=np.exp(sc); pr/=pr.sum()
            y=int(rng.choice(K,p=pr))
            out.setdefault(pk,[]).append(y)
        page[fk][y]+=1; para[pk][y]+=1; line[lk].append(y)
    if use_core and li!=len(flat): raise RuntimeError(("unused latent",li,len(flat)))
    return list(out.values())

FAMILIES=("URN","SOURCE_R","TABLE_R","SOURCE_TABLE_R")

def generate(fam,seed,beta):
    if fam=="URN": return render(None,seed,beta)
    if fam=="SOURCE_R": return render(sample_source(seed),seed,beta)
    if fam=="TABLE_R": return render(table_core(seed),seed,beta)
    if fam=="SOURCE_TABLE_R": return render(source_table_core(seed),seed,beta)
    raise KeyError(fam)

def task(arg):
    fam,seed,beta=arg
    return fam,metrics(generate(fam,seed,beta)).tolist()

def run(beta,NS=160,train_n=100):
    jobs=[(fam,202610050000+fi*100000+i,float(beta)) for fi,fam in enumerate(FAMILIES) for i in range(NS)]
    samples={f:[] for f in FAMILIES}
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
        for fam,v in ex.map(task,jobs,chunksize=2): samples[fam].append(v)
    samples={f:np.array(v,float) for f,v in samples.items()}
    TR=np.arange(min(train_n,NS)); TE=np.arange(min(train_n,NS),NS)
    pool=np.vstack([samples[f][TR] for f in FAMILIES])
    mu=pool.mean(0); sd=pool.std(0,ddof=1); sd=np.where(sd<1e-8,1,sd)
    cent={f:((samples[f][TR]-mu)/sd).mean(0) for f in FAMILIES}
    def classify(x,allow=None):
        z=(np.asarray(x)-mu)/sd; fs=list(allow) if allow else list(FAMILIES)
        ds={f:float(np.mean((z-cent[f])**2)) for f in fs}
        return min(ds,key=ds.get),ds
    conf={f:collections.Counter() for f in FAMILIES}
    for f in FAMILIES:
        for x in samples[f][TE]: conf[f][classify(x)[0]]+=1
    recall={f:conf[f][f]/len(TE) for f in FAMILIES}
    overall=sum(conf[f][f] for f in FAMILIES)/(len(FAMILIES)*len(TE))
    def pairacc(a,b):
        good=tot=0
        for f in (a,b):
            for x in samples[f][TE]:
                good+=classify(x,(a,b))[0]==f; tot+=1
        return good/tot
    pairwise={}
    for f in FAMILIES[1:]: pairwise["URN_"+f]=pairacc("URN",f)
    pairwise["SOURCE_R_TABLE_R"]=pairacc("SOURCE_R","TABLE_R")
    pairwise["SOURCE_R_SOURCE_TABLE_R"]=pairacc("SOURCE_R","SOURCE_TABLE_R")
    pairwise["TABLE_R_SOURCE_TABLE_R"]=pairacc("TABLE_R","SOURCE_TABLE_R")
    gate=(overall>=.80 and min(recall.values())>=.65 and
          pairwise["SOURCE_R_TABLE_R"]>=.80 and
          min(pairwise["URN_"+f] for f in FAMILIES[1:])>=.80)
    def score(x):
        pred,ds=classify(x); pct={}
        for f in FAMILIES:
            vals=[]
            for y in samples[f][TE]:
                z=(y-mu)/sd
                vals.append(float(np.mean((z-cent[f])**2)))
            pct[f]=sum(v<=ds[f] for v in vals)/len(vals)
        return {"nearest":pred,"distance":ds,"within_family_distance_percentile":pct}
    halves={k:score(metrics(v)) for k,v in HALVES.items()}
    groups={
      "lag_memory":[i for i,k in enumerate(METS) if k.startswith("mi") or k.startswith("rep") or k.startswith("ng_") or k in ("aaa_excess","aba_excess","aababb_excess")],
      "dispersion":[i for i,k in enumerate(METS) if k.startswith("fano") or k.startswith("return")],
      "topology":[i for i,k in enumerate(METS) if k in ("reciprocity","top10_edge","eff_outdegree","top20_bigram","top50_trigram")],
      "reset":[i for i,k in enumerate(METS) if k=="start_js"]
    }
    abl={}
    for g,ix in groups.items():
        keep=[i for i in range(len(METS)) if i not in ix]
        pmu=pool[:,keep].mean(0); psd=pool[:,keep].std(0,ddof=1); psd=np.where(psd<1e-8,1,psd)
        pc={f:((samples[f][TR][:,keep]-pmu)/psd).mean(0) for f in FAMILIES}
        z=(TARGET_VEC[keep]-pmu)/psd; ds={f:float(np.mean((z-pc[f])**2)) for f in FAMILIES}
        abl["without_"+g]={"nearest":min(ds,key=ds.get),"distance":ds}
    return {
      "beta":float(beta),"n_control_per_family":NS,"train_per_family":len(TR),"test_per_family":len(TE),
      "calibration":{"overall_accuracy":overall,"recall":recall,"pairwise_accuracy":pairwise,"gate_pass":gate,
                     "confusion":{f:dict(conf[f]) for f in FAMILIES}},
      "target":score(TARGET_VEC),"target_halves":halves,"metric_ablations":abl,
      "control_centroids":{f:{k:float(v) for k,v in zip(METS,samples[f][TR].mean(0))} for f in FAMILIES}
    }

if __name__=="__main__":
    primary=run(1.5,160,100)
    print("PRIMARY_REINFORCED_JSON="+json.dumps(primary,separators=(",",":")),flush=True)
