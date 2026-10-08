#!/usr/bin/env python3
# RECIPES-ENTRYPOS1 — preregistered 2026-10-08
import collections,hashlib,json,math,os,re,urllib.request
import numpy as np

INNOV_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
ZL_URL="https://raw.githubusercontent.com/noah-chelednik/voynich-data/472ef7366606a799fc8f1044c037e06b413f6ddd/data_sources/cache/ZL3b-n.txt"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
ZL_SHA="bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
SEED=202610081330; NREP=100000

ii={"__name__":"innov2"}
exec(compile(urllib.request.urlopen(INNOV_URL,timeout=120).read().decode(),INNOV_URL,"exec"),ii)
rows=ii["rows"]; para_id=ii["para_id"]; base_prob=ii["base_prob"]; fit_pos=ii["fit_pos"]; tilt_p=ii["tilt_p"]; fit_tilt=ii["fit_tilt"]
K=ii["K"]; fnum=ii["fnum"]; segment=ii["m"]["segment"]
TARGET=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,117)) for s in ("r","v"))
TARGET.discard("f116v")

# Exact source verification.
zl=urllib.request.urlopen(ZL_URL,timeout=120).read()
got=hashlib.sha256(zl).hexdigest()
if got!=ZL_SHA: raise RuntimeError(("ZL_SHA_MISMATCH",got))
zltxt=zl.decode("utf-8")
obj=json.loads(urllib.request.urlopen(CORPUS_URL,timeout=120).read())

def safe_tokens(txt):
    out=[]
    for t in str(txt).split():
        t=t.lower()
        if not re.fullmatch(r"[a-z]+",t): continue
        try: segment(t)
        except Exception: continue
        out.append(t)
    return out

# Parse exact literal <%> ... <$> physical entry spans at the line level.
entries=[]; current={}
pat=re.compile(r"^<(f\d+[rv]\d*)\.(\d+),[^>]+>\s+(.*)$")
for ln in zltxt.splitlines():
    m=pat.match(ln)
    if not m: continue
    fol,lno,txt=m.group(1),int(m.group(2)),m.group(3)
    if fol not in TARGET: continue
    if "<%>" in txt:
        if fol in current: raise RuntimeError(("nested_entry",fol,lno))
        current[fol]={"folio":fol,"eno":1+sum(e["folio"]==fol for e in entries),"lines":[]}
    if fol in current:
        current[fol]["lines"].append(lno)
    if "<$>" in txt and fol in current:
        entries.append(current.pop(fol))
if current: raise RuntimeError(("open_entries",sorted(current)))
if len(entries)!=285: raise RuntimeError(("entry_count",len(entries)))

# Build token-position map from the canonical slim ZLZI text, using exactly the
# same alpha + segmentability filter as the strict row universe.
ROLEMAP={}; ENTRY_META={}
for eid,e in enumerate(entries):
    toks=[]
    perline=[]
    for lno in e["lines"]:
        rec=obj["pages"].get(e["folio"],{}).get(str(lno),{})
        lt=safe_tokens(rec.get("t",{}).get("ZLZI",""))
        perline.append((lno,lt))
        toks.extend(lt)
    n=len(toks)
    ENTRY_META[eid]={"folio":e["folio"],"eno":e["eno"],"n_tokens":n,"lines":e["lines"]}
    if n<6: continue
    off=0
    for lno,lt in perline:
        for pos,tok in enumerate(lt):
            ix=off+pos
            if ix==0: role="P1"
            elif ix==1: role="P2"
            elif ix==2: role="P3"
            elif ix==n-3: role="E2"
            elif ix==n-2: role="E1"
            elif ix==n-1: role="E0"
            else: role="BODY"
            ROLEMAP[(e["folio"],str(lno),pos)]=(eid,ix,n,role,tok)
        off+=len(lt)

# Compact SELECT q0 on full physical-star target, exactly RECIPES-REG1 shape.
page=collections.defaultdict(lambda:np.zeros(K,float)); para=collections.defaultdict(lambda:np.zeros(K,float)); hist=collections.defaultdict(list)
by=collections.OrderedDict(); candidate=0; mapped=0; mismatch=0; f115_lines=set()
for r in rows:
    fol=r["folio"]; ln=int(r["line"]); lk=(fol,ln); pk=fol; pid=para_id(pk,ln); pq=(pk,pid)
    y=int(r["start"]); pos=int(r["pos"])
    if pos==0:
        pass
    else:
        prev_piece=hist[lk][-1][1]; rc=np.zeros(K,float)
        for yy,pp in hist[lk][-6:]: rc[int(yy)]+=1
        p=base_prob(r["section"],prev_piece,page[pk],para[pq],rc)
        if fol in TARGET:
            candidate+=1
            mp=ROLEMAP.get((fol,str(r["line"]),pos))
            if mp is not None:
                if mp[4]==r["token"]: mapped+=1
                else: mismatch+=1
            if fol=="f115r": f115_lines.add(str(r["line"]))
            e={"p":p,"y":y,"folio":fol,"line":lk,"rowpos":pos,"token":r["token"],"map":mp}
            by.setdefault(lk,[]).append(e)
    page[pk][y]+=1; para[pq][y]+=1; hist[lk].append((y,r["final_piece"]))

agreement=mapped/max(candidate,1)
all_f115_plus={str(k) for k,v in obj["pages"].get("f115r",{}).items() if str(v.get("u",""))=="+P0" and safe_tokens(v.get("t",{}).get("ZLZI",""))}
missing_f115=sorted(all_f115_plus-f115_lines,key=lambda x:int(x))
if agreement<.995 or mismatch>0 or missing_f115:
    raise RuntimeError(("ENTRY_MAPPING_FAIL",candidate,mapped,mismatch,agreement,missing_f115))

raw=list(by.values())
mods={0:fit_pos(raw,0),1:fit_pos(raw,1)}
def lrole(i,n):
    if i==0:return "FIRST"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n): return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

q0=[]
for seq in raw:
    fol=seq[0]["folio"]; tr=1-fnum(fol)%2; gb,rb,cb=mods[tr]; n=len(seq); zz=[]
    for i,e in enumerate(seq):
        ro=lrole(i,n); b=cb.get((ro,lbin(n)),rb.get(ro,gb)); x=dict(e); x["p"]=tilt_p(e["p"],b)
        if x["map"] is not None:
            x["eid"],x["entry_ix"],x["entry_n"],x["entry_role"],_ = x["map"]
        else:
            x["eid"]=x["entry_ix"]=x["entry_n"]=x["entry_role"]=None
        zz.append(x)
    q0.append(zz)

# Cross-fitted physical-entry-role nuisance.
ROLES=("P1","P2","P3","BODY","E2","E1","E0")
rolemods={}
for par in (0,1):
    d=collections.defaultdict(list)
    for seq in q0:
        if fnum(seq[0]["folio"])%2!=par: continue
        for e in seq:
            if e["entry_role"] in ROLES: d[e["entry_role"]].append(e)
    rolemods[par]={r:fit_tilt(d[r]) for r in ROLES if len(d[r])>=20}

qer=[]
for seq in q0:
    fol=seq[0]["folio"]; tr=1-fnum(fol)%2; zz=[]
    for e in seq:
        x=dict(e); b=rolemods[tr].get(e["entry_role"])
        if b is not None: x["p"]=tilt_p(e["p"],b)
        zz.append(x)
    qer.append(zz)

def collect(lines):
    folsum=collections.defaultdict(lambda:[0.,0])
    entrysum=collections.defaultdict(lambda:[0.,0])
    rolesum=collections.defaultdict(lambda:[0.,0])
    linesum=collections.defaultdict(lambda:[0.,0])
    for seq in lines:
        fol=seq[0]["folio"]
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]); b=int(seq[t-1]["y"])
            if a==b: continue
            e=seq[t]
            if e["eid"] is None: continue
            y=int(e["y"]); val=(1.0 if y==a else 0.0)-float(e["p"][a])
            folsum[fol][0]+=val; folsum[fol][1]+=1
            entrysum[e["eid"]][0]+=val; entrysum[e["eid"]][1]+=1
            rolesum[(fol,e["entry_role"])][0]+=val; rolesum[(fol,e["entry_role"])][1]+=1
            linesum[(fol,int(e["line"][1]))][0]+=val; linesum[(fol,int(e["line"][1]))][1]+=1
    return folsum,entrysum,rolesum,linesum
F0,E0,R0,L0=collect(q0); FE,EE,RE,LE=collect(qer)
def score(d,k): return d[k][0]/d[k][1] if d.get(k,[0,0])[1] else None

raw115=score(F0,"f115r"); er115=score(FE,"f115r")
if raw115 is None or abs(raw115-(-0.058317877794920626))>5e-4:
    raise RuntimeError(("RAW_F115_MISMATCH",raw115))
ret=abs(er115)/abs(raw115)

# matched physical-entry randomization, secondary/calibrative
def ebin(n):
    if n<=24:return "L24"
    if n<=34:return "L34"
    if n<=49:return "L49"
    return "L50P"
obsids=[eid for eid,m in ENTRY_META.items() if m["folio"]=="f115r" and EE.get(eid,[0,0])[1]>0]
obs_counts=collections.Counter(ebin(ENTRY_META[e]["n_tokens"]) for e in obsids)
donors=collections.defaultdict(list)
for eid,m in ENTRY_META.items():
    if m["folio"]=="f115r" or EE.get(eid,[0,0])[1]==0: continue
    donors[ebin(m["n_tokens"])].append(eid)
rng=np.random.default_rng(SEED); null=np.empty(NREP)
for j in range(NREP):
    pick=[]
    for b,n in obs_counts.items():
        pool=donors[b]
        if len(pool)<n: raise RuntimeError(("donor_short",b,n,len(pool)))
        pick.extend(rng.choice(pool,size=n,replace=False).tolist())
    s=sum(EE[e][0] for e in pick); n=sum(EE[e][1] for e in pick); null[j]=s/n
mu=float(null.mean()); sd=float(null.std(ddof=1)); p=float((1+np.sum(np.abs(null-mu)>=abs(er115-mu)))/(NREP+1))
pct=float(np.mean(null<=er115))

# role and handoff localization
roleloc={}
for r in ROLES:
    roleloc[r]={"q0":score(R0,("f115r",r)),"qER":score(RE,("f115r",r)),
                "n":RE.get(("f115r",r),[0,0])[1]}
def linepart(d,lo=None,hi=None):
    ss=nn=0.
    for (f,l),(s,n) in d.items():
        if f!="f115r":continue
        if lo is not None and l<lo:continue
        if hi is not None and l>hi:continue
        ss+=s;nn+=n
    return {"score":ss/nn if nn else None,"n":int(nn)}
handoff={"q0_first12":linepart(L0,hi=12),"q0_after12":linepart(L0,lo=13),
         "qER_first12":linepart(LE,hi=12),"qER_after12":linepart(LE,lo=13),
         "q0_first18":linepart(L0,hi=18),"q0_after18":linepart(L0,lo=19),
         "qER_first18":linepart(LE,hi=18),"qER_after18":linepart(LE,lo=19)}

if ret<=.50: decision="ENTRY_POSITION_ABSORBS_MAJORITY"
elif ret>=.80: decision="ENTRY_POSITION_DOES_NOT_EXPLAIN_F115R"
else: decision="MIXED_ENTRY_POSITION_EFFECT"

out={"programme":"RECIPES-ENTRYPOS1","status":"complete",
     "source":{"zl_sha":got,"entries":len(entries),"mapping_candidate":candidate,"mapped":mapped,"agreement":agreement},
     "scores":{"f103_f114_q0":sum(F0[f][0] for f in F0 if f not in ("f115r","f115v","f116r"))/sum(F0[f][1] for f in F0 if f not in ("f115r","f115v","f116r")),
               "f103_f114_qER":sum(FE[f][0] for f in FE if f not in ("f115r","f115v","f116r"))/sum(FE[f][1] for f in FE if f not in ("f115r","f115v","f116r")),
               "f115r_q0":raw115,"f115r_qER":er115,
               "f115v_q0":score(F0,"f115v"),"f115v_qER":score(FE,"f115v"),
               "f116r_q0":score(F0,"f116r"),"f116r_qER":score(FE,"f116r"),
               "retention":ret,"decision":decision},
     "entry_randomization":{"observed_entry_count":len(obsids),"length_bin_counts":dict(obs_counts),
                            "pseudo_mean":mu,"pseudo_sd":sd,"two_sided_p":p,"percentile":pct,
                            "q005":float(np.quantile(null,.005)),"q995":float(np.quantile(null,.995))},
     "f115r_role_localization":roleloc,"handoff_localization":handoff}
print("RECIPES_ENTRYPOS1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
