#!/usr/bin/env python3
# Shared core for VMS R1 observable-channel ladder, frozen 2026-10-08.
import collections,math,urllib.request
import numpy as np

R4_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d1daa5877fc7d8ba2be9bfa98ec69150d95af8ee/research/vms_residrec4b_recipes_qualification_20261008.py"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"

src=urllib.request.urlopen(R4_URL,timeout=120).read().decode()
r4={"__name__":"r4defs"}
exec(compile(src.split("\nap=argparse.ArgumentParser()")[0],R4_URL,"exec"),r4)

lsrc=urllib.request.urlopen(LAT_URL,timeout=120).read().decode()
lat={"__name__":"latdefs"}
# Load full definitions including annotated canonical rows but avoid expensive tests below fit functions.
exec(compile(lsrc.split("\ndef flatten")[0],LAT_URL,"exec"),lat)

K=r4["K"]
ROWS=lat["rows"]
ST=lat["ST"]
CO="RECIPES_FULL"

def fnum(f):
    import re
    m=re.match(r"f([0-9]+)",str(f))
    return int(m.group(1)) if m else -1

def piecebin(n):
    return 0 if n<=1 else (1 if n==2 else (2 if n==3 else 3))

def charbin(n):
    return 0 if n<=2 else (1 if n<=4 else (2 if n<=6 else 3))

def line_id_from_event(e):
    lk=e.get("line")
    if isinstance(lk,(tuple,list)) and len(lk)>=2:
        return (str(lk[0]),int(lk[1]))
    return (str(e["folio"]),int(lk))

RAW=collections.OrderedDict()
for rr in ROWS:
    n=fnum(rr["folio"])
    if 103<=n<=116:
        RAW.setdefault((rr["folio"],int(rr["line"])),[]).append(rr)

def reconstruct_parent_q3():
    """Returns all Recipes lines once, each on its untouched outer TEST fold, with q3 p."""
    out=[]
    fold_meta=[]
    real=r4["real_lines"]()
    for j in range(5):
        tr,va,te=r4["q1_split"](real,j)
        model=r4["train_fixed"](tr,j)
        q3,b3,n=r4["predict"](model,te)
        b1,n1=r4["bits_q1"](te)
        assert n==n1
        # hard parent reconstruction gate
        got=b3/n
        ref=r4["PHASE_BPE"][j]
        if abs(got-ref)>2e-5:
            raise RuntimeError(("R1_PARENT_Q3_RECONSTRUCTION_FAIL",j,got,ref))
        out.extend(q3)
        fold_meta.append({"fold":j,"n":n,"q1_bpe":b1/n,"q3_bpe":got,"gain":(b1-b3)/n})
    return out,fold_meta

def enrich_q3_lines(q3lines):
    """
    Attach only historical observable channels. Current-token FORM never enters x.
    Each returned event keeps q3 probability p and current y.
    """
    out=[]
    mapped=0
    for seq in q3lines:
        if not seq: continue
        lk=line_id_from_event(seq[0])
        raw=RAW.get(lk)
        if raw is None:
            raise RuntimeError(("R1_RAW_LINE_MISSING",lk))
        scored=raw[1:]
        if len(scored)!=len(seq):
            raise RuntimeError(("R1_LINE_LENGTH_MISMATCH",lk,len(scored),len(seq)))
        hist=[raw[0]]
        zz=[]
        for t,(e,cur) in enumerate(zip(seq,scored)):
            prev=hist[-1]
            if int(e["y"])!=int(cur["start"]):
                raise RuntimeError(("R1_Y_MISMATCH",lk,t,e["y"],cur["start"]))
            if int(e["prev"])!=int(prev["start"]):
                raise RuntimeError(("R1_PREV_MISMATCH",lk,t,e["prev"],prev["start"]))
            recent=hist[-6:]
            fc=np.zeros(K,float); lc=np.zeros(4,float); joint=np.zeros((K,K),float)
            fams=[]
            for h in recent:
                fcl=int(ST[h["final_piece"]]); cb=charbin(len(h["token"]))
                fc[fcl]+=1.; lc[cb]+=1.; joint[int(h["start"]),fcl]+=1.
                fams.append(f'{int(h["start"])}:{fcl}:{cb}')
            prev_fc=int(ST[prev["final_piece"]])
            prev_cb=charbin(len(prev["token"]))
            prev_pb=piecebin(len(prev["pieces"]))
            prev_fam=f'{int(prev["start"])}:{prev_fc}:{prev_cb}'
            rec=dict(e)
            rec.update({
              "fold":int(cur["fold"]),
              "folio":str(cur["folio"]),
              "line_no":int(cur["line"]),
              "event_index":t,
              "prev_final_class":prev_fc,
              "prev_piece_bin":prev_pb,
              "prev_char_bin":prev_cb,
              "prev_family":prev_fam,
              "recent_final_counts":fc,
              "recent_char_counts":lc,
              "recent_joint_counts":joint.reshape(-1),
              "recent_prev_family_count":float(sum(x==prev_fam for x in fams)),
              "recent_unique_families":float(len(set(fams))),
            })
            zz.append(rec); mapped+=1; hist.append(cur)
        out.append(zz)
    return out,mapped

def build():
    q3,fold_meta=reconstruct_parent_q3()
    lines,n=enrich_q3_lines(q3)
    return lines,fold_meta,n

def flatten_lines(lines):
    ev=[e for s in lines for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    F=np.array([int(e["fold"]) for e in ev],int)
    return ev,P,Y,F
