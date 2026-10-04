#!/usr/bin/env python3
# Phase L2a: fast oracle half of corrected structured-source calibration.
import json,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from concurrent.futures import ProcessPoolExecutor,as_completed

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c6384a58449b0f27f959be580c5e9613d71d83eb/research/structured_source_context_calibration_phaseL2_20261004.py"
m={"__name__":"l2lib"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)

def one(fam,seed):
    rng=np.random.default_rng(seed);U,Ve,Vr=m["encoder"](seed)
    rows=[];allz=[];allsec=[];alls=[];alld=[]
    for sec in range(m["NSEC"]):
        z,A,pi=m["source_sequence"](fam,sec,m["SECLEN"],rng)
        obs,L=m["render_and_ll"](z,U,Ve,Vr,rng)
        Ec=m["E_control"](z);Ef=m["E_form"](L)
        _,gc,_=m["fb"](Ec,A,pi,False);_,gf,_=m["fb"](Ef,A,pi,False)
        sl=slice(m["NFIT"]+m["NVAL"],None);zt=z[sl]
        rows.append({"section":sec,
          "sig_decode_acc":float(np.mean(L.argmax(1)==m["SIG_OF"][z])),
          "control":m["metrics"](zt,gc[sl],seed+sec*11),
          "form":m["metrics"](zt,gf[sl],seed+sec*13)})
        allz.extend(z.tolist());allsec.extend([sec]*len(z));alls.extend(m["SIG_OF"][z].tolist());alld.extend(L.argmax(1).tolist())
    def agg(ch,k):return float(np.nanmean([x[ch][k] for x in rows]))
    return {"family":fam,"seed":seed,
      "source_section_nmi":float(normalized_mutual_info_score(allsec,allz)),
      "signature_section_nmi":float(normalized_mutual_info_score(allsec,alls)),
      "decoded_signature_section_nmi":float(normalized_mutual_info_score(allsec,alld)),
      "sig_decode_acc":float(np.mean(np.array(alls)==np.array(alld))),
      "control":{"source_nmi":agg("control","source_nmi"),"collision_nmi":agg("control","collision_nmi"),"auc":agg("control","same_source_auc")},
      "form":{"source_nmi":agg("form","source_nmi"),"collision_nmi":agg("form","collision_nmi"),"auc":agg("form","same_source_auc")},
      "sections":rows}

if __name__=="__main__":
    specs=[(f,s) for f in ("LANG","NOTATION","TABLE") for s in (20262301,20262302,20262303,20262304,20262305)]
    out=[]
    with ProcessPoolExecutor(max_workers=15) as ex:
        fut={ex.submit(one,*x):x for x in specs}
        for q in as_completed(fut):
            r=q.result();out.append(r);print("L2A_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in out if x["family"]==fam]
        med=lambda ch,k:float(np.median([x[ch][k] for x in rr]))
        summary[fam]={"n":len(rr),
          "source_section_nmi":float(np.median([x["source_section_nmi"] for x in rr])),
          "signature_section_nmi":float(np.median([x["signature_section_nmi"] for x in rr])),
          "sig_decode_acc":float(np.median([x["sig_decode_acc"] for x in rr])),
          "control_source_nmi":med("control","source_nmi"),"control_collision_nmi":med("control","collision_nmi"),"control_auc":med("control","auc"),
          "form_source_nmi":med("form","source_nmi"),"form_collision_nmi":med("form","collision_nmi"),"form_auc":med("form","auc")}
    print("STRUCTURED_SOURCE_PHASEL2A_JSON="+json.dumps({"summary":summary,"replicates":out},separators=(",",":")),flush=True)
