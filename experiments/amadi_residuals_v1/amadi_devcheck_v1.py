# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json
import amadi_driver_v1c as d
m=d.m

std,vc,r12=m.load_lms(smoke=True)
rows=[]
for k in m.PWA_RULES:
    ctl=m.make_control(std,vc,r12,"PWA",k,"latin",91+k,"DEV",1200,1200)
    r=m.cand_fit(ctl,std,vc,r12,"PWA",k,"latin",f"DEV:PWA:{k}",False)
    z={"family":"PWA","rule":k,"recovery":r["recovery"],"agreement":r["agreement"],"state_agreement":r.get("state_agreement"),"converged":r["converged"],"restarts_each":r["restarts_each"],"score_diff":r["score_diff"]}; rows.append(z); print("DEV",json.dumps(z,sort_keys=True),flush=True)
for lang in ["latin","italian"]:
    ctl=m.make_control(std,vc,r12,"GH",5,lang,111 if lang=="latin" else 112,"DEV",1200,1200)
    r=m.cand_fit(ctl,std,vc,r12,"GH",5,lang,f"DEV:GH:{lang}",False)
    z={"family":"GH","language":lang,"recovery":r["recovery"],"agreement":r["agreement"],"state_agreement":r.get("state_agreement"),"converged":r["converged"],"restarts_each":r["restarts_each"],"score_diff":r["score_diff"]}; rows.append(z); print("DEV",json.dumps(z,sort_keys=True),flush=True)
print("RESULT_JSON",json.dumps(rows,sort_keys=True))
