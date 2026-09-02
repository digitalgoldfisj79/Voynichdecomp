#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
import urllib.request
URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v14-e-frame-mediation-20260902/experiments/vbm_v14_e_frame/vbm_v14_e_frame.py'
req=urllib.request.Request(URL,headers={'User-Agent':'VBMV14COnly/2026-09-02'})
src=urllib.request.urlopen(req,timeout=120).read().decode('utf-8')
ns={'__name__':'v14module'};exec(compile(src,URL,'exec'),ns)
v11=ns['load_v11']();data=v11['get_json'](v11['DATA_URL']);segments,_=v11['build_corpus'](data)
tr,cnt,elig,pairs,dec=ns['full_setup'](segments);hold=ns['occurrences'](segments,'HOLD')
C=ns['branch_C'](tr,hold,elig)
import json
print('V14_C='+json.dumps(C,sort_keys=True),flush=True)
print('V14_C_ONLY_META='+json.dumps({'eligible_nuclei':len(elig),'e_ladder_pairs':len(pairs),'train_occ':len(tr),'hold_occ':len(hold),'B_status':'NON_EVALUABLE_FROZEN_NULL_SUPPORT_FAILURE','A_status':'PRESERVE_PRIOR_BINDING_OUTPUT'},sort_keys=True),flush=True)
