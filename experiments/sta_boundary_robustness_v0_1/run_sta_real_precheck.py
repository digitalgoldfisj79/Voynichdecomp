#!/usr/bin/env python3
"""Stage-1 execution of the already-frozen STA preregistration.
Computes only source validation + real R1-R4 gates. It changes no statistic or decision rule.
The full held-out S0/S1/S2 run remains required iff the representation gates survive.
"""
import importlib.util, json, math, sys, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sta',HERE/'run_sta_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def main():
    rf=sys.argv[1] if len(sys.argv)>1 else '/tmp/RF1b.txt'
    sections=m.load_sections(); raw,lines,pa=m.parse_rf(rf,sections)
    folios=sorted({x['folio'] for x in lines}); overlap=sum(f in sections for f in folios)/len(folios) if folios else 0
    validation={'header_ok':raw.startswith(b'#=IVTFF STA1 2.0'),'folios':len(folios),'segments':len(lines),'tokens':sum(len(x['tokens']) for x in lines),'section_overlap':overlap}
    validation['pass']=validation['header_ok'] and validation['folios']>=200 and validation['tokens']>=25000 and overlap>=.95
    out={'metadata':{'seed':m.SEED,'nperm':m.NPERM,'source_sha256':hashlib.sha256(raw).hexdigest(),'parser':pa},'validation':validation}
    if validation['pass']:
        real=m.real_stats(lines); out['real']=real
        lg=lambda x:abs(math.log(x)) if x and x>0 else float('inf')
        r1=real['E2_N0']['ratio']>=1.10 and real['E2_N0']['z']>=2 and lg(real['E2_N1']['ratio'])<lg(real['E2_N0']['ratio']) and lg(real['E2_N3']['ratio'])<lg(real['E2_N0']['ratio']) and ((real['E2_N1']['ratio']<1.10 or abs(real['E2_N1']['z'])<2) or (real['E2_N3']['ratio']<1.10 or abs(real['E2_N3']['z'])<2))
        r2=real['ED1_N0']['ratio']>=1.10 and real['ED1_N0']['z']>=2 and lg(real['ED1_N3']['ratio'])<lg(real['ED1_N0']['ratio']) and (real['ED1_N3']['ratio']<1.10 or abs(real['ED1_N3']['z'])<2)
        long=real['LEN_long']; r3_power=long['observed']>=60; r3=(long['ratio']>=1.15 and long['z']>=2) if r3_power else None
        r4=abs(real['direction']['acc_red']['z'])<2 and abs(real['direction']['subsite']['z'])<2
        out['representation_gates']={'R1_E2':bool(r1),'R2_ED1':bool(r2),'R3_long':('PASS' if r3 else 'FAIL') if r3_power else 'UNDERPOWERED','R4_direction_absent':bool(r4)}
        out['stage1_decision']='PROCEED_TO_HELDOUT' if (r1 and r2 and r4) else 'REPRESENTATION_SENSITIVE_STOP'
    else: out['stage1_decision']='INCONCLUSIVE'
    p=Path('results/sta_boundary_robustness_v0_1');p.mkdir(parents=True,exist_ok=True)
    (p/'REAL_PRECHECK_20260815.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
