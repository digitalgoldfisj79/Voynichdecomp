#!/usr/bin/env python3
import argparse, json, hashlib
from pathlib import Path
import source_id_guard as guard

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--summary',required=True); ap.add_argument('--power',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); m=json.load(open(a.manifest)); s=json.load(open(a.summary)); p=json.load(open(a.power))
    candidate={'complete':True,'report':p}
    if not guard.validate_power_report(m,candidate): raise SystemExit('INVALID_POWER_REPORT')
    s['power_reporting']=candidate
    s['summary_sha256']=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    Path(a.output).write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
    print('P1FINAL_SUMMARY_SHA='+s['summary_sha256'])
if __name__=='__main__': main()
