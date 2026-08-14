from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from .common import GateFailure, atomic_json

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mapping-panel",type=Path,required=True,
                    help="28-row CSV: currier_page,expected_label,skip12_folio,keep12_folio")
    ap.add_argument("--folio-labels",type=Path,required=True,
                    help="CSV: folio,label")
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    panel=read_csv(a.mapping_panel); labs=read_csv(a.folio_labels)
    req={"currier_page","expected_label","skip12_folio","keep12_folio"}
    if len(panel)!=28 or not panel or not req.issubset(panel[0]):
        raise GateFailure("mapping panel must contain exactly 28 rows and required columns")
    labmap={r["folio"]:r["label"] for r in labs}
    scores={}
    missing={}
    for scheme,col in [("skip12","skip12_folio"),("keep12","keep12_folio")]:
        ok=0; miss=[]
        for r in panel:
            fol=r[col]
            if fol not in labmap:
                miss.append(fol); continue
            ok += int(labmap[fol]==r["expected_label"])
        scores[scheme]=ok; missing[scheme]=sorted(set(miss))
    qualifying=[s for s,v in scores.items() if v>=26 and not missing[s]]
    if len(qualifying)==1:
        verdict="PASS"; selected=qualifying[0]
    elif len(qualifying)>1:
        verdict="ABSTAIN_UNRESOLVED"; selected=None
    elif max(scores.values())<20:
        verdict="FAIL"; selected=None
    else:
        verdict="ABSTAIN_UNRESOLVED"; selected=None
    res={"formal_verdict":verdict,"target_opened":False,"scores":scores,
         "missing_folios":missing,"selected_mapping":selected,
         "criterion":">=26/28 with complete labels; if both qualify, mapping remains ambiguous; if both <20 reject linear mapping"}
    a.out.mkdir(parents=True,exist_ok=True); atomic_json(a.out/"U2_MAPPING_GATE.json",res)
    print(json.dumps(res,indent=2))
if __name__=="__main__":
    main()
