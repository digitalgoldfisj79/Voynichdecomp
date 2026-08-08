#!/usr/bin/env python3
"""Stage 3 network/image inventory for Alpine–Venetian Corridor v0.1.

No VMS images, embeddings or similarity scores are read here. The script only
verifies machine-readable facsimiles for the sealed cohort and counts IIIF
canvases / checks direct image or HTML availability.
"""
from __future__ import annotations
import json, sys, time
from urllib.parse import urlparse
import requests

UA={"User-Agent":"VoynichCorridorResearch/0.1 (+noncommercial manuscript census)"}
TIMEOUT=25

ITEMS=[
# corridor
("external:bsb_cod_icon_242","iiif","https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00013084/manifest"),
("external:bodl_canon_misc_554","iiif","https://iiif.bodleian.ox.ac.uk/iiif/manifest/6ae78449-a166-439c-9c98-ddfc7b6c8bf8.json"),
("external:vat_lat_4082","iiif","https://digi.vatlib.it/iiif/MSS_Vat.lat.4082/manifest.json"),
("registry:merlon_beinecke_ms_327_venice","iiif","https://collections.library.yale.edu/manifests/10269817"),
("registry:cr_british_library_egerton_ms_2020_erbario_carrarese","html","https://commons.wikimedia.org/wiki/Category:Carrara_Herbal_(c.1400)_-_BL_Egerton_MS_2020"),
("registry:cr_venezia_biblioteca_marciana_lat_vi_59_2548_roccabonell","html","https://nbm.regione.veneto.it/Generale/ricerca/AnteprimaManoscritto.html"),
("registry:zeg_de_virga_world_map","image","https://upload.wikimedia.org/wikipedia/commons/e/ef/DeVirgaWorldMap.jpg"),
("external:pizzigani_parm_1612_1367","image","https://upload.wikimedia.org/wikipedia/commons/7/70/Pizigani_1367_Chart_10MB.jpg"),
("external:pizzigano_ambrosiana_1373","html","https://artsandculture.google.com/story/hAUxVcTL66hQKw"),
("external:pizzigano_bell_1424","html","https://umedia.lib.umn.edu/"),
("external:andrea_bianco_atlas_1436","html","https://commons.wikimedia.org/wiki/Category:Atlante_di_Andrea_Bianco_dell%27anno_1436"),
("external:fra_mauro_map_1450","html","https://mostre.museogalileo.it/framauro/"),
# controls Lombardy
("external:bnf_lat_7342","iiif","https://gallica.bnf.fr/iiif/ark:/12148/btv1b9068035n/manifest.json"),
("external:bnf_nal_1673","iiif","https://gallica.bnf.fr/iiif/ark:/12148/btv1b105380445/manifest.json"),
("external:sloane_4016","iiif","https://bl.digirati.io/iiif/ark:/81055/vdc_100165172997.0x000001"),
("external:casanatense_4182","html","https://casanatense.contentdm.oclc.org/digital/collection/miniature/search/searchterm/4182"),
# controls Bavaria/Swabia
("registry:cr_munchen_bayerische_staatsbibliothek_cgm_38_konrad_von_","iiif","https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00043227/manifest"),
("registry:mn_munchen_bayerische_staatsbibliothek_clm_14622","iiif","https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00109392/manifest"),
("external:bsb_clm_14684","iiif","https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00086348/manifest"),
("external:bsb_clm_14783","html","https://ptolemaeus.badw.de/ms/663"),
("external:walsperger_pal_lat_1362b","iiif","https://digi.vatlib.it/iiif/MSS_Pal.lat.1362.pt.B/manifest.json"),
("external:mendel_amb_317_2_pre1451","html","https://online-service.nuernberg.de/viewer/hausbuecher/"),
]

def parse_manifest(m):
    if m.get("sequences"):
        cvs=(m.get("sequences") or [{}])[0].get("canvases") or []
        labels=[c.get("label") for c in cvs]
        return "v2",len(cvs),labels[:5],labels[-5:]
    cvs=m.get("items") or []
    def lab(c):
        x=c.get("label")
        if isinstance(x,dict):
            vals=next(iter(x.values()),[]); return vals[0] if vals else None
        return x
    labels=[lab(c) for c in cvs]
    return "v3",len(cvs),labels[:5],labels[-5:]

def probe(key,kind,url):
    out={"candidate_key":key,"kind":kind,"url":url,"ok":False}
    try:
        r=requests.get(url,headers=UA,timeout=TIMEOUT,allow_redirects=True,stream=(kind=="image"))
        out.update(status=r.status_code,final_url=r.url,content_type=r.headers.get("content-type"),content_length=r.headers.get("content-length"))
        if kind=="image":
            out["ok"]=r.ok and "image" in (r.headers.get("content-type") or "")
            r.close()
        elif kind=="iiif":
            if r.ok:
                m=r.json(); pres,n,first,last=parse_manifest(m)
                out.update(ok=n>0,presentation=pres,canvas_count=n,first_labels=first,last_labels=last,manifest_id=m.get("@id") or m.get("id"))
            else: out["error"]="http"
        else:
            out["ok"]=r.ok
            out["bytes_sampled"]=len(r.content[:100000])
    except Exception as e:
        out["error"]=type(e).__name__+": "+str(e)[:300]
    return out

def main():
    res=[]
    for i,(k,t,u) in enumerate(ITEMS,1):
        x=probe(k,t,u); res.append(x)
        print(json.dumps(x,ensure_ascii=False),flush=True)
        time.sleep(.15)
    summary={
        "n":len(res),"ok":sum(bool(x.get("ok")) for x in res),
        "iiif_n":sum(x["kind"]=="iiif" for x in res),
        "iiif_ok":sum(x["kind"]=="iiif" and x.get("ok") for x in res),
        "canvas_total":sum(int(x.get("canvas_count",0)) for x in res),
        "failures":[x["candidate_key"] for x in res if not x.get("ok")],
    }
    print("STAGE3_SUMMARY="+json.dumps(summary,ensure_ascii=False),flush=True)
    print("STAGE3_JSON="+json.dumps(res,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
