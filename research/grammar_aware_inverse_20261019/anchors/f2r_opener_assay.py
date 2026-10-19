#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""F2R-ANCHOR-0. Descriptive, known-post-selected f2r plant-name opener hypothesis.
DOES NOT translate anything. Exact within-register line-position matched null.
"""
import collections,hashlib,json,re,urllib.request
from statistics import median

SRC="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REG="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ed0d3e5cbba93701520cfc733ab16a94b2d0ab3a/research/grammar_aware_inverse_20261019/anchors/herbal_a_register.json"
TARGET="f2r"
def read(url):return urllib.request.urlopen(url,timeout=120).read()
def lv(a,b):
    if len(a)<len(b):a,b=b,a
    d=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        nd=[i]+[0]*len(b)
        for j,s in enumerate(b,1):nd[j]=min(nd[j-1]+1,d[j]+1,d[j-1]+(c!=s))
        d=nd
    return d[-1]
def pct(num,den):return num/den if den else None
def line0(pages,folio,line):
    rec=pages.get(folio,{}).get(str(line))
    if not isinstance(rec,dict) or "P" not in str(rec.get("u","")):return None
    toks=str(rec.get("t",{}).get("ZLZI","")).strip().lower().split()
    if not toks or not re.fullmatch("[a-z]+",toks[0]):return None
    return toks[0]
def getpair(pages,folio):
    a=line0(pages,folio,1);b=line0(pages,folio,8)
    if a is None or b is None:return None
    return {"folio":folio,"word_1":a,"word_8":b,"lengths":[len(a),len(b)],
            "edit_distance":lv(a,b),"normalized_edit":lv(a,b)/max(len(a),len(b)),
            "same_initial":a[0]==b[0]}
def main():
    raw=read(SRC)
    assert hashlib.sha256(raw).hexdigest()==SHA,"Transcription SHA mismatch"
    corpus=json.loads(raw);pages=corpus["pages"]
    reg=json.loads(read(REG));ids={x["folio"] for x in reg["records"]}
    assert len(ids)==89 and TARGET in ids,("wrong register",len(ids))
    t=getpair(pages,TARGET)
    assert t is not None and (t["word_1"],t["word_8"])==("kydainy","kydain"),t
    pool=[]
    for folio in sorted(ids-{TARGET}):
        row=getpair(pages,folio)
        if row is None:continue
        L=row["lengths"]
        if not (5<=L[0]<=10 and 5<=L[1]<=10):continue
        if abs(L[0]-len(t["word_1"]))>1 or abs(L[1]-len(t["word_8"]))>1:continue
        pool.append(row)
    w=[]
    for line in range(2,14):
        if line==8:continue
        token=line0(pages,TARGET,line)
        if token is None:continue
        if not 5<=len(token)<=10 or abs(len(token)-len(t["word_8"]))>1:continue
        d=lv(t["word_1"],token)
        w.append({"line":line,"word":token,"len":len(token),"edit_distance":d,
                  "normalized_edit":d/max(len(t["word_1"]),len(token))})
    def stats(xs):
        return {"n":len(xs),"within_ed_1":sum(x["edit_distance"]<=1 for x in xs),
                "fraction_ed_le_1":pct(sum(x["edit_distance"]<=1 for x in xs),len(xs)),
                "normalized_le_target":sum(x["normalized_edit"]<=t["normalized_edit"]+1e-12 for x in xs),
                "fraction_normalized_le_target":pct(sum(x["normalized_edit"]<=t["normalized_edit"]+1e-12 for x in xs),len(xs)),
                "median_edit":median([x["edit_distance"] for x in xs]) if xs else None}
    out={"schema":"F2R_CENTAURY_OPENER_DESCRIPTIVE_20261019",
       "complete":True,"target":t,"matched_cross_folio":{"stats":stats(pool),"records":pool},
       "within_f2r":{"stats":stats(w),"records":w},
       "scope":{"n_currier_a_herbal":len(ids),"eligible_folios":len(pool),
                "selection":"matching positions 1/8; each word length within 1 of target; same Currier A Herbal; strict P running units"},
       "audit":{"transcription_url":SRC,"corpus_sha256":SHA,"metadata_url":REG,
                "not_exhaustive_of_herbal_b":True,"no_physical_bifolium_holdout":True,
                "f2r_was_selected_by_prior_literature":True},
       "interpretation":"Prior Bax/Sherwood centaury candidate, NOT independently verified botanical identification or decrypted word; line8 paragraph boundary pending visual review. Similarity is not a discriminating semantic anchor."}
    print("F2R_ANCHOR_RESULT="+json.dumps(out,separators=(",",":"),sort_keys=True),flush=True)
if __name__=="__main__":main()
