"""Stability of VoynichStats quire_optimize suggestions across 6 settings (metric x unit), ZLZI, 500 runs, seed 42.
Values transcribed from the MCP outputs of 2026-11-14 (suggested order; beatRecorded = percentile of the RECORDED order among random orders)."""
import pandas as pd
from collections import Counter
S = {  # quire: [(setting, suggested, beatRecorded)]
"A":[("cos-w","3124",.08),("tfidf-w","4213",.032),("jac-w","3124",.032),("cos-b","3124",.16),("tfidf-b","3124",.16),("jac-b","1324",.412)],
"B":[("cos-w","4312",.432),("tfidf-w","4132",.614),("jac-w","2314",.464),("cos-b","2143",.156),("tfidf-b","2143",.156),("jac-b","4321",.682)],
"C":[("cos-w","1234",.964),("tfidf-w","1234",.916),("jac-w","1234",.964),("cos-b","1234",.964),("tfidf-b","1234",.964),("jac-b","1234",.918)],
"D":[("cos-w","2341",.232),("tfidf-w","3412",.306),("jac-w","2314",.352),("cos-b","2341",.388),("tfidf-b","2341",.388),("jac-b","1432",.414)],
"E":[("cos-w","1234",.952),("tfidf-w","2134",.844),("jac-w","1234",.84),("cos-b","1234",.952),("tfidf-b","1234",.952),("jac-b","2143",.66)],
"F":[("cos-w","1342",.25),("tfidf-w","1342",.158),("jac-w","1324",.278),("cos-b","1342",.194),("tfidf-b","1342",.194),("jac-b","1243",.692)],
"G":[("cos-w","2341",.188),("tfidf-w","2341",.28),("jac-w","1432",.158),("cos-b","2341",.292),("tfidf-b","2341",.258),("jac-b","2143",.452)],
"H":[(s,"21",0) for s in ["cos-w","tfidf-w","jac-w","cos-b","tfidf-b","jac-b"]],
"M":[("cos-w","32145",.692),("tfidf-w","41235",.528),("jac-w","32145",.55),("cos-b","54123",.554),("tfidf-b","54123",.674),("jac-b","42135",.448)],
"O":[("cos-w","12",.53),("tfidf-w","12",.53),("jac-w","12",.53),("cos-b","12",.53),("tfidf-b","12",.53),("jac-b","21",0)],
"Q":[("cos-w","12",.518),("tfidf-w","12",.518),("jac-w","12",.518),("cos-b","12",.518),("tfidf-b","12",.518),("jac-b","21",0)],
"S":[(s,"21",0) for s in ["cos-w","tfidf-w","jac-w","cos-b","tfidf-b","jac-b"]],
"T":[("cos-w","324561",.866),("tfidf-w","324561",.744),("jac-w","423561",.528),("cos-b","324561",.654),("tfidf-b","532416",.624),("jac-b","164235",.508)]}
rows=[]
for q,v in S.items():
    c=Counter(s for _,s,_ in v); mode,n=c.most_common(1)[0]; rec="".join(str(i) for i in range(1,len(mode)+1))
    br=[b for *_,b in v]
    rows.append(dict(quire=q,sheets=len(mode),modal_order=mode,settings_agree=f"{n}/6",modal_is_recorded=mode==rec,
                     recorded_pct_min=min(br),recorded_pct_median=sorted(br)[2],recorded_pct_max=max(br)))
R=pd.DataFrame(rows); R.to_csv("quire_stability.csv",index=False); print(R.to_string(index=False))
