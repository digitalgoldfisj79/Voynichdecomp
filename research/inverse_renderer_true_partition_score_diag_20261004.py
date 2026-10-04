import urllib.request,json
u="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/e188d1fc665e6676c84d79797dabcba709201cfd/research/inverse_renderer_collapsed_partition_phaseE_20261004.py"
ns={"__name__":"phaseE"};exec(compile(urllib.request.urlopen(u,timeout=60).read().decode(),u,"exec"),ns)
L=ns["ztr"].astype("int16")
ec,et,tc,tt=ns["init_counts"](L,ns["ctx"],ns["opt"],ns["nd"],ns["K"],ns["C"],ns["O"])
sc=ns["collapsed_score"](ec,et,tc,tt,ns["PB"],5.0,.10,ns["K"],ns["C"],ns["O"])
print("TRUE_PARTITION_COLLAPSED_SCORE_JSON="+json.dumps({"score":float(sc),"nmi":1.0,"best_found":-22396.505113980907,"margin_vs_best_found":float(sc+22396.505113980907)}),flush=True)
