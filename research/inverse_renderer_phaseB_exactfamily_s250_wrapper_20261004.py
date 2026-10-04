import urllib.request,types
u="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/de5364d44bc8e64a5513f850e273c7b844327742/research/inverse_renderer_recoverability_phaseB_20261004.py"
ns={"__name__":"phaseB"};exec(compile(urllib.request.urlopen(u,timeout=60).read().decode(),u,"exec"),ns)
a=types.SimpleNamespace(N=4000,K=16,d=4,rank=2,strength=2.5,seed=20261004,restarts=4,dense_epochs=45,sparse_epochs=20,msteps=25,lr=.04,device="cpu")
ns["run"](a)
