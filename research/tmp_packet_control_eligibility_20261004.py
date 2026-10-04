import urllib.request,re,collections
u="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
ns={"__name__":"x"};exec(compile(urllib.request.urlopen(u).read().decode(),u,"exec"),ns)
mu="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/80d5c9778cb27e3dd20b049ab8804a7bb8832fd2/research/data/physical_metadata_nonp70_20261004.tsv"
meta={}
for i,ln in enumerate(urllib.request.urlopen(mu).read().decode().splitlines()):
    if i==0:continue
    f,q,b,c,h,w,l=ln.split("\t")
    mm=re.search(r"\\d+",f)
    if not mm: continue
    n=int(mm.group()); meta.setdefault(n,[]).append((f,q,c,h))
def mode(n,idx):
    z=[x[idx] for x in meta.get(n,[])]; return collections.Counter(z).most_common(1)[0][0] if z else "NA"
leaves=sorted({int(re.search(r"\d+",l["folio"]).group()) for l in ns["LINES"] if l["fold"] in (0,1)})
out=[]
for n in leaves:
    q,c,h=mode(n,1),mode(n,2),mode(n,3)
    exact=[m for m in leaves if m!=n and mode(m,1)==q and mode(m,2)==c and mode(m,3)==h]
    sameq=[m for m in leaves if m!=n and mode(m,1)==q]
    out.append((n,q,c,h,len(exact),len(sameq)))
print("N",len(out))
print("exact>=2",sum(x[4]>=2 for x in out),"exact>=1",sum(x[4]>=1 for x in out),"sameq>=2",sum(x[5]>=2 for x in out))
print(out)
