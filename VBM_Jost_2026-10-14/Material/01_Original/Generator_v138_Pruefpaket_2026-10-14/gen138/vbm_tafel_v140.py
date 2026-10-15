# v130: aus v127. Wortfugen-Schicht: Ereignis mit Wortgrenze im Cluster und >= 3 Konsonanten -> Zelle(Auslaut) + Vorbau = Blatt3(Anlaut, Folgevokal);
# Blatt 3 = Rangzuordnung deutscher (Anlaut, Folgevokal)-Paare auf die 564 VMS-Vorbau-Strings. Kurze Wortfugen (2 Konsonanten) und Binnencluster wie bisher.
import os,sys,collections,importlib.util,json,random,re,unicodedata,bisect
from pathlib import Path
sys.argv=['x']
spec=importlib.util.spec_from_file_location('gen','Paket_v100/vbm_tafel_v119det.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
VMS='vms_B_REVA_atomstream.txt'; BAU=os.environ.get('BAU','klartext/hartlieb_klartext.txt')
VOW=set('aeiou')
words=gen.words_of(Path(BAU).read_text(encoding='utf-8',errors='ignore'))
s=''; bl=[]
for w in words: s+=w; bl.append(len(s))
vi=[i for i,c in enumerate(s) if c in VOW]
ev=[]   # (v, K, v2, fl, aus, an)  aus/an leer = keine Wortgrenze im Cluster
for p,q in zip(vi,vi[1:]):
    K=s[p+1:q]; i=bisect.bisect_right(bl,p)
    if i<len(bl) and bl[i]<q and K.replace('\u00a7',''): b=bl[i]; aus=s[p+1:b]; an=s[b:q]
    else: aus=an=''
    ev.append((s[p],K.replace('\u00a7',''),s[q],'\u00a7' in K,aus.replace('\u00a7',''),an.replace('\u00a7','')))
cm=collections.Counter(K for v,K,v2,f,aus,an in ev if K)
b1=gen.Blatt1(VMS,cm,{K:0.7 for K in cm})
# volles Zelleninventar nach Rang
from panel_v126 import atoms as _atoms, cellparse as _cp
_cells=collections.Counter()
for _l in open(VMS,encoding='utf-8'):
    for _t in _l.rstrip('\n').split('\t')[-1].split():
        _p=_cp(_atoms(_t))
        if _p and _p[2] and not _p[3] and _p[0] in 'ynrl': _cells[(_p[0],_p[1],_p[2])]+=1
CELLS=[c for c,n in _cells.most_common()]
# deutsche Cluster-Einheiten, die als Grundzelle geschrieben werden: Binnencluster und kurze Wortfugen (als String)
_cu=collections.Counter(K if not (aus and an and len(aus)+len(an)>=3) else aus for v,K,v2,f,aus,an in ev if K)
CLU=[k for k,n in _cu.most_common()]
RANK={k:CELLS[i] if i<len(CELLS) else CELLS[len(CELLS)-1-(i-len(CELLS))%max(1,len(CELLS)//3)] for i,k in enumerate(CLU)}
# v140: Homophone je Cluster. Die Zellen werden nach Clustermasse verteilt (Restgroessenverfahren, wie eine klassische homophone Tafel):
# haeufige Cluster bekommen mehrere Zellen mit verschiedenen Koepfen, seltene eine. Der Schreiber nimmt fuer jedes Vorkommen reihum die naechste.
import os
HOMK=float(os.environ.get('HOMK','1.0'))   # Anteil der Zellen, die als zusaetzliche Homophone verteilt werden
_tot=sum(_cu.values()); _extra=int(HOMK*len(CELLS))
_quota={k:_extra*n/_tot for k,n in _cu.items()}
_alloc={k:int(q) for k,q in _quota.items()}
for k,_ in sorted(_quota.items(),key=lambda x:-(x[1]-int(x[1])))[:_extra-sum(_alloc.values())]: _alloc[k]+=1
HOM={k:[RANK[k]] for k in CLU}
_pool=[c for c in CELLS]; _pi=0
for k in CLU:
    for j in range(_alloc.get(k,0)):
        c=_pool[(_pi)%len(_pool)]; _pi+=7   # Schrittweite 7: Homophone eines Clusters liegen weit auseinander (andere Koepfe)
        if c not in HOM[k]: HOM[k].append(c)
HCNT=collections.Counter()
def hom_peek(k): return HOM[k][HCNT[k]%len(HOM[k])]
def hom_take(k):
    c=HOM[k][HCNT[k]%len(HOM[k])]; HCNT[k]+=1; return c
dek=[c for c in b1.counts if c[1]=='de' and c[2]=='k']; old=b1.row_of.get(('de','k'))
if old and old in b1.map: b1.map[old]=[c for c in b1.map[old] if not (c[1]=='de' and c[2]=='k')] or b1.map[old]
b1.map['n']=[c for c in dek for _ in range(max(1,round(b1.counts[c]/25)))]; b1.row_of[('de','k')]='n'
# Blatt 3 neu: (Anlaut, Folgevokal) nach Rang -> VMS-Vorbau-Strings nach Rang
VB=[k for k,n in json.load(open('vorbau_strings_vms.json'))]
fc=collections.Counter((an,v2) for v,K,v2,f,aus,an in ev if aus and an and len(aus)+len(an)>=3)
FJ=[k for k,n in fc.most_common()]
B3={k:VB[i] if i<len(VB) else VB[-1] for i,k in enumerate(FJ)}
# Blatt 3 alt fuer Binnen-Reste
RS=[k for k,n in json.load(open('rest_stuecke_dt.json'))]; B3r={r:VB[i] if i<len(VB) else VB[-1] for i,r in enumerate(RS)}
def pieces(K):
    if not K: return ['']
    if K in b1.map: return [K]
    for k in range(len(K)-1,0,-1):
        if K[:k] in b1.map and K[k:] in b1.map: return [K[:k],K[k:]]
    K2=b1.norm(K)
    if K2!=K and K2 in b1.map: return [K2]
    out=[]; r=K2
    while r:
        for k in range(len(r),0,-1):
            if r[:k] in b1.map: out.append(r[:k]); r=r[k:]; break
        else: return None
    return out
R=json.load(open('belegung.json')); pair_v={tuple(k.split('|')):p['vowel'] for k,p in R['pairs'].items()}
pair_v={p:v for p,v in pair_v.items() if p[0]!='y' and not (p[0]=='oq' and p[1] in 'nrl')}
byclass={cl:{tuple(k.split('|')):n for k,n in d.items()} for cl,d in R['byclass'].items()}
pair_v={p:v for p,v in pair_v.items() if p[0]!='y' and not (p[0]=='oq' and p[1] in 'nrl')}
for mb in ('lo','la','ro','ra'):
    for k in 'ynrl':
        if (mb[-1],k) in pair_v: pair_v[(mb,k)]=pair_v[(mb[-1],k)]
from panel_v126 import atoms as _at2, cellparse as _cp2
def fam(k): return 'y' if k=='y' else 'n' if k=='n' else 'rl'
PC=collections.Counter()
for _l in open(VMS,encoding='utf-8'):
    _tk=_l.rstrip('\n').split('\t')[-1].split()
    for _a,_b in zip(_tk,_tk[1:]):
        _p=_cp2(_at2(_a)); _q=_cp2(_at2(_b))
        if not _p or not _q: continue
        _s=''.join(_at2(_a)); _br=None
        for mb in ('lo','la','ro','ra'):
            if _s.endswith(mb) and len(_s)>len(mb): _br=mb; break
        if _br is None: _br=_p[4] if _p[4]!='-' else '-'
        PC[(gen.hclass(_p[2] or ''),fam(_q[0]),(_br,_q[0]))]+=1
# Vorbau-Pool als ganze Formen je Kopf (Tafel 6), Anlaut nach Rang, injektiv solange Formen reichen, sonst Anlautklasse (gezaehlt)
POOL=collections.defaultdict(collections.Counter)
for _l in open(VMS,encoding='utf-8'):
    for _t in _l.rstrip('\n').split('\t')[-1].split():
        _p=_cp2(_at2(_t))
        if _p and _p[2] and _p[3]: POOL[_p[2]][''.join(_p[3])]+=1
POOL={h:[v for v,n in c.most_common()] for h,c in POOL.items()}
POOLZ=collections.defaultdict(collections.Counter)
for _l in open(VMS,encoding='utf-8'):
    for _t in _l.rstrip('\n').split('\t')[-1].split():
        _p=_cp2(_at2(_t))
        if _p and _p[2] and _p[3]: POOLZ[(_p[1],_p[2])][''.join(_p[3])]+=1
POOLZ={z:[v for v,n in c.most_common()] for z,c in POOLZ.items() if sum(c.values())>=10}
ALLP=[v for v,n in collections.Counter(x for c in POOL.values() for x in c).most_common()]
ANr=[k for k,n in collections.Counter(an for v,K,v2,f,aus,an in ev if aus and an and len(aus)+len(an)>=3).most_common()]
VERLUST=[0]
def vorbau_pool(head,an,st=''):
    lst=POOLZ.get((st,head)) or POOL.get(head) or ALLP
    i=ANr.index(an) if an in ANr else len(ANr)
    if i<len(lst): return lst[i]
    VERLUST[0]+=1; cls=an[0]; j=ANr.index(cls) if cls in ANr else 0
    return lst[j%len(lst)]
# Nahtregel Schwelle 1
TRI=set(); BI=set()
for _l in open(VMS,encoding='utf-8'):
    for _t in _l.rstrip('\n').split('\t')[-1].split():
        _a=_at2(_t)
        for i in range(len(_a)-1): BI.add((_a[i],_a[i+1]))
        for i in range(len(_a)-2): TRI.add((_a[i],_a[i+1],_a[i+2]))
def ok(tok):
    a=_at2(tok)
    return all((a[i],a[i+1]) in BI for i in range(len(a)-1)) and all((a[i],a[i+1],a[i+2]) in TRI for i in range(len(a)-2))
NAHT=[0,0,0]
SPLITR={('ch','l'):0.51,('sh','l'):0.62,('d','l'):0.56,('o','l'):0.62,('a','l'):0.50,('o','r'):0.76,('a','r'):0.59,('ch','r'):0.50,('e','l'):0.50,('e','r'):0.50}
SPLITC=collections.Counter(); SPLITN=[0]
G={'k','t','p','f'}; B={'ch','sh','d','s'}
def hc(h): return 'G' if h in G else 'B' if h in B else 'L' if h in ('l','r') else 'X' if h else 'S'
BEW={'lo','la','ro','ra','niia','nia','niio','ma','so','l','r','s','ria'}
rng=random.Random(1420); toks=[]; key='y'; NJ=[0,0]
for i,(v,K,v2,fl,aus,an) in enumerate(ev):
    junct = bool(aus and an and len(aus)+len(an)>=3)
    Kc = aus if junct else K
    if not Kc:
        nxt=ev[i+1] if i+1<len(ev) else None
        nK=(nxt[4] if (nxt and nxt[4] and nxt[5] and len(nxt[4])+len(nxt[5])>=3) else nxt[1]) if nxt else ''
        allowed={hom_peek(nK)[0]} if nK in HOM else set('ynrl')
        opts=[(p,PC.get(('S',fam(p[1]),p),0)) for p,vv in pair_v.items() if vv==v2 and p[1] in allowed] or [(p,0) for p,vv in pair_v.items() if vv==v2]
        p=max(opts,key=lambda o:(o[1],str(o[0])))[0]; br,nk=p
        toks.append(key+('' if br=='-' else br)); key=nk; continue
    cell=hom_take(Kc) if Kc in HOM else None
    if cell is None: continue
    U=[[cell]]; extra=''
    if junct: extra=vorbau_pool(cell[2],an,cell[1]); NJ[0]+=1
    else:
        pc=pieces(K)
        if pc and len(pc)>1 and len(''.join(pc[1:]))>=2: extra=B3r.get(''.join(pc[1:]),''); NJ[1]+=1
    nxt=ev[i+1] if i+1<len(ev) else None
    nK=(nxt[4] if (nxt and nxt[4] and nxt[5] and len(nxt[4])+len(nxt[5])>=3) else nxt[1]) if nxt else None
    allowed={hom_peek(nK)[0]} if nK in HOM else {'y'}
    cls=gen.hclass(cell[2] or '')
    opts=[(p,PC.get((cls,fam(p[1]),p),0)) for p,vv in pair_v.items() if vv==v2 and p[1] in allowed] or [(p,0) for p,vv in pair_v.items() if vv==v2]
    p=max(opts,key=lambda o:(o[1],str(o[0])))[0]; br,nk=p; tr='' if br=='-' else br
    if extra and tr=='oq' and v2=='e': tr=''
    k0=('s',cell[1],cell[2]) if fl else cell
    tok=k0[0]+k0[1]+k0[2]+extra+tr
    if not ok(tok):
        fixed=False
        if extra:
            for cand in (POOL.get(k0[2]) or ALLP)[:40]:
                if ok(k0[0]+k0[1]+k0[2]+cand+tr): extra=cand; tok=k0[0]+k0[1]+k0[2]+extra+tr; fixed=True; NAHT[0]+=1; break
        if not fixed:
            for p2,vv in sorted(pair_v.items(),key=lambda o:-PC.get((cls,fam(o[0][1]),o[0]),0)):
                if vv!=v2 or p2[1] not in allowed: continue
                tr2='' if p2[0]=='-' else p2[0]
                if ok(k0[0]+k0[1]+k0[2]+extra+tr2): tr,nk=tr2,p2[1]; tok=k0[0]+k0[1]+k0[2]+extra+tr; fixed=True; NAHT[1]+=1; break
        if not fixed and extra: tok=k0[0]+k0[1]+k0[2]+tr; NAHT[2]+=1
    a=_at2(tok)
    if len(a)>=4 and a[-2] in ('l','r') and a[-1] in ('o','a') and (a[-3],a[-2]) in SPLITR:
        key2=(a[-3],a[-2]); SPLITC[key2]+=1
        if (SPLITC[key2]*SPLITR[key2])%1.0 < SPLITR[key2]:
            toks.append(''.join(a[:-2])); tok=''.join(a[-2:]); SPLITN[0]+=1
    if tok in BEW and toks:
        pa=gen.atoms(toks[-1]); pcl=('oq' if len(pa)>=2 and pa[-2]=='o' and pa[-1]=='q' else pa[-1])
        has_closer=pcl in {'ch','sh','oq','d','ckh','cth','cph','cfh'} or (len(pa)>=2 and pa[-2] in 'ktpd' and pa[-1]=='o') or any(''.join(pa).endswith(b) for b in BEW)
        if not has_closer and ''.join(pa) not in BEW: toks[-1]=toks[-1]+tok
        else: toks.append(tok)
    else: toks.append(tok)
    key=nk
lines=gen.wrap(toks)
L3=[]
for li,l in enumerate(lines):
    f,rest=(l.split(' ',1)+[''])[:2]
    if (li*37)%100<37 and f[1:2] in 'ai' and len(f)>2: f='m'+f[1:]
    L3.append((f+' '+rest).strip())
lines=L3
START=['p']*10+['t']*8+['k']*2+['f']*1; out=[]; left=0; prev=None
for l in lines:
    if left==0: left=rng.randint(4,10); first=True
    else: first=False
    left-=1; mk=None
    if first:
        mk=rng.choice(START)
        while mk==prev: mk=rng.choice(START)
    prev=mk if first else None
    out.append(l+(mk or ''))
Path('GENERAT_v140_hartlieb.txt').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('v140: Token',len(toks),'Vorbau-Verlust',VERLUST[0],'abgetrennte Schliesser',SPLITN[0],'Naht Vorbau/Paar/gestrichen',NAHT,'Zellen im Inventar',len(CELLS),'Clustertypen',len(CLU),'Wortfugen-Token',NJ[0],f'({100*NJ[0]/len(toks):.1f} %)','Binnen-Rest-Token',NJ[1],'Blatt-3-Paare',len(FJ))
