# Panel: dieselben Zahlen fuer VMS-B (Zeilendatei) und jedes Generat. LAM als eigene Schicht gemessen und dann abgezogen.
import sys,collections,math,random
RUN={'e','o','a','i'}; TR={'o','ch','sh','l','d','y','s'}; G={'k','t','p','f'}; BANK={'ch','sh','d','s'}
ATW=('ckh','cth','cph','cfh','ch','sh')
def atoms(x):
    out=[];i=0
    while i<len(x):
        for a in ATW:
            if x.startswith(a,i): out.append(a);i+=len(a);break
        else: out.append(x[i]);i+=1
    return out
def cellparse(a):
    if not a or a[0] not in 'ynrlms': return None
    i=1; sw=''
    if a[0]=='y' and i<len(a) and a[i]=='d': sw='d'; i+=1
    run=[]
    while i<len(a) and a[i] in RUN: run.append(a[i]); i+=1
    body=a[i:]
    if not body: return a[0],sw+''.join(run),'',[],'-'
    head=body[0]; rest=body[1:]
    if rest[-2:]==['o','q']: br='oq'; mid=rest[:-2]
    elif rest and rest[-1] in TR|{'a'}: br=rest[-1]; mid=rest[:-1]
    else: br='-'; mid=rest
    return a[0],sw+''.join(run),head,mid,br
def H(c):
    s=sum(c.values()); return -sum(v/s*math.log2(v/s) for v in c.values()) if s else 0
def Hb(p): return 0 if p in (0,1) else -p*math.log2(p)-(1-p)*math.log2(1-p)
def panel(fn,limit=None,name=''):
    lines=[]
    for n,l in enumerate(open(fn,encoding='utf-8')):
        if limit and n>=limit: break
        t=l.rstrip('\n').split('\t')[-1].split()
        if t: lines.append(t)
    toks_all=sum(len(l) for l in lines)
    # LAM-Schicht: letztes Atom des REVA-letzten Tokens; m-Schluessel am REVA-Anfang
    lam=collections.Counter(); mstart=0
    L2=[]
    for l in lines:
        a=atoms(l[-1]); l=list(l)
        if len(a)>2 and a[-1] in G: lam[a[-1]]+=1; l[-1]=''.join(a[:-1])
        else: lam['-']+=1
        if len(atoms(l[0]))>2 and l[0].startswith('m'): mstart+=1; l=l[1:]
        L2.append(l)
    lines=L2; R={'name':name,'Zeilen':len(lines),'Token':sum(len(l) for l in lines)}
    R['LAM p/t/k/f % der Zeilen']=round(100*(1-lam['-']/len(lines)),1); R['LAM Verteilung']=' '.join(f'{k}{100*v/len(lines):.0f}' for k,v in lam.most_common() if k!='-')
    R['m-Token am REVA-Zeilenanfang %']=round(100*mstart/len(lines),1)
    P=[[cellparse(atoms(t)) for t in l] for l in lines]
    flat=[p for l in P for p in l if p]
    n=len(flat)
    R['ohne Schluessel %']=round(100*(1-n/R['Token']),1)
    R['kopflos %']=round(100*sum(1 for p in flat if not p[2])/n,1)
    R['kopflos Top']=' '.join(f'{k}' for k,_ in collections.Counter(''.join([p[0],p[1],p[2]]+p[3]+([p[4]] if p[4]!='-' else [])) for p in flat if not p[2]).most_common(6))
    vb=[p for p in flat if p[3]]
    R['Vorbau %']=round(100*len(vb)/n,1)
    R['Vorbau mit Lauf %']=round(100*sum(1 for p in vb if any(x in RUN for x in p[3]))/len(vb),1) if vb else 0
    inner=collections.Counter(x for p in vb for x in p[3] if x not in RUN); ti=sum(inner.values())
    R['Vorbau innen l %']=round(100*inner['l']/ti,1) if ti else 0; R['Vorbau innen Bank %']=round(100*sum(inner[x] for x in ('ch','sh','ckh','cth'))/ti,1) if ti else 0
    # Fugentafel: oq->y, a->r, Spaces aus Paar
    trans=collections.defaultdict(lambda:[0,0]); fuge=collections.Counter(); tr=[]
    for l in lines:
        seq=[atoms(t) for t in l]
        for a,b in zip(seq,seq[1:]):
            L='oq' if len(a)>=2 and a[-2]=='o' and a[-1]=='q' else a[-1]; fuge[(L,b[0])]+=1
            trans[(a[-1],b[0])][0]+=1
        for a in seq:
            for x,y in zip(a,a[1:]): trans[(x,y)][1]+=1
    tot=sum(sum(v) for v in trans.values()); nb=sum(v[0] for v in trans.values())
    hc=sum(sum(v)/tot*Hb(v[0]/sum(v)) for v in trans.values()); R['Space aus Paar %']=round(100*(1-hc/Hb(nb/tot)),0)
    for L,K in (('oq','y'),('l','y'),('a','r'),('ch','n')):
        row=sum(v for (x,y),v in fuge.items() if x==L); R[f'Fuge {L}->{K} %']=round(100*fuge[(L,K)]/row,0) if row else None
    # Paar-Restentropie zweiseitig (Einzelzellen)
    ctx=collections.defaultdict(collections.Counter)
    for l in P:
        for i in range(len(l)-1):
            p,q=l[i],l[i+1]
            if p and q and p[2] and q[2] and not p[3] and not q[3]: ctx[((p[1],p[2]),(q[1],q[2]))][(p[4],q[0])]+=1
    big=[c for c in ctx.values() if sum(c.values())>=20]; M=sum(sum(c.values()) for c in big)
    R['Paar H|Zellen bit']=round(sum(H(c)*sum(c.values()) for c in big)/M,2) if M else None
    R['Paar Top %']=round(100*sum(max(c.values()) for c in big)/M,0) if M else None
    # Typen/Hapax auf 20000 Token
    tk=[t for l in lines for t in l][:20000]; c=collections.Counter(tk)
    R['Typen/20000']=len(c); R['Hapax %']=round(100*sum(1 for v in c.values() if v==1)/len(c),1)
    R['WL']=round(sum(len(atoms(t)) for t in tk)/len(tk),2)
    return R
if __name__=='__main__':
    rows=[panel(fn,name=fn.split('/')[-1]) for fn in sys.argv[1:]]
    keys=[k for k in rows[0] if k!='name']
    print(f'{"":34s}'+''.join(f'{r["name"][:22]:>24}' for r in rows))
    for k in keys: print(f'{k:34s}'+''.join(f'{str(r[k]):>24}' for r in rows))
