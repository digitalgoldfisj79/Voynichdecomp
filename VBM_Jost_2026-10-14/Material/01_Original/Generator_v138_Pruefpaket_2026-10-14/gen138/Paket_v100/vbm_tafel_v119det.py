#!/usr/bin/env python3
"""VBM Tafel-Schreiber v93 (2026-09-28) - v92 + Blatt 1 fuehrt nur Einheiten bis 2 Konsonanten; laengere werden geteilt und angehaengt (Vorbau) - REVA. v91 + (1) freie Wahl unter Homophonen (Anweisung "nimm eines beliebig", Zufall mit festem Saatwert), (2) Blatt 2 nach KOEPFEN (Zeile davor = Grundglyphe, nicht Klasse).
Blatt 1: Zellen = alle VMS-Kerne (Schluessel, Zustand, Grundglyphe) mit >= 20 Belegen, ueber alle Bloecke.
         Fuellung: deutsche Cluster nach Masse, je Cluster so viele Zellen wie sein Massenanteil verlangt (PLATZHALTER).
Blatt 2: Vokaltafel: Vokal x Klasse der Einheit DAVOR x Klasse der Einheit DANACH -> Paare (Traeger, Schluessel).
         Paare und ihre Gewichte aus den VMS-Fugen; Zuordnung Paar -> Vokal je Zelle nach bairischer Vokalverteilung
         in derselben Klassenumgebung (Massenpassung je Zelle, PLATZHALTER).
Wahl unter Zellen/Paaren: nach vorigem und naechstem Vokal (ohne Gedaechtnis). Keine Wortenden, kein Zaehler.
Aufruf: python3 vbm_tafel_v91.py QUELLE AUSGABE [--vms vms_lines_REVA.txt]
Schreibt zusaetzlich Vokaltafel_v116c.csv und Zelltafel_v116c.csv.
"""
import re, sys, unicodedata, collections, csv, random
random.seed(1420)
from pathlib import Path
ATW=('ckh','cth','cph','cfh','ch','sh')
def atoms(s):
    out=[];i=0
    while i<len(s):
        for a in ATW:
            if s.startswith(a,i): out.append(a);i+=len(a);break
        else: out.append(s[i]);i+=1
    return out
VOW=set('aeiou'); RUN={'e','o','a','i'}; TR={'o','ch','sh','l','d','y','s'}
TWO={'chk','cht','chp','chf','kl','chl','shl','shk','ko','ke'}
GAL={'k','t','p','f','kl','ko','ke'}; BANK={'ch','sh','chl','shl'}; BGAL={'ckh','cth','cph','cfh','chk','cht','chp','chf','shk'}
def hclass(h): return 'G' if h in GAL else 'B' if h in BANK else 'X' if h in BGAL else ('E' if h=='' else 'S')
def norm_word(w):
    w=unicodedata.normalize('NFD',w.lower()); w=''.join(c for c in w if unicodedata.category(c)!='Mn')
    w=w.replace('ß','ss').replace('v','u').replace('y','i').replace('j','i')
    w=re.sub(r'ei|ey|ai|ay','e',w).replace('ie','i'); w=re.sub(r'au|eu|ou','u',w); w=re.sub(r'([bcdfghklmnpqrstwz])\1+',r'\1',w)
    return re.sub(r'[^a-z]','',w)
def words_of(t): return [('\u00a7' if re.fullmatch(r'unde?',w) else w) for x in re.findall(r'[A-Za-zÄÖÜäöüß]+',t) if (w:=norm_word(x))]
def events(words):
    s=''.join(words); vi=[i for i,c in enumerate(s) if c in VOW]
    return [(s[p],s[p+1:q],s[q]) for p,q in zip(vi,vi[1:])]
def parse(t):
    a=atoms(t)
    if a[0] not in 'ynrlms': return None
    key=a[0]; i=1; sw=''
    if key=='y' and i<len(a) and a[i]=='d': sw='d'; i+=1
    run=[]
    while i<len(a) and a[i] in RUN: run.append(a[i]); i+=1
    tr=''
    if len(a)>=2 and a[-2]=='o' and a[-1]=='q': tr='oq'; end=len(a)-2
    elif a[-1] in TR and len(a)-1>i: tr=a[-1]; end=len(a)-1
    else: end=len(a)
    body=a[i:end]
    if not body: head=''
    elif len(body)>=2 and body[0]+body[1] in TWO: head=body[0]+body[1]
    else: head=body[0]
    return key,sw+''.join(run),head,tr
VI={'a':0,'e':1,'i':2,'o':3,'u':4}
class Blatt1:
    """Blatt 1 v95: eine ZEILE = (Zustand, Grundglyphe), quer ueber alle Familienbloecke.
    Ein Cluster bekommt ganze Zeilen. Zeilen ohne Grundglyphe sind fuer die leere Einheit reserviert."""
    def __init__(self,vms_lines,clustermass,clusterprof=None):
        cells=collections.Counter()
        for l in open(vms_lines,encoding='utf-8'):
            for t in l.rstrip('\n').split('\t')[-1].split():
                p=parse(t)
                if p: cells[(p[0],p[1],p[2])]+=1
        rows=collections.defaultdict(list)
        for (k,st,h),n in cells.items():
            if n>=5: rows[(st,h)].append((k,st,h))
        rowmass={r:sum(cells[c] for c in cs) for r,cs in rows.items()}
        self.empty=[c for r,cs in rows.items() if r[1]=='' for c in cs for _ in range(max(1,round(cells[c]/25)))]
        R=[r for r in rows if r[1] not in ('','h') and rowmass[r]>=20]
        R.sort(key=lambda r:-rowmass[r])
        # v98: jede Clusterspalte beginnt mit einer Zeile, die alle Familienbloecke hat
        full=[r for r in R if len(rows[r])>=3]; rest=[r for r in R if len(rows[r])<3]
        # v104: Spalten fuer Gruppen aller Laengen, hoechstens zwei Zeilen je Spalte -
        # so reichen die Zeilen fuer mehr Gruppen und der Vorbau wird seltener.
        short=dict(clustermass); tot=sum(short.values())
        # Zeilenprofil: oq/(oq+o) am Tokenende je (Zustand,Kopf) aus dem VMS
        rp=collections.defaultdict(lambda:[0,0])
        for l in open(vms_lines,encoding='utf-8'):
            for t in l.rstrip('\n').split('\t')[-1].split():
                p=parse(t); a=atoms(t)
                if not p: continue
                if len(a)>=2 and a[-2]=='o' and a[-1]=='q': rp[(p[1],p[2])][0]+=1
                elif a[-1]=='o' and len(a)>1: rp[(p[1],p[2])][1]+=1
        rowprof={r:v[0]/(v[0]+v[1]) for r,v in rp.items() if v[0]+v[1]>=10}
        EHEADS={'k','chk','ckh'}
        self.rowclass={r:('E' if rowprof.get(r, 1.0 if r[1] in EHEADS else 0.0)>=0.5 else 'I') for r in rows}
        self.rowE={r:rowprof.get(r, 0.66 if r[1] in EHEADS else 0.35) for r in rows}
        cp=clusterprof or {}
        cm=sum(rowmass[r] for r in R); self.map={}; self.row_of={}
        full=list(full); rest=list(rest)
        def pick(pool,prof,W=10):
            w=pool[:W]
            if prof<0.68:
                j=min(range(len(w)),key=lambda k:abs(rowprof.get(w[k],0.7)-prof))
            else: j=0
            return pool.pop(j)
        for c in sorted(short,key=lambda x:-short[x]):
            if not full and not rest: break
            need=short[c]/tot*cm; got=0; mine=[]; prof=cp.get(c,0.7)
            if full: r=pick(full,prof); mine.append(r); got+=rowmass[r]
            while rest and got<need and len(mine)<6:
                have={self.rowclass[x] for x in mine}
                if len(have)==1:
                    want='I' if 'E' in have else 'E'
                    w=[x for x in rest[:12] if self.rowclass[x]==want]
                    if w: r=w[0]; rest.remove(r); mine.append(r); got+=rowmass[r]; continue
                r=pick(rest,prof); mine.append(r); got+=rowmass[r]
            if not mine and rest: mine.append(rest.pop(0))
            mine.sort(key=lambda r:-rowmass[r])
            self.map[c]=[cell for r in mine for cell in rows[r] for _ in range(max(1,round(cells[cell]/25)))]
            for r in mine: self.row_of[r]=c

        self.miss=0; self.counts=cells
    def units(self,K):
        if not K: return [self.empty]
        if K in self.map: return [self.map[K]]
        for k in range(len(K)-1,0,-1):
            if K[:k] in self.map and K[k:] in self.map: return [self.map[K[:k]],self.map[K[k:]]]
        K2=self.norm(K)
        if K2!=K and K2 in self.map: return [self.map[K2]]
        out=[];r=K2
        while r:
            for k in range(len(r),0,-1):
                if r[:k] in self.map: out.append(self.map[r[:k]]); r=r[k:]; break
            else: self.miss+=1; return None
        return out
    @staticmethod
    def norm(K):
        K=K.replace('sch','\x01').replace('ch','\x02')
        K=K.replace('ck','k').replace('tz','z').replace('pf','f').replace('x','ks').replace('q','k').replace('c','k')
        return K.replace('\x01','sch').replace('\x02','ch')
class Blatt2:
    """Vokal x Klasse davor x Klasse danach -> Liste von (Traeger, Schluessel) mit Gewicht"""
    def __init__(self,vms_lines,de_events,b1):
        pairs=collections.defaultdict(collections.Counter)
        for l in open(vms_lines,encoding='utf-8'):
            ts=l.rstrip('\n').split('\t')[-1].split(); P=[parse(t) for t in ts]
            for i in range(len(P)-1):
                if not P[i] or not P[i+1]: continue
                tr=P[i][3]; key=P[i+1][0]
                pairs[(P[i][2] or '',hclass(P[i+1][2]))][(tr,key)]+=1
        # deutsche Vokalverteilung je Klassenumgebung, Klassen ueber die Zellfuellung von Blatt 1
        def cls_of(K):
            u=b1.units(K) if K else [b1.empty]
            if not u: return None
            return hclass(u[0][0][2])
        def head_of(K):
            u=b1.units(K) if K else [b1.empty]
            if not u: return None
            return u[0][0][2] or ''
        dev=collections.defaultdict(collections.Counter)
        for i in range(1,len(de_events)):
            v,K,v2=de_events[i-1]; cb=head_of(K); ca=cls_of(de_events[i][1])
            if cb is not None and ca: dev[(cb,ca)][v2]+=1
        self.table={}
        for env,pc in pairs.items():
            vd=dev.get(env) or dev.get(('k','G')) or collections.Counter({'e':38,'i':22,'a':18,'u':15,'o':7})
            plist=[(p,n) for p,n in pc.most_common() if n>=5] or pc.most_common(3)
            totp=sum(n for _,n in plist); totv=sum(vd.values()) or 1; row={v:[] for v in 'aeiou'}
            q=max(1,totp//400)
            # v102: Spaltenzuordnung nach Tafel I - der Traeger bestimmt die Vokalspalte
            # (E=oq, I=o, A=ch|sh, O=l, U=d|r|k|t); traegerlose Paare fuellen nach Restbedarf auf.
            CLS={'oq':'e','o':'i','ch':'a','sh':'a','l':'o','d':'u','r':'u','k':'u','t':'u'}
            need={v:vd.get(v,0)/totv*totp for v in 'aeiou'}; got={v:0 for v in 'aeiou'}
            cnt={v:{} for v in 'aeiou'}; rest=[]
            for p,n in plist:
                v=CLS.get(p[0])
                if v: cnt[v][p]=cnt[v].get(p,0)+n; got[v]+=n
                else: rest.append((p,n))
            for p,n in sorted(rest,key=lambda x:-x[1]):
                v=max('aeiou',key=lambda x:need[x]-got[x])
                cnt[v][p]=cnt[v].get(p,0)+n; got[v]+=n
            # v103: Ausgleich - traegt eine Spalte ihre deutsche Vokalmasse nicht, wandern ganze
            # Paartypen aus der Ueberschussspalte hinein (Umgebungs-Sonderform, z.B. E nach t = o.x).
            for _ in range(300):
                d=max('aeiou',key=lambda x:need[x]-got[x]); su=max('aeiou',key=lambda x:got[x]-need[x])
                dv=need[d]-got[d]; sv=got[su]-need[su]
                if dv<=totp*0.004 or sv<=0 or not cnt[su]: break
                p=min(cnt[su],key=lambda x:abs(cnt[su][x]-dv))
                n=cnt[su].pop(p); cnt[d][p]=cnt[d].get(p,0)+n; got[d]+=n; got[su]-=n
            for v in 'aeiou':
                row[v]=[p for p,n in cnt[v].items() for _ in range(max(1,round(n/q)))]
            for v in 'aeiou':
                if not row[v]: row[v]=[plist[0][0]]
            self.table[env]=row
        self.dev=dev
    def pair(self,v,cb,ca,sel,allowed_keys):
        row=self.table.get((cb,ca)) or self.table.get(('k','G')) or next(iter(self.table.values()))
        L=[p for p in row[v] if p[1] in allowed_keys] or row[v]   # v112: Spaltentreue, kein Ausweich in fremde Vokalspalten
        return mode(L)                                          # "nimm eines beliebig"
DAVOR=0
def mode(L):
    c=collections.Counter(L); m=max(c.values())
    for x in L:
        if c[x]==m: return x

def encode(ev,b1,b2):
    ev=[(v,K.replace('\u00a7',''),v2,'\u00a7' in K) for v,K,v2 in ev]
    U=[b1.units(K) for v,K,v2,fl in ev]
    toks=[]; key='y'
    for i,(v,K,v2,fl) in enumerate(ev):
        if U[i] is None: toks.append('?'); key='y'; continue
        sel=VI[v]*5+VI[v2]
        cand=[c for c in U[i][0] if c[0]==key] or [c for c in U[i][0] if c[0] not in 'sm'] or U[i][0]
        want='E' if v2=='e' else ('I' if v2=='i' else None)   # v116: Zeile nach Folgevokal waehlen
        if want:
            w=[(c, b1.rowE.get((c[1],c[2]),0.5)) for c in cand]
            cand=[c for c,e in w for _ in range(max(1,round(10*(e if want=='E' else 1-e))))]
        if DAVOR and v in 'ei':
            c3=[c for c in cand if ('e' in c[1])==(v=='e')]
            if c3: cand=c3
        cell=mode(cand)
        extra=''
        for u in U[i][1:]:
            m=min(len(c[1]) for c in u)                     # v105: im Vorbau eine strichkurze Zeile (bis 1 Strich mehr)
            c2=mode([c for c in u if len(c[1])<=m+1])
            extra+=c2[1]+c2[2]
        if i+1<len(ev) and U[i+1]:
            allowed={c[0] for c in U[i+1][0]}-{'s','m'} or {'y'}; ca=hclass(U[i+1][0][0][2])
        else: allowed={'y'}; ca='G'
        tr,nk=b2.pair(v2,cell[2] or '',ca,sel,allowed)
        if extra and tr=='oq' and v2=='e': tr=''    # v107: nach Vorbau entfaellt die E-Bruecke (VMS: 30% traegerlos nach Vorbau, oq nur 8%)
        if fl:
            # und = s anstelle des Schluessels; s-Blockform bevorzugt; nie so, dass s mit dem Rest verschmilzt
            cs=[c for c in U[i][0] if c[0]=='s'] or [cell]
            c2=mode(cs); toks.append('s'+c2[1]+c2[2]+extra+tr)
        else:  toks.append(cell[0]+cell[1]+cell[2]+extra+tr)
        key=nk
    return toks
def wrap(tokens, widths=(60,64,68,62,66)):
    lines=[];cur=[];chars=0;wi=0
    for w in tokens:
        add=len(w)+(1 if cur else 0)
        if cur and chars+add>widths[wi%len(widths)]: lines.append(' '.join(cur)); cur=[];chars=0;wi+=1;add=len(w)
        cur.append(w); chars+=add
    if cur: lines.append(' '.join(cur))
    return lines
def tafel_fingerprint(b1,b2):
    import hashlib
    h=hashlib.md5()
    for K in sorted(b1.map): h.update((K+'|'+';'.join(map(str,b1.map[K]))).encode())
    h.update(str(sorted(map(str,b1.empty))).encode())
    for env in sorted(b2.table):
        for v in 'aeiou': h.update((str(env)+v+';'.join(map(str,b2.table[env][v]))).encode())
    return h.hexdigest()

CRIB=int(__import__('os').environ.get('CRIB','1'))
def main(bau,anwend,out,vms):
    ev_bau=events(words_of(Path(bau).read_text(encoding='utf-8',errors='ignore')))
    ev2b=[(v,K.replace('\u00a7',''),v2) for v,K,v2 in ev_bau]
    cm=collections.Counter(K for v,K,v2 in ev2b if K)
    _prof=collections.defaultdict(lambda:[0,0])
    for v,K,v2 in ev2b:
        if v2=='e': _prof[K][0]+=1
        elif v2=='i': _prof[K][1]+=1
    prof={K:(a/(a+b) if a+b else 0.7) for K,(a,b) in _prof.items()}
    b1=Blatt1(vms,cm,prof)
    if CRIB:
        dek=[c for c in b1.counts if c[1]=='de' and c[2]=='k']
        old=b1.row_of.get(('de','k'))
        if old and old in b1.map: b1.map[old]=[c for c in b1.map[old] if not (c[1]=='de' and c[2]=='k')] or b1.map[old]
        b1.map['n']=[c for c in dek for _ in range(max(1,round(b1.counts[c]/25)))]
        b1.row_of[('de','k')]='n'
        print('crib dek=n gesetzt, vorher',old)
    b2=Blatt2(vms,ev2b,b1)
    print('tafel_fingerprint',tafel_fingerprint(b1,b2))
    ev=events(words_of(Path(anwend).read_text(encoding='utf-8',errors='ignore')))
    random.seed(1420)
    toks=encode(ev,b1,b2); lines=wrap(toks)
    lines=[(l.split(' ',1)[0].replace('niia','ma',1)+(' '+l.split(' ',1)[1] if ' ' in l else '')) if l.startswith('niia') else l for l in lines]
    # v114 Gliederungsspur (Layoutschicht NACH der Chiffre):
    # Einheitenlaengen aus der ZL-Absatzstruktur (absatz_layout.json);
    # Erstzeile: Anfangsmarke ans REVA-Zeilenende (Staffel p9 t4 k3 f1 -3),
    # Fortsetzungszeile: Staffel (-7 s1). Der Leser ueberliest Randmarken.
    import json as _json
    _lens=_json.load(open('absatz_layout.json'))['lens']
    START=['p']*9+['t']*4+['k']*3+['f']*1+[None]*3
    CONT=[None]*7+['s']
    out2=[]; left=0; firsts=0
    for l in lines:
        if left==0: left=random.choice(_lens); first=True
        else: first=False
        left-=1
        mk=random.choice(START if first else CONT)
        firsts+=first
        out2.append(l+(mk or ''))
    lines=out2
    print(f"layout: einheiten={firsts}")

    Path(out).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    n=len(toks); q=sum(t=='?' for t in toks)
    print(f"tokens={n} types={len(set(toks))} zeilen={len(b1.row_of)} cluster_belegt={len(b1.map)} ohne_zelle={q} ({100*q/n:.1f}%)")

if __name__=="__main__":
    args=[x for x in sys.argv[1:] if not x.startswith('--') and x!=(sys.argv[sys.argv.index('--vms')+1] if '--vms' in sys.argv else None)]
    vms=sys.argv[sys.argv.index('--vms')+1] if '--vms' in sys.argv else 'vms_B_REVA.txt'
    main(args[0],args[1],args[2],vms)
