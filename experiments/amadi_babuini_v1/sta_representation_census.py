# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,hashlib,json,os,re,subprocess,tempfile,urllib.request
import amadi_residuals_v1 as ar

STA_URL='https://voynich.nu/data/sta/RF1b.txt'
STA_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'
BIT_URL='https://www.voynich.nu/software/bitrans/bitrans.c'
BIT_SHA='3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a'
MAP_URL='https://www.voynich.nu/software/bitrans/STA-aaa.bit'
MAP_SHA='622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64'
HEAD={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36','Referer':'https://voynich.nu/transcr.html','Accept':'text/plain,*/*;q=0.8'}
ar.HEADERS={
    'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}
STA_RE=re.compile(r'[A-Z][0-9a-z]')

def get(url,want):
    req=urllib.request.Request(url,headers=HEAD)
    b=urllib.request.urlopen(req,timeout=120).read();got=hashlib.sha256(b).hexdigest()
    if got!=want:raise RuntimeError((url,got,want))
    return b

def locus(txt):
    for ln in txt.splitlines():
        if not ln.startswith('<') or ln.startswith('#'):continue
        m=re.match(r'<([^>]+)>\s+(.*)$',ln)
        if not m:continue
        loc,body=m.groups()
        if body.startswith('<!'):continue
        page=loc.split('.')[0];body=re.sub(r'\[[^\]]*\]','.',body);body=re.sub(r'<[^>]*>','.',body);body=body.replace(',','')
        yield page,body

def parse_sta(txt):
    p=collections.defaultdict(list)
    for pg,b in locus(txt):
        for ch in b.split('.'):
            z=STA_RE.findall(ch)
            if z:p[pg].append(z)
    return dict(p)

def aaa_units(s):
    out=[];i=0
    while i<len(s):
        if i+1<len(s) and s[i].islower() and s[i+1].isdigit():
            u=s[i:i+2];i+=2
            while i<len(s) and s[i]==':' and i+2<len(s) and s[i+1].islower() and s[i+2].isdigit():u+=':'+s[i+1:i+3];i+=3
            out.append(u)
        else:i+=1
    return out

def parse_aaa(txt):
    p=collections.defaultdict(list)
    for pg,b in locus(txt):
        for ch in b.split('.'):
            z=aaa_units(ch)
            if z:p[pg].append(z)
    return dict(p)

def census(pages,folios):
    C=collections.Counter();words=[]
    for f in folios:
        for w in pages.get(f,[]):C.update(w);words.append(w)
    ordered=sorted(C,key=lambda x:(-C[x],x));tot=sum(C.values())
    rows={}
    for K in [19,22,26,36,61,70,89,100,112,124,136,150,166]:
        V=set(ordered[:K]);occ=sum(C[x] for x in V)/max(1,tot);kw=sum(all(x in V for x in w) for w in words);kc=sum(len(w) for w in words if all(x in V for x in w));allc=sum(map(len,words))
        rows[K]={'occurrence_coverage':occ,'word_coverage':kw/max(1,len(words)),'whole_word_char_coverage':kc/max(1,allc),'types_selected':min(K,len(ordered))}
    return {'types':len(C),'events':tot,'words':len(words),'top':ordered[:20],'rows':rows}

def main():
    sb=get(STA_URL,STA_SHA);bb=get(BIT_URL,BIT_SHA);mb=get(MAP_URL,MAP_SHA);td=tempfile.mkdtemp(prefix='babsta_');sp=os.path.join(td,'RF.txt');bp=os.path.join(td,'bitrans.c');mp=os.path.join(td,'STA-aaa.bit');ap=os.path.join(td,'RF.aaa.txt');open(sp,'wb').write(sb);open(bp,'wb').write(bb);open(mp,'wb').write(mb);exe=os.path.join(td,'bitrans');subprocess.run(['gcc','-O2','-o',exe,bp],check=True);p=subprocess.run([exe,'-1','-m2','-f',mp,sp,ap],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert p.returncode==0,p.stderr
    sta=parse_sta(sb.decode('utf-8','replace'));aaa=parse_aaa(open(ap,encoding='utf-8').read());rf,_=ar.parse_rf();T,H,C1,H2,C2=ar.target_split(rf);fit=T+H
    out={'fit_folios':len(fit),'sealed_old_C2_folios':len(C2),'sta':census(sta,fit),'aaa':census(aaa,fit),'source_sha':{'STA':STA_SHA,'bitrans':BIT_SHA,'map':MAP_SHA}}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
