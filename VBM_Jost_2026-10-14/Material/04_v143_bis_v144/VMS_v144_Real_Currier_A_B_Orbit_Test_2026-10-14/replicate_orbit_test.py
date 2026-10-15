#!/usr/bin/env python3
"""Independent, auditable rerun of the *narrow* surface-operator assay.

Requires an explicit local Zandbergen–Landini IVTFF ZL3b file from the pinned
GitHub source. It does not download data or assume synthetic VBM plaintext.
Differences from the original connector V8 implementation: Python RNG rather
than JS xorshift seeds and model permutations (expect similar, not identical,
null means/SD). No numerical claim is independently reverified without the
original input file.
"""
import argparse, collections, json, math, random, re, hashlib, pickle, os
from pathlib import Path
FOLIO=re.compile(r'^<(f[^.>\s]+)>\s+<!([^>]+)>')
LINE=re.compile(r'^<(f[^.>\s]+)\.[^,>]+,([^>]+)>\s*(.*)$')
FOLATTR=re.compile(r'\$([A-Z])=([^\s>]+)')
ATOM=re.compile(r'cth|ckh|cph|cfh|ch|sh|.')

def parse(filename, reverse=False):
    page={};sections=collections.defaultdict(list); rejected=collections.Counter()
    for line in Path(filename).read_text(encoding='utf-8').splitlines():
        h=FOLIO.match(line)
        if h: page[h[1]]=dict(FOLATTR.findall(h[2]));continue
        a=LINE.match(line)
        if not a or not re.search(r'[=@+*]P',a[2]):continue
        info=page.get(a[1],{});key=info.get('I','')+info.get('L','')
        if key not in {'HA','HB','PA','BB','SB'}:continue
        # Illegal pieces are null, never removed before predecessor assignment
        raw=a[3].replace('<->','?')
        raw=re.sub(r'<[^>]*>|\[[^\]]*\]|\{[^}]*\}|@\d+;', '?',raw)
        parts=[x for x in re.split(r'[.,\s]+',raw) if x]
        words=[]
        for x in parts:
            if re.fullmatch(r'[a-z]+',x):
                words.append(''.join(reversed(ATOM.findall(x))) if reverse else x)
            else:words.append(None);rejected[key]+=1
        sections[key].append({'folio':a[1], 'quire':info.get('Q','?'),
                              'leaf':re.sub(r'[rv]$','',a[1]), 'words':words})
    return sections,rejected

def discover(lines):
    counts=collections.defaultdict(collections.Counter)
    for line in lines:
        for w in line['words']:
            if w and 4<=len(w)<=10:
                for j in range(1,len(w)-1):
                    counts[w[:j]+'*'+w[j+1:]][w[j]]+=1
    candidates={};n_pairs=collections.Counter()
    for sh, c in counts.items():
        if len(c)!=2 or sum(c.values())<8 or min(c.values())<3:continue
        pair=''.join(sorted(c))
        candidates[sh]=pair;n_pairs[pair]+=1
    return {s:p for s,p in candidates.items() if n_pairs[p]>=3}

def events(lines, families, planted=False):
    ans=[]
    for line in lines:
        words=line['words']
        for i in range(1,len(words)):
            w=words[i];prev=words[i-1]
            if not w or not prev or not 4<=len(w)<=10:continue
            options=[]
            for j in range(1,len(w)-1):
                sh=w[:j]+'*'+w[j+1:];pair=families.get(sh)
                if pair and w[j] in pair:options.append((sh,pair,int(w[j]==pair[1])))
            if not options:continue
            sh,pair,y=min(options)
            suffix=prev[-2:]
            if planted:
                token=f"{line['folio']}:{i}:{w}:{suffix}"
                # Deterministic UTF8 FNV32 to match placement convention
                hh=2166136261
                for code in token.encode():hh=((hh^code)*16777619)&0xffffffff
                if hh/2**32<0.15:y=int(suffix.endswith('y'))
            ans.append((sh,pair,y,suffix,prev))
    return ans

def fit_and_gain(train, test, grouping=None, signs=None):
    sh=collections.defaultdict(lambda:[0.,0.]);sf=collections.Counter()
    flip=lambda e:1-e[2] if signs and signs.get(e[0],False) else e[2]
    for e in train:sh[e[0]][0]+=1;sh[e[0]][1]+=flip(e);sf[e[3]]+=1
    pr={k:(v[1]+1)/(v[0]+2) for k,v in sh.items()}
    top={s for s,_ in sorted(sf.items(),key=lambda a:(-a[1],a[0]))[:6]}
    glob=collections.defaultdict(lambda:[0.,0.]);own=collections.defaultdict(lambda:[0.,0.])
    for e in train:
        f,p,_,s,_=e;v=flip(e);st=s if s in top else 'OTHER';grp=(grouping.get(f,p) if grouping else p)
        for counts,key in ((glob,(grp,st)),(own,(f,st))):
            counts[key][0]+=1;counts[key][1]+=v-pr[f]
    gain=0; num=0
    for e in test:
        f,p,_,s,_=e
        if f not in pr:continue
        y=flip(e);state=s if s in top else 'OTHER';grp=grouping.get(f,p) if grouping else p
        n,res=glob[(grp,state)];own_n,own_res=own[(f,state)]
        q=max(.015,min(.985,pr[f]+(res-own_res)/(n-own_n+16)))
        b=pr[f]
        ll=lambda x,z:x*math.log2(z)+(1-x)*math.log2(1-z)
        gain+=ll(y,q)-ll(y,b);num+=1
    return gain,num

def run(src, reverse=False, leaf=False, planted=False, nulls=60, seed=20261014):
    sections,excluded=parse(src,reverse)
    out={'source_sha256':hashlib.sha256(Path(src).read_bytes()).hexdigest(),
         'representation':'REVA' if reverse else 'EVA', 'holdout':'leaf' if leaf else 'quire_or_folio',
         'planted_15pct':planted,'sections':{},'filtered_parts':dict(excluded)}
    for k,lines in sections.items():
        if k=='PA' or (reverse and k!='SB'):continue
        unit=lambda x:x['quire'] if k in ('HA','HB','PA') else x['leaf'] if leaf else x['folio']
        units=sorted({unit(x) for x in lines});fold=[]
        for u in units:
            tr=[x for x in lines if unit(x)!=u];te=[x for x in lines if unit(x)==u]
            fam=discover(tr);a=events(tr,fam,planted);b=events(te,fam,planted)
            if b:fold.append((fam,a,b,u))
        N=sum(len(f[2]) for f in fold)
        if N<200 or len(fold)<4:
            out['sections'][k]={'status':'INSUFFICIENT','N':N,'folds':len(fold)};continue
        score=sum(fit_and_gain(tr,te)[0] for _,tr,te,_ in fold)/N
        rng=random.Random(seed+sum(map(ord,k)))
        values=[];positive=0
        for fam,tr,te,u in fold:positive+=fit_and_gain(tr,te)[0]>0
        for _ in range(nulls):
            summ=0
            for fam,tr,te,u in fold:
                sh=list(fam)
                flips={key:rng.random()<.5 for key in sh}
                summ+=fit_and_gain(tr,te,signs=flips)[0]
            values.append(summ/N)
        mean=sum(values)/nulls
        sd=(sum((x-mean)**2 for x in values)/(nulls-1))**.5
        out['sections'][k]={'n':N,'holdouts':len(fold),'positive_folds':positive,
            'gain_bits':score,'rotation_null_mean':mean,'rotation_null_sd':sd,
            'effect':score-mean,'z':(score-mean)/sd if sd else None}
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('source');a.add_argument('--reverse',action='store_true');a.add_argument('--leaf',action='store_true');a.add_argument('--planted',action='store_true');a.add_argument('--nulls',type=int,default=60);a.add_argument('--out',default='replication_result.json');args=a.parse_args()
    r=run(args.source,args.reverse,args.leaf,args.planted,args.nulls)
    p=Path(args.out);p.write_text(json.dumps(r,indent=2))
    tmp=p.with_suffix('.pkl.tmp');final=p.with_suffix('.pkl')
    with tmp.open('wb') as f:pickle.dump(r,f,protocol=5);f.flush();os.fsync(f.fileno())
    os.replace(tmp,final)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
