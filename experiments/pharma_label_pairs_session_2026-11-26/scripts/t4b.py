import csv, random, re, statistics as st, json
random.seed(4)
C=list(csv.DictReader(open('data/concordance_upload.csv')))
ORDER=['f88r','f88v','f89r1','f89r2','f89v2','f89v1','f99r','f99v','f100r','f100v','f101r','f101v','f102r1','f102r2','f102v2','f102v1']
def norm(s):
    s=re.sub(r'\[([^:\]]*):[^\]]*\]',r'\1',s); s=re.sub(r'[{}?,.!<>0-9 ]','',s); return s
lab={}; page={}
for r in C:
    if r['locus_type']=='Lf' and r['fragment_num']:
        lab.setdefault(r['fragment_num'],[]).append(norm(r['label_zl'])); page[r['fragment_num']]=r['page']
def lev(a,b):
    d=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        p,d[0]=d[0],i
        for j,cb in enumerate(b,1):
            p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(ca!=cb))
    return d[-1]
def sim(a,b): return 1-lev(a,b)/max(len(a),len(b),1)
def fsim(f,g): return max(sim(x,y) for x in lab[f] for y in lab[g])
pairs=[('71','103'),('88','98'),('135','132'),('40','124'),('87','80'),('95','103'),('78','94'),('15','16'),('131','134'),('186','187'),('71','95')]
obs=[fsim(a,b) for a,b in pairs]; S=sum(obs)/len(obs)
frs=list(lab)
def pd(f,g): return abs(ORDER.index(page[f])-ORDER.index(page[g]))
null=[]
for _ in range(10000):
    v=[]
    for a,b in pairs:
        d=pd(a,b)
        while True:
            x,y=random.sample(frs,2)
            if pd(x,y)==d: break
        v.append(fsim(x,y))
    null.append(sum(v)/len(v))
m,s=st.mean(null),st.pstdev(null)
print(json.dumps(dict(pairs=pairs,labels=[(lab[a],lab[b]) for a,b in pairs],pair_sim=[round(x,3) for x in obs],S=round(S,3),null_mean=round(m,3),null_sd=round(s,3),
    ratio=round((S-m)/s,2),clear_2sd=round(m+2*s,3),identical=any(x==1 for x in obs),p_ge=sum(1 for x in null if x>=S)/len(null))))
# planted power: second label = first with k random edits
alph='acdehiklmnopqrsty'
def mutate(w,k):
    w=list(w)
    for _ in range(k):
        op=random.choice('sdi'); i=random.randrange(len(w)) if w else 0
        if op=='s' and w: w[i]=random.choice(alph)
        elif op=='d' and len(w)>1: w.pop(i)
        else: w.insert(i,random.choice(alph))
    return ''.join(w)
for k in range(7):
    det=0
    for _ in range(500):
        v=[sim(lab[a][0],mutate(lab[a][0],k)) for a,b in pairs]
        det+= (sum(v)/len(v))>=m+2*s
    print('planted k',k,'detect rate',det/500)
