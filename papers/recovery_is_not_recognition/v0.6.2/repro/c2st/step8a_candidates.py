"""Step 8a: build candidate corpora for the replacement eval harness.
- real_A / real_B : disjoint folio halves of VMS (C2ST sanity -> should be ~0.5 AUC)
- line-shuffle, word-shuffle : trivial controls
- gen_template_v10 : GENUINE output from Ed's generator (build_spec/produce_manuscript)
- delex_char3 : faithful reimpl of the SECTION-2 delexicalised generator (per-section order-3 char model
  + section length profile + line-start opener + no-exact-repeat)
Each candidate stored as (tokens, line_lens) ; line_lens = VMS real line-length sequence imposed on all
so line-length is not a trivial C2ST tell. Atomic+pickled."""
import pickle, random, importlib.util, sys, os
from collections import defaultdict, Counter
recs=pickle.load(open('/tmp/vms/repo/enriched_records.pkl','rb'))
corp=pickle.load(open('/tmp/vms/work/corpus.pkl','rb'))
line_tokens=corp['line_tokens']; line_lens=[len(l) for l in line_tokens]; tokens=corp['tokens']
fol=defaultdict(list)
for r in recs: fol[r['folio']].append(r['token'])
folios=list(fol)

def segment(flat,lens):
    out=[];i=0
    for L in lens:
        out.append(flat[i:i+L]); i+=L
        if i>=len(flat): break
    return out

cand={}
# real A/B
rng=random.Random(0); fs=folios[:]; rng.shuffle(fs); h=len(fs)//2
A=[t for f in fs[:h] for t in fol[f]]; B=[t for f in fs[h:] for t in fol[f]]
cand['real_A']=(A,line_lens); cand['real_B']=(B,line_lens)
# controls
r=random.Random(1); lt=[list(l) for l in line_tokens]; r.shuffle(lt)
cand['line-shuffle']=([t for l in lt for t in l],line_lens)
r=random.Random(2); ws=tokens[:]; r.shuffle(ws); cand['word-shuffle']=(ws,line_lens)

# genuine generator: gen_template_v10
gp='/tmp/vms/repo/Paper/Generators/gen_template_v10.py'
spec_mod=importlib.util.spec_from_file_location("gv10",gp)
mod=importlib.util.module_from_spec(spec_mod)
sys.argv=['x']  # guard
spec_mod.loader.exec_module(mod)
import inspect
bs_params=inspect.signature(mod.build_spec).parameters
kw={}
if 'p70c_path' in bs_params: kw['p70c_path']='/tmp/vms/repo/Paper/p70c_full_spec_v1.json'
if 'records_path' in bs_params: kw['records_path']='/tmp/vms/repo/enriched_records.pkl'
try:
    gspec=mod.build_spec(**kw)
    g10=mod.produce_manuscript(gspec,n_tokens=len(tokens),seed=42)
    cand['gen_template_v10']=(list(g10),line_lens)
    print("gen_template_v10:",len(g10),"tokens generated")
except Exception as e:
    print("gen_template_v10 FAILED:",repr(e)[:200])

# faithful SECTION-2 delexicalised generator
sec_tokens=corp['sec_tokens']; sec_lines=corp['sec_lines']
GALLOWS=set('pfkt')
def build_charmodel(toks):
    m=defaultdict(Counter); lens=Counter()
    for t in toks:
        s='^^^'+t+'$'; lens[len(t)]+=1
        for i in range(3,len(s)): m[s[i-3:i]][s[i]]+=1
    return m,lens
def sample_word(m,lens,rng,opener=False,maxlen=14):
    # target length from section profile (opener -> bias longer)
    Ls=list(lens); ws=[lens[L] for L in Ls]
    L=rng.choices(Ls,weights=ws)[0]
    if opener: L=max(L,rng.choices(Ls,weights=ws)[0])  # draw twice, take longer
    out=''; ctx='^^^'; tries=0
    while True:
        d=m.get(ctx)
        if not d: break
        nxt=rng.choices(list(d),weights=list(d.values()))[0]
        if nxt=='$':
            if len(out)>=2 or tries>6: break
            tries+=1; continue
        out+=nxt; ctx=(ctx+nxt)[-3:]
        if len(out)>=maxlen: break
    if opener and out and out[0] not in GALLOWS:
        # prefer a gallows-initial realisation
        out=rng.choice(list(GALLOWS))+out
    return out or 'okeey'
def gen_delex(seed=7):
    rng=random.Random(seed); models={s:build_charmodel(sec_tokens[s]) for s in sec_tokens}
    # section token budget proportional to real
    out=[]; prev=None
    # walk sections in real proportion, line by line
    for s in sec_tokens:
        m,lens=models[s]; nlines=len(sec_lines[s])
        reall=[len(l) for l in sec_lines[s]]
        for li in range(nlines):
            L=reall[li]
            for pos in range(L):
                w=sample_word(m,lens,rng,opener=(pos==0))
                k=0
                while w==prev and k<5: w=sample_word(m,lens,rng,opener=(pos==0)); k+=1
                out.append(w); prev=w
    return out
dx=gen_delex(7)
cand['delex_char3']=(dx,line_lens)
print("delex_char3:",len(dx),"tokens generated")

pickle.dump(cand,open('/tmp/vms/work/candidates.pkl','wb'))
print("OK -> candidates.pkl ; candidates:",list(cand))
