#!/usr/bin/env python3
"""
REF15-NEUROCIPHER-CAL1
Historical-German positive control for the validated ACL2019 NeuroCipher refactor.

Frozen algorithm: imports research/neurodecipher_acl2019_refactor.py unchanged.
Only the corpus adapter changes.

Discovery geometry mirrors the official noisy-Ugaritic benchmark:
  lost vocab 735, known vocab 4103, exactly 221 true cognate-bearing lost forms.
Gold cognacy = exact ReF lemma + HiTS POS + morphology.
Lost side = 15th-c Alemannic manuscript spellings under one fixed hidden
monoalphabetic character substitution.
Known side = independent 15th-c Bavarian/Austrian manuscript spellings.

Checkpoint selection sees only the 221 discovery cognates.
Sealed final evaluation expands to top 3000 Alemannic x top 10000 Bavarian forms,
and reports both all gold-bearing forms and novel lost types not in discovery.
"""
import argparse, collections, io, json, random, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np, torch

import neurodecipher_acl2019_refactor as nd

REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
CIPHER_SEED=20261005
DISC_LOST=735
DISC_KNOWN=4103
DISC_COG=221
FULL_LOST=3000
FULL_KNOWN=10000

def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def header(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None);d={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,v=line.split(":",1);d[k.strip().lower()]=v.strip()
    return d

def parse_ref():
    print("REF15_DOWNLOAD",flush=True)
    rb=urllib.request.urlopen(REF_URL,timeout=300).read()
    tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
    freq={s:collections.Counter() for s in ("BAV","ALEM")}
    ann={s:collections.defaultdict(set) for s in ("BAV","ALEM")}
    docs=collections.Counter()
    for name in [n for n in tar.getnames() if n.endswith(".xml")]:
        try:root=ET.fromstring(tar.extractfile(name).read())
        except:continue
        md=header(root);med=md.get("medium","").lower();tm=md.get("time","").lower();area=md.get("language-area","").lower()
        if "handschrift" not in med or not tm.startswith("15,"):continue
        bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
        alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
        if not(bav or alem):continue
        side="BAV" if bav else "ALEM";docs[side]+=1
        for tok in root.iter():
            if tok.tag.split("}")[-1]!="token":continue
            for x in [z for z in tok if z.tag.split("}")[-1] in ("tok_anno","mod")]:
                le=child(x,"lemma");pos=child(x,"pos");mor=child(x,"morph") or child(x,"inflection") or "--"
                form=(x.attrib.get("ascii") or x.attrib.get("utf") or x.attrib.get("trans") or "").strip().lower()
                if not(form and le and le not in ("--","[!]") and pos and not pos.startswith("$")):continue
                # Keep alphabetic historical word forms only, matching the word-decipherment task.
                if not all(ch.isalpha() for ch in form):continue
                freq[side][form]+=1
                ann[side][form].add((le,pos,mor))
    return freq,ann,docs

def gold_map(lost,known,ann_l,ann_k):
    inv=collections.defaultdict(set)
    for w in known:
        for key in ann_k[w]:inv[key].add(w)
    out={}
    for w in lost:
        z=set()
        for key in ann_l[w]:z |= inv.get(key,set())
        if z:out[w]=z
    return out

def build():
    freq,ann,docs=parse_ref()
    full_lost=[w for w,n in freq["ALEM"].most_common(FULL_LOST)]
    full_known=[w for w,n in freq["BAV"].most_common(FULL_KNOWN)]
    disc_known=[w for w,n in freq["BAV"].most_common(DISC_KNOWN)]
    gdisc_all=gold_map([w for w,n in freq["ALEM"].most_common()],disc_known,ann["ALEM"],ann["BAV"])
    ranked=[w for w,n in freq["ALEM"].most_common()]
    cog=[w for w in ranked if w in gdisc_all][:DISC_COG]
    non=[w for w in ranked if w not in gdisc_all][:DISC_LOST-DISC_COG]
    assert len(cog)==DISC_COG and len(non)==DISC_LOST-DISC_COG,(len(cog),len(non))
    # Interleave by original frequency rank rather than putting cognates first.
    chosen=set(cog+non)
    disc_lost=[w for w in ranked if w in chosen][:DISC_LOST]
    assert len(disc_lost)==DISC_LOST
    disc_gold=gold_map(disc_lost,disc_known,ann["ALEM"],ann["BAV"])
    assert len(disc_gold)==DISC_COG,len(disc_gold)

    chars=sorted(set("".join(full_lost)))
    pool=[chr(0x0400+i) for i in range(len(chars))]
    rng=random.Random(CIPHER_SEED);rng.shuffle(pool)
    cmap=dict(zip(chars,pool))
    enc=lambda w:"".join(cmap[ch] for ch in w)

    d_lost=[enc(w) for w in disc_lost]
    dgold={enc(w):set(v) for w,v in disc_gold.items()}
    fc_lost=[enc(w) for w in full_lost]
    fgold0=gold_map(full_lost,full_known,ann["ALEM"],ann["BAV"])
    fgold={enc(w):set(v) for w,v in fgold0.items()}
    reverse={enc(w):w for w in full_lost}

    corp=nd.Corpus(sorted(d_lost),sorted(disc_known),dgold)
    audit={"docs":dict(docs),"tokens":{s:int(sum(freq[s].values())) for s in freq},
      "types":{s:len(freq[s]) for s in freq},"disc_lost":len(d_lost),"disc_known":len(disc_known),
      "disc_gold":len(dgold),"full_lost":len(fc_lost),"full_known":len(full_known),
      "full_gold":len(fgold),"full_novel_gold":sum(w not in set(d_lost) for w in fgold),
      "gold_definition":"exact lemma+HiTS POS+morphology",
      "cipher_seed":CIPHER_SEED,"cipher_chars":len(chars)}
    return corp,fc_lost,full_known,fgold,set(d_lost),reverse,list(cmap.values()),sorted(set("".join(full_known))),audit

CORP,FULL_L,FULL_K,FULL_G,DISC_L_SET,REV,CIPHER_CHARS,KNOWN_CHARS,AUDIT=build()
print("REF15_CAL_AUDIT="+json.dumps(AUDIT,ensure_ascii=False),flush=True)

# Monkeypatch only corpus/alphabet hooks consumed by the frozen runner.
nd.load_corpus=lambda url=None: CORP
nd.UGA_CHARS=CIPHER_CHARS
nd.HEB_CHARS=KNOWN_CHARS

class GermanRunner(nd.Runner):
    def __init__(self,args):
        super().__init__(args)
        # Override misleading Ugaritic audit with explicit German audit.
        print("REF15_RUN="+json.dumps({"seed":args.seed,"device":str(self.dev),
          "algorithm_commit":"e36031a9a6b8b67fcebb4d6f4af1c3753fad4287",
          "discovery_cognates":DISC_COG}),flush=True)

    def full_eval(self):
        lost=sorted([w for w in FULL_L if w in FULL_G],
                    key=lambda w:len(self.lcs.ids(w)),reverse=True)
        known=list(FULL_K)
        kid,klen=nd.pad_words(known,self.kcs,self.dev)
        n,k=len(lost),len(known)
        costs=np.empty((n,k),dtype=np.float32)
        mle=np.empty(n,dtype=np.int64)
        bs=self.a.full_eval_batch
        self.model.eval()
        for st in range(0,n,bs):
            en=min(st+bs,n)
            lid,llen=nd.pad_words(lost[st:en],self.lcs,self.dev)
            with torch.no_grad():
                lp,sc,_=self.model(lid,llen,kid,klen)
            mle[st:en]=sc.argmax(1).cpu().numpy()
            costs[st:en]=nd.expected_edits(lp,sc,known,self.kcs,True)
            if en%256==0 or en==n:
                print("REF15_FULL_PROGRESS="+json.dumps({"done":en,"total":n}),flush=True)
        def ev(pred,subset=None):
            ids=range(n) if subset is None else subset
            hit=tot=0
            for i in ids:
                tot+=1
                if known[int(pred[i])] in FULL_G[lost[i]]:hit+=1
            return hit/max(1,tot),hit,tot
        ma,mh,mn=ev(mle)
        demand=len(lost)
        flow,cost=nd.mincost(costs,demand,5,3)
        fp=flow.argmax(1)
        fa,fh,fn=ev(fp)
        novel=[i for i,w in enumerate(lost) if w not in DISC_L_SET]
        nma,nmh,nmn=ev(mle,novel)
        nfa,nfh,nfn=ev(fp,novel)
        out={"gold_lost":n,"known_vocab":k,"mcf_demand":demand,
             "all":{"mle":ma,"mle_hits":mh,"flow_edit":fa,"flow_hits":fh,"n":fn},
             "novel_lost_only":{"mle":nma,"mle_hits":nmh,"flow_edit":nfa,"flow_hits":nfh,"n":nfn},
             "mcf_cost":cost}
        print("REF15_FULL_EVAL="+json.dumps(out,separators=(",",":")),flush=True)
        return out

    def run(self):
        if self.a.self_test:return super().run()
        final=None;best_small=-1.;best_meta=None;best_state=None
        for rnd in range(1,self.a.rounds+1):
            if rnd==1:
                self.flow.fill(DISC_COG/self.flow.size)
                print("REF15_ESTEP="+json.dumps({"round":1,"warmup_uniform":True,
                    "flow_total":float(self.flow.sum())}),flush=True)
            else:
                demand=min((rnd-1)*50,DISC_COG)
                self.e_step(demand,edit=(rnd>self.a.warm_up_steps));self.reset_model()
            for ep in range(1,self.a.epochs+1):
                loss,nll,reg=self.train_epoch();ge=(rnd-1)*self.a.epochs+ep
                if ep%self.a.log_every==0:
                    print("REF15_TRAIN="+json.dumps({"round":rnd,"epoch":ep,"global_epoch":ge,
                          "loss":loss,"nll":nll,"reg":reg}),flush=True)
                if ep%self.a.eval_every==0:
                    final=self.eval(min(rnd*50,DISC_COG),with_edit=True)
                    score=float(final["flow_edit"]) if final["flow_edit"] is not None else -1.
                    if score>best_small:
                        best_small=score
                        best_meta={"round":rnd,"epoch":ep,"global_epoch":ge,
                                   "discovery_flow_edit":score,"discovery_mle":float(final["mle"])}
                        best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                        print("REF15_BEST_CHECKPOINT="+json.dumps(best_meta),flush=True)
        self.model.load_state_dict(best_state);self.model.to(self.dev)
        print("REF15_SELECTED_CHECKPOINT="+json.dumps(best_meta),flush=True)
        full=self.full_eval()
        out={"phase":"REF15_NEUROCIPHER_CAL1","status":"complete","seed":self.a.seed,
             "selected_checkpoint":best_meta,"full":full,"audit":AUDIT,
             "algorithm":"ACL2019 NeuroCipher refactor frozen at e36031a9; corpus adapter only",
             "decision_metrics":{"primary":"novel_lost_only.flow_edit",
               "secondary":"all.flow_edit","discovery":"selected checkpoint only"}}
        print("REF15_CAL_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--warm-up-steps",type=int,default=5)
    p.add_argument("--reg-hyper",type=float,default=.5)
    p.add_argument("--full-eval-batch",type=int,default=32)
    p.add_argument("--full-eval",action=argparse.BooleanOptionalAction,default=True)
    GermanRunner(p.parse_args()).run()
if __name__=="__main__":main()
