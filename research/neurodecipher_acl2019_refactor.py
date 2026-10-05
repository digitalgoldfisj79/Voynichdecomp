#!/usr/bin/env python3
"""
NeuroDecipher Ugaritic benchmark — modern self-contained refactor.

Algorithm preserved from:
Jiaming Luo, Yuan Cao, Regina Barzilay (ACL 2019),
"Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B"
Official implementation: https://github.com/j-luo93/NeuroDecipher
Pinned upstream commit/data: 480bad2487820e3737fecfdd108214efa769e34b

This file removes the obsolete arglib/dev_misc/MagicTensor/TensorBoard/TensorFlow
plumbing and ports OR-Tools calls to the current API.  It preserves the substantive
algorithm:
  * character-level attentional seq2seq model
  * universal character embeddings
  * monotonic alignment regularizer
  * norm-controlled residual connection
  * EM-style alternation between neural M-steps and min-cost-flow E-steps
  * flow momentum/decay
  * periodic neural reset
  * expected-edit-distance flow costs after warm-up
  * Ugaritic noisy benchmark settings from the paper/repository.

The paper reports 65.9% cognate identification accuracy in the noisy Ugaritic
setting.  This script must reproduce that benchmark approximately before it is
adapted to any Voynich or historical-German task.
"""
from __future__ import annotations
import argparse, json, math, random, urllib.request
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ortools.graph.python import min_cost_flow
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

UPSTREAM = "480bad2487820e3737fecfdd108214efa769e34b"
DATA_URL = f"https://raw.githubusercontent.com/j-luo93/NeuroDecipher/{UPSTREAM}/data/uga-heb.small.no_spe.cog"
FULL_DATA_URL = f"https://raw.githubusercontent.com/j-luo93/NeuroDecipher/{UPSTREAM}/data/uga-heb.no_spe.cog"

PAD_ID, SOW_ID, EOW_ID, UNK_ID = 0, 1, 2, 3
START = ["<PAD>", "<SOW>", "<EOW>", "<UNK>"]
UGA_CHARS = list("$&*<@HSTZabdghiklmnpqrstuvwxyz")
HEB_CHARS = list("$&<HSTabdghklmnpqrstwyz")

@dataclass
class Corpus:
    lost: List[str]
    known: List[str]
    cognates: Dict[str, Set[str]]

class Charset:
    def __init__(self, chars):
        self.id2char = START + chars
        self.char2id = {c:i for i,c in enumerate(self.id2char)}
    def __len__(self): return len(self.id2char)
    def ids(self, word:str):
        return [self.char2id.get(c, UNK_ID) for c in word] + [EOW_ID]
    def decode_ids(self, ids):
        out=[]
        for i in ids:
            c=self.id2char[int(i)]
            if int(i)==EOW_ID: break
            if int(i) < 4: c="|"
            out.append(c)
        return "".join(out)

def load_corpus(url=DATA_URL) -> Corpus:
    raw=urllib.request.urlopen(url,timeout=120).read().decode("utf-8")
    lines=[x for x in raw.splitlines() if x.strip()]
    assert lines[0].split("\t")==["uga-no_spe","heb-no_spe"]
    lv,kv=set(),set()
    cog:Dict[str,Set[str]]={}
    for line in lines[1:]:
        a,b=(line.split("\t")+["_","_"])[:2]
        aa=[x for x in a.split("|") if x!="_"]
        bb=[x for x in b.split("|") if x!="_"]
        lv.update(aa); kv.update(bb)
        if aa and bb:
            for x in aa:
                cog.setdefault(x,set()).update(bb)
    return Corpus(sorted(lv),sorted(kv),cog)

def pad_words(words, cs:Charset, device):
    seq=[cs.ids(w) for w in words]
    lens=torch.tensor([len(x) for x in seq],dtype=torch.long,device="cpu")
    mx=int(lens.max())
    arr=torch.full((len(seq),mx),PAD_ID,dtype=torch.long)
    for i,x in enumerate(seq): arr[i,:len(x)]=torch.tensor(x,dtype=torch.long)
    return arr.to(device), lens

class UniversalChars(nn.Module):
    def __init__(self,n_lost,n_known,d=250,u=50):
        super().__init__()
        self.universal=nn.Parameter(torch.empty(u,d))
        self.lost_map=nn.Parameter(torch.empty(n_lost,u))
        self.known_map=nn.Parameter(torch.empty(n_known,u))
    def weights(self,side):
        m=self.lost_map if side=="lost" else self.known_map
        return m @ self.universal
    def embed(self,ids,side): return self.weights(side)[ids]
    def project(self,x,side): return x @ self.weights(side).T
    def start(self,side): return self.weights(side)[SOW_ID]
    def soft(self,p,side): return p @ self.weights(side)

class NeuroCipher(nn.Module):
    def __init__(self,n_lost_chars,n_known_chars,dropout=.3,d=250,h=250,u=50):
        super().__init__()
        self.d=d; self.h=h
        self.uc=UniversalChars(n_lost_chars,n_known_chars,d,u)
        self.encoder=nn.LSTM(d,h,num_layers=1,bidirectional=True,batch_first=True)
        self.decoder=nn.LSTMCell(d+h,h)
        self.Wa=nn.Parameter(torch.empty(2*h,h))
        self.hidden=nn.Linear(3*h,d)
        self.drop=nn.Dropout(dropout)
        self.reinit_like_upstream()

    def reinit_like_upstream(self):
        # Mirrors Trainer._init_params in the published repository.
        for name,p in self.named_parameters():
            if p.ndim==2:
                nn.init.xavier_uniform_(p)
            elif "bias_ih" in name or "bias_hh" in name:
                n=p.numel()
                with torch.no_grad(): p[n//4:n//2]=1.0

    def forward(self,lost_ids,lost_lens,target_ids,target_lens):
        # lost batch is always the full selected lost vocabulary, sorted by length.
        emb=self.uc.embed(lost_ids,"lost")
        packed=nn.utils.rnn.pack_padded_sequence(self.drop(emb),lost_lens,batch_first=True,enforce_sorted=True)
        B=lost_ids.size(0)
        h0=torch.zeros(2,B,self.h,device=lost_ids.device)
        c0=torch.zeros(2,B,self.h,device=lost_ids.device)
        hp,(hn,cn)=self.encoder(packed,(h0,c0))
        hs,_=nn.utils.rnn.pad_packed_sequence(hp,batch_first=True)
        # Published LSTMState.from_pytorch sums the two directions.
        h=hn.view(1,2,B,self.h).sum(1)[0]
        c=cn.view(1,2,B,self.h).sum(1)[0]
        mask=(lost_ids!=PAD_ID).float()
        T=int(target_lens.max())
        input_emb=self.uc.start("known").expand(B,-1)
        htilde=torch.zeros(B,self.h,device=lost_ids.device)
        Whs=self.drop(hs).reshape(-1,2*self.h).mm(self.Wa).view(B,hs.size(1),self.h)
        means=[]; lps=[]
        for _ in range(T):
            inp=self.drop(torch.cat([htilde,input_emb],dim=-1))
            h,c=self.decoder(inp,(h,c))
            scores=(Whs * self.drop(h).unsqueeze(1)).sum(-1)
            scores=scores*mask + (-9999.0)*(1.0-mask)
            att=torch.softmax(scores,dim=-1)
            ctxs=(att.unsqueeze(-1)*hs).sum(1)
            raw=F.leaky_relu(self.hidden(self.drop(torch.cat([ctxs,h],dim=-1))))
            # relative norm-controlled residual, r=0.2
            ctxemb=(att.unsqueeze(-1)*emb).sum(1)
            base_norm=ctxemb.norm(dim=-1,keepdim=True)
            raw_norm=raw.norm(dim=-1,keepdim=True)
            adjusted=torch.minimum(raw_norm,base_norm*0.2)
            htilde=ctxemb + F.normalize(raw,dim=-1)*adjusted
            logits=self.uc.project(self.drop(htilde),"known")
            lp=torch.log_softmax(logits,dim=-1)
            input_emb=self.uc.soft(lp.exp(),"known")
            lps.append(lp)
            pos=torch.arange(hs.size(1),device=hs.device,dtype=torch.float32)
            means.append((att*pos).sum(-1))
        log_probs=torch.stack(lps,dim=1) # B x T x C
        mean_pos=torch.stack(means,dim=1)
        prev=torch.cat([torch.full((B,1),-1.0,device=mean_pos.device),mean_pos[:,:-1]],dim=1)
        reg_weight=(lost_lens.to(mean_pos.device).float().view(-1,1)-1.0-prev).clamp(0.0,1.0)
        rel=mean_pos-prev
        reg=(((rel-1.0)**2)*(rel!=1.0).float()*reg_weight).sum()
        scores=score_words(log_probs,target_ids,target_lens)
        return log_probs,scores,reg

def score_words(log_probs,target_ids,target_lens):
    B,T,C=log_probs.shape
    K=target_ids.size(0)
    score=torch.zeros(B,K,device=log_probs.device)
    for p in range(T):
        ids=target_ids[:,p]
        v=log_probs[:,p,:][:,ids]
        score += v * (target_lens.to(log_probs.device)>p).float().view(1,-1)
    return score

def batches(n,batch,device):
    order=torch.randperm(n,device="cpu").tolist()
    for i in range(0,n,batch): yield order[i:i+batch]

def mincost(dists:np.ndarray,demand:int,n_similar=5,capacity=3):
    costs=(dists*100.0).astype(np.int64)
    nt,ns=costs.shape
    demand=min(int(demand),nt,ns)
    if demand<=0: return np.zeros((nt,ns),np.float32),0
    idx=np.argpartition(costs,n_similar-1,axis=1)[:,:n_similar]
    allowed=set(map(int,idx.ravel()))
    if len(allowed)<demand:
        rem=list(set(range(ns))-allowed)
        allowed.update(random.sample(rem,demand-len(allowed)))
    allowed=sorted(allowed)
    m=min_cost_flow.SimpleMinCostFlow()
    src,sink=0,1
    for t in range(nt):
        m.add_arc_with_capacity_and_unit_cost(src,t+2,1,0)
    for s in range(ns):
        m.add_arc_with_capacity_and_unit_cost(s+2+nt,sink,int(capacity),0)
    for t in range(nt):
        for s in allowed:
            m.add_arc_with_capacity_and_unit_cost(t+2,s+2+nt,1,int(costs[t,s]))
    m.set_node_supply(src,demand);m.set_node_supply(sink,-demand)
    status=m.solve()
    if status!=m.OPTIMAL: raise RuntimeError(f"min-cost-flow status {status}")
    flow=np.zeros((nt,ns),np.float32)
    for i in range(m.num_arcs()):
        a,b=m.tail(i),m.head(i)
        if a>1 and b>1+nt and m.flow(i):
            flow[a-2,b-2-nt]=m.flow(i)
    return flow,int(m.optimal_cost())

def expected_edits(log_probs, exact_scores, known_forms, kcs:Charset, edit:bool, nsamp=10, alpha=10.0):
    if not edit:
        return (-exact_scores).detach().cpu().numpy()
    with torch.no_grad():
        sharp=torch.log_softmax(log_probs*alpha,dim=-1)
        B,T,C=sharp.shape
        samp=torch.multinomial(sharp.exp().reshape(B*T,C),nsamp,replacement=True).view(B,T,nsamp)
        sg=torch.gather(sharp,2,samp)
        samp_np=samp.permute(0,2,1).cpu().numpy()
        tokens=np.empty((B,nsamp),dtype=object)
        slen=np.zeros((B,nsamp),dtype=np.int64)
        for b in range(B):
            for q in range(nsamp):
                tok=kcs.decode_ids(samp_np[b,q])
                tokens[b,q]=tok;slen[b,q]=len(tok)+1
        mask=(torch.arange(T,device=sharp.device).view(1,T,1) <
              torch.tensor(slen,device=sharp.device).view(B,1,nsamp))
        slog=(sg*mask).sum(1).cpu().numpy() # B x nsamp
        exact=exact_scores.detach().cpu().numpy()
    # duplicate samples: keep first occurrence, matching upstream.
    base_valid=np.ones((B,nsamp),bool)
    for b in range(B):
        seen=set()
        for q in range(nsamp):
            tok=tokens[b,q]
            if tok in seen: base_valid[b,q]=False
            else: seen.add(tok)
    out=[]
    flat=[str(x) for x in tokens.reshape(-1)]
    for st in range(0,len(known_forms),1000):
        forms=known_forms[st:st+1000];K=len(forms)
        # rapidfuzz gives identical Levenshtein insertion/deletion/substitution distance.
        raw=process.cdist(flat,forms,scorer=Levenshtein.distance,dtype=np.int16,workers=-1)
        raw=raw.reshape(B,nsamp,K).transpose(0,2,1).astype(np.float32)
        flen=np.asarray([len(x) for x in forms],dtype=np.int64)
        denom=np.minimum(flen.reshape(1,K,1),slen.reshape(B,1,nsamp))+1
        d=raw/denom
        d=np.concatenate([np.zeros((B,K,1),np.float32),d],axis=2)
        valid=np.broadcast_to(base_valid[:,None,:],(B,K,nsamp)).copy()
        for q in range(nsamp):
            eq=np.asarray(forms,dtype=object).reshape(1,K)==tokens[:,q].reshape(B,1)
            valid[:,:,q] &= ~eq
        dup=np.concatenate([np.ones((B,K,1),bool),valid],axis=2)
        logits=np.concatenate([exact[:,st:st+K,None],np.broadcast_to(slog[:,None,:],(B,K,nsamp))],axis=2)
        logits=np.where(dup,logits,logits-999.0)
        mx=logits.max(2,keepdims=True)
        p=np.exp(logits-mx);p/=p.sum(2,keepdims=True)
        out.append((d*p).sum(2))
    return np.concatenate(out,axis=1)

def evaluate_preds(pred_idx,lost_words,known_words,cog):
    hit=0
    for i,j in enumerate(pred_idx):
        if known_words[int(j)] in cog.get(lost_words[i],set()): hit+=1
    return hit/len(lost_words),hit

class Runner:
    def __init__(self,args):
        self.a=args
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        self.dev=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.c=load_corpus();self.lcs=Charset(UGA_CHARS);self.kcs=Charset(HEB_CHARS)
        # Match collate_fn behaviour: lost batches are sorted descending by sequence length.
        self.lost=sorted(self.c.lost,key=lambda w:len(self.lcs.ids(w)),reverse=True)
        self.known=list(self.c.known)
        self.lost_ids,self.lost_len=pad_words(self.lost,self.lcs,self.dev)
        self.known_ids,self.known_len=pad_words(self.known,self.kcs,self.dev)
        self.eval_lost=sorted([w for w in self.lost if self.c.cognates.get(w)],
                              key=lambda w:len(self.lcs.ids(w)),reverse=True)
        self.eval_ids,self.eval_len=pad_words(self.eval_lost,self.lcs,self.dev)
        self.flow=np.zeros((len(self.lost),len(self.known)),np.float32)
        self.model=NeuroCipher(len(self.lcs),len(self.kcs),dropout=.3).to(self.dev)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)
        print("REFAC_AUDIT="+json.dumps({"upstream":UPSTREAM,"lost_vocab":len(self.lost),
             "known_vocab":len(self.known),"eval_cognate_lost":len(self.eval_lost),
             "device":str(self.dev),"paper_target_noisy":0.659,
             "reg_hyper":args.reg_hyper}),flush=True)

    def reset_model(self):
        self.model.reinit_like_upstream()
        self.opt=torch.optim.Adam(self.model.parameters(),lr=.005)

    def model_all(self,lost_ids=None,lost_len=None):
        if lost_ids is None: lost_ids,lost_len=self.lost_ids,self.lost_len
        self.model.eval()
        with torch.no_grad():
            return self.model(lost_ids,lost_len,self.known_ids,self.known_len)

    def e_step(self,demand,edit):
        lp,sc,_=self.model_all()
        costs=expected_edits(lp,sc,self.known,self.kcs,edit)
        nf,cost=mincost(costs,demand,5,3)
        self.flow=.25*self.flow+.75*nf
        # Diagnostic true-pair accuracy among nonzero best rows.
        best=self.flow.argmax(1);nz=self.flow.max(1)>0
        n=int(nz.sum());hit=0
        for i in np.where(nz)[0]:
            hit += self.known[int(best[i])] in self.c.cognates.get(self.lost[i],set())
        print("REFAC_ESTEP="+json.dumps({"demand":demand,"edit":edit,"mcf_cost":cost,
              "nonzero_rows":n,"true_best":hit,"true_best_rate":hit/max(1,n)}),flush=True)

    def train_epoch(self):
        self.model.train()
        for ids in batches(len(self.known),500,self.dev):
            # Collate sorts known batch by length descending.
            ids=sorted(ids,key=lambda j:int(self.known_len[j]),reverse=True)
            tid=self.known_ids[ids]
            tl=self.known_len[ids]
            lp,sc,reg=self.model(self.lost_ids,self.lost_len,tid,tl)
            fs=torch.tensor(self.flow[:,ids],dtype=torch.float32,device=self.dev)
            fk=fs.sum(0);total=fk.sum()
            if float(total)<=0: continue
            nll=torch.logsumexp(sc+torch.log(fs+1e-8),dim=0)
            nll=-(nll*fk).sum()/total
            rloss=reg/total
            loss=nll+self.a.reg_hyper*rloss
            self.opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
            self.opt.step()
        return float(loss.detach()),float(nll.detach()),float(rloss.detach())

    def eval(self,demand,with_edit=True):
        self.model.eval()
        with torch.no_grad():
            lp,sc,_=self.model(self.eval_ids,self.eval_len,self.known_ids,self.known_len)
        mle=sc.argmax(1).cpu().numpy()
        ma,mh=evaluate_preds(mle,self.eval_lost,self.known,self.c.cognates)
        c0=expected_edits(lp,sc,self.known,self.kcs,False)
        f0,_=mincost(c0,demand,5,3);p0=f0.argmax(1)
        a0,h0=evaluate_preds(p0,self.eval_lost,self.known,self.c.cognates)
        ae=he=None
        if with_edit:
            ce=expected_edits(lp,sc,self.known,self.kcs,True)
            fe,_=mincost(ce,demand,5,3);pe=fe.argmax(1)
            ae,he=evaluate_preds(pe,self.eval_lost,self.known,self.c.cognates)
        out={"demand":demand,"mle":ma,"flow_noedit":a0,"flow_edit":ae,
             "hits":{"mle":mh,"flow_noedit":h0,"flow_edit":he}}
        print("REFAC_EVAL="+json.dumps(out),flush=True)
        return out

    def full_eval(self):
        """Paper-faithful test: train on the 10% subset, test on the full Ugaritic corpus."""
        fc=load_corpus(FULL_DATA_URL)
        lost=sorted([w for w in fc.lost if fc.cognates.get(w)],
                    key=lambda w:len(self.lcs.ids(w)),reverse=True)
        known=list(fc.known)
        kid,klen=pad_words(known,self.kcs,self.dev)
        n,k=len(lost),len(known)
        costs=np.empty((n,k),dtype=np.float32)
        mle=np.empty(n,dtype=np.int64)
        self.model.eval()
        bs=self.a.full_eval_batch
        for st in range(0,n,bs):
            en=min(st+bs,n)
            lid,llen=pad_words(lost[st:en],self.lcs,self.dev)
            with torch.no_grad():
                lp,sc,_=self.model(lid,llen,kid,klen)
            mle[st:en]=sc.argmax(1).cpu().numpy()
            costs[st:en]=expected_edits(lp,sc,known,self.kcs,True)
            print("REFAC_FULL_PROGRESS="+json.dumps({"done":en,"total":n}),flush=True)
        ma,mh=evaluate_preds(mle,lost,known,fc.cognates)
        print("REFAC_FULL_MLE="+json.dumps({"accuracy":ma,"hits":mh,"n":n}),flush=True)
        flow,cost=mincost(costs,2214,5,3)
        pred=flow.argmax(1)
        fa,fh=evaluate_preds(pred,lost,known,fc.cognates)
        out={"lost_with_cognate":n,"known_vocab":k,"paper_cognate_rows":2214,
             "mle":ma,"mle_hits":mh,"flow_edit":fa,"flow_edit_hits":fh,
             "mcf_cost":cost,"paper_target":0.659}
        print("REFAC_FULL_EVAL="+json.dumps(out),flush=True)
        return out

    def self_test(self):
        # Exercises neural forward/backward, expected edits and current OR-Tools flow.
        li=self.lost_ids[:20];ll=self.lost_len[:20]
        ki=self.known_ids[:40];kl=self.known_len[:40]
        self.model.train();lp,sc,reg=self.model(li,ll,ki,kl)
        loss=-sc.mean()+1e-5*reg
        self.opt.zero_grad();loss.backward();self.opt.step()
        self.model.eval()
        with torch.no_grad():lp,sc,_=self.model(li,ll,ki,kl)
        c=expected_edits(lp,sc,self.known[:40],self.kcs,True,nsamp=3)
        f,cost=mincost(c,10,5,3)
        ok=bool(f.sum()==10 and np.isfinite(c).all())
        print("REFAC_SELFTEST="+json.dumps({"pass":ok,"flow":float(f.sum()),"cost":cost,
              "shape":list(c.shape)}),flush=True)
        return ok

    def run(self):
        if self.a.self_test:
            if not self.self_test(): raise SystemExit(2)
            return
        final=None
        for rnd in range(1,self.a.rounds+1):
            # Published E step schedule.
            if rnd==1:
                self.flow.fill(221/self.flow.size)
                print("REFAC_ESTEP="+json.dumps({"round":1,"warmup_uniform":True,
                      "flow_total":float(self.flow.sum())}),flush=True)
            else:
                demand=min((rnd-1)*50,221)
                self.e_step(demand,edit=(rnd>self.a.warm_up_steps))
                self.reset_model()
            for ep in range(1,self.a.epochs+1):
                loss,nll,reg=self.train_epoch()
                global_ep=(rnd-1)*self.a.epochs+ep
                if ep%self.a.log_every==0:
                    print("REFAC_TRAIN="+json.dumps({"round":rnd,"epoch":ep,"global_epoch":global_ep,
                          "loss":loss,"nll":nll,"reg":reg}),flush=True)
                if ep%self.a.eval_every==0:
                    final=self.eval(min(rnd*50,221),with_edit=True)
        if final is None: final=self.eval(221,with_edit=True)
        full=self.full_eval() if self.a.full_eval else None
        out={"status":"complete","algorithm":"NeuroCipher ACL2019 refactor",
             "upstream_commit":UPSTREAM,"paper_noisy_target":0.659,
             "small_subset_diagnostic":final,"full_paper_eval":full,"settings":{"rounds":self.a.rounds,"epochs_per_round":self.a.epochs,
             "batch_size":500,"capacity":3,"n_similar":5,"momentum":0.25,
             "warm_up_steps":self.a.warm_up_steps,"reg_hyper":self.a.reg_hyper,
             "seed":self.a.seed}}
        print("REFAC_RESULT_JSON="+json.dumps(out),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--epochs",type=int,default=150)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--warm-up-steps",type=int,default=5)
    p.add_argument("--reg-hyper",type=float,default=0.5,
                   help="paper §5: alignment regularization hyperparameter 0.5")
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--full-eval",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--full-eval-batch",type=int,default=32)
    Runner(p.parse_args()).run()
if __name__=="__main__": main()
