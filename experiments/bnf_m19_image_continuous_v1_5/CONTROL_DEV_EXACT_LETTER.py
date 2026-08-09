#!/usr/bin/env python3
import urllib.request,json,math
import numpy as np
import torch
U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/90dd44f655844aa60bc5afbe50c10286faddba1f/experiments/bnf_m19_image_continuous_v1_5/run_v15.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
ns={'__name__':'v15exactdev'};exec(compile(src,'run_v15.py','exec'),ns)
b=ns['b'];SIGMA=0.01021827930671254;DEVICE=ns['DEVICE'];K=19;NL=len(b['ALPH'])
LOGM=torch.full((NL,K),-1e30,dtype=torch.float32,device=DEVICE)
M=torch.tensor(b['EMIT'],dtype=torch.float32,device=DEVICE);LOGM=torch.where(M>0,torch.log(torch.clamp(M,min=1e-30)),LOGM)

def exact_comp(lm):
    return tuple(torch.log(torch.clamp(torch.tensor(x,dtype=torch.float32,device=DEVICE),min=1e-30)) for x in [lm['T'],lm['st'],lm['en']])

def logp_values(X,mu,sigma):
    D=X.shape[-1];x2=(X*X).sum(-1,keepdim=True);m2=(mu*mu).sum(-1).view(*([1]*(X.ndim-1)),-1);cross=torch.matmul(X,mu.t());return -.5*D*math.log(2*math.pi*sigma*sigma)-torch.clamp(x2+m2-2*cross,min=0)/(2*sigma*sigma)

def fb_exact(X,mu,sigma,tc,need_resp=False):
    lt,ls,le=tc;lp=logp_values(X,mu,sigma); # B,L,V
    lemit=torch.logsumexp(lp[:,:,:,None] + LOGM.t()[None,None,:,:],dim=2) # B,L,letters; value dim summed
    B,L,_=lemit.shape;alph=[];a=ls[None,:]+lemit[:,0,:];alph.append(a)
    for t in range(1,L):a=lemit[:,t,:]+torch.logsumexp(a[:,:,None]+lt[None,:,:],dim=1);alph.append(a)
    ll=torch.logsumexp(a+le[None,:],dim=1)
    if not need_resp:return ll,None
    beta=le[None,:].expand(B,NL);gam=[None]*L;resp=[None]*L;gam[L-1]=torch.softmax(alph[L-1]+beta,dim=1)
    for t in range(L-2,-1,-1):
        beta=torch.logsumexp(lt[None,:,:]+lemit[:,t+1,:][:,None,:]+beta[:,None,:],dim=2);gam[t]=torch.softmax(alph[t]+beta,dim=1)
    # r_t(v)=sum_l gamma_t(l) p(v|l,x_t)
    for t in range(L):
        # B,letters,values
        lq=LOGM[None,:,:]+lp[:,t,None,:]-lemit[:,t,:,None];q=torch.exp(torch.clamp(lq,min=-80,max=20));resp[t]=(gam[t][:,:,None]*q).sum(1)
    return ll,resp

def length_ll(L,tc):
    lt,ls,le=tc;a=ls
    for _ in range(1,L):a=torch.logsumexp(a[:,None]+lt,dim=0)
    return torch.logsumexp(a+le,dim=0)

def exact_em(groups,mu0,sigma,lm,iters=10):
    mu=torch.tensor(mu0,dtype=torch.float32,device=DEVICE);tc=exact_comp(lm);cnt=None;trll=None
    for it in range(iters):
        sums=torch.zeros_like(mu);cnt=torch.zeros(K,device=DEVICE);tot=0.;n=0
        with torch.no_grad():
            for L,X in groups.items():
                ll,r=fb_exact(X,mu,sigma,tc,True);tot+=float(ll.sum().cpu());n+=X.shape[0]*L
                for t,rv in enumerate(r):sums+=rv.t()@X[:,t,:];cnt+=rv.sum(0)
            mu=sums/torch.clamp(cnt[:,None],min=1e-6);trll=tot/max(1,n)
        print('EXACT_EM',it+1,round(trll,6),flush=True)
    return mu,trll,cnt

def exact_score(groups,mu,sigma,lm):
    tc=exact_comp(lm);tot=llen=0.;n=0
    with torch.no_grad():
        for L,X in groups.items():
            ll,_=fb_exact(X,mu,sigma,tc,False);tot+=float(ll.sum().cpu());llen+=float(length_ll(L,tc).cpu())*X.shape[0];n+=X.shape[0]*L
    const=-.5*next(iter(groups.values())).shape[-1]*math.log(2*math.pi*sigma*sigma);joint=tot/n;length=llen/n;return {'joint':joint,'visual_gain':joint-length-const,'length':length}

def fit_exact(words,groups,sigma,lm,comp,tag):
    fits=[]
    for rs in [408,409]:
        mu0=ns['init_means'](words,comp,(tag,rs),rs);mu,ll,cnt=exact_em(groups,mu0,sigma,lm,10);fits.append((mu,ll,cnt))
    best=0 if fits[0][1]>=fits[1][1] else 1;m1,m2=fits[0][0],fits[1][0];cos=torch.sum(m1*m2,1)/(torch.linalg.norm(m1,dim=1)*torch.linalg.norm(m2,dim=1)+1e-12);w=(fits[0][2]+fits[1][2])/2;agr=float((cos*w).sum().cpu()/w.sum().cpu());return fits[best][0],fits[best][1],fits[best][2],agr

def one(target,lms,pools,comps):
    trtxt,hotxt=ns['split_plain'](pools[target],('v15qual',target));tw,P,sig=ns['synth_words'](trtxt,target,64,SIGMA,'tr');hw,P2,_=ns['synth_words'](hotxt,target,64,SIGMA,'ho');tg=ns['group_tensors'](tw);hg=ns['group_tensors'](hw);rows=[];correct=None
    for cand in b['LANGS']:
        mu,ll,cnt,agr=fit_exact(tw,tg,sig,lms[cand],comps[cand],('exactdev',target,cand));sc=exact_score(hg,mu,sig,lms[cand]);rows.append((cand,sc,agr));
        if cand==target:correct=(mu,agr)
    rows.sort(key=lambda x:x[1]['visual_gain'],reverse=True);rec=ns['mean_recovery'](correct[0],P);out={'target':target,'top':rows[0][0],'margin':rows[0][1]['visual_gain']-rows[1][1]['visual_gain'],'rank':1+next(i for i,x in enumerate(rows) if x[0]==target),'recovery':rec,'agreement':correct[1],'ranking':[(x[0],x[1]['visual_gain'],x[1]['joint']) for x in rows]};print('EXACT_DEV',json.dumps(out,separators=(',',':')),flush=True);return out

def main():
    lms,pools,_=b['load_lms']();comps={la:b['induced'](lms[la]) for la in b['LANGS']};out=[one('latin',lms,pools,comps)];print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
