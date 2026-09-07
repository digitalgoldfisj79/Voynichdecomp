#!/usr/bin/env python3
"""Implementation-only fix for v0.2 Stage M. No assay changes."""
import numpy as np
import vms_textual_connections_v02_m as base


def dataset_fixed(body):
    pars=base.parse_paragraphs(body); blocks=base.blockify(pars)
    alltok=[t for _,ls in pars for line in ls for t in line]
    from collections import Counter
    import math
    freq=Counter(alltok);rare={t:1/math.log2(2+freq[t]) for t in freq if 2<=freq[t]<=20}
    X=[];y=[];groups=[];para=[]
    for pi,page,bs,bls in blocks:
        for i in range(len(bs)):
            for j in range(i+1,len(bs)):
                X.append(base.feature(bs[i],bs[j],bls[i],bls[j],rare));y.append(1 if j==i+1 else 0);groups.append(page);para.append(pi)
    ya=np.asarray(y,int)
    return np.asarray(X,float),ya,np.asarray(groups),np.asarray(para),{'paragraphs_raw':len(pars),'paragraphs_admitted':len(blocks),'pages':len(set(groups)),'pairs':len(y),'positives':int(ya.sum()),'base_rate':float(ya.mean()) if len(ya) else 0.0}

base.dataset=dataset_fixed
if __name__=='__main__': base.main()
