# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import babuini_sta_v1 as b

# Release safety patch frozen before formal qualification: BAB_C1 identities may be
# listed in the manifest, but its STA/aaa contents are not projected, counted or scored.
def sealed_target_rep(lm,pages,rep,fit,h1,c1):
    v=b.select_vocab(pages,fit,89)
    fw,fcov=b.project(pages,fit,v)
    hw,hcov=b.project(pages,h1,v)
    if fcov['whole_word_char_coverage']<.995 or hcov['whole_word_char_coverage']<.995:
        return {'rep':rep,'status':'SURFACE_INCOMPATIBLE','vocab':v,'fit_coverage':fcov,'H1_coverage':hcov}
    sol=b.c.solve(b.c.stats(fw),lm,f'TARGET:{rep}',False)
    hs=b.c.fixed(b.c.stats(hw),lm,sol['dec'])
    return {'rep':rep,'vocab':v,'fit_score':sol['fit_score'],'H1_score':hs,'agreement':sol['agreement'],'converged':sol['converged'],'fit_coverage':fcov,'H1_coverage':hcov}

b.target_rep=sealed_target_rep
if __name__=='__main__':b.main()
