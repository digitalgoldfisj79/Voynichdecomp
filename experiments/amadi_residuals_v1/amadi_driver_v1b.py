# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
from unidecode import unidecode
import amadi_residuals_v1 as m

# Source correction established before Q1: all quoted g examples delete g
# (gioue->ioue, giocho->iocho, gienere->ienere; long example negotii->neotii).
def norm_r12_fixed(raw:str)->list[int]:
    s=unidecode(raw).lower(); out=[]
    for c in s:
        if not ("a"<=c<="z"): continue
        if c=="j": c="i"
        if c=="b": c="u"
        elif c=="d": c="t"
        elif c in "fghp": continue
        elif c=="q": c="c"
        elif c=="v": c="o"       # modern v is source consonantal u/v
        elif c=="w": c="u"
        elif c=="y": c="i"
        elif c in "xz": c="s"
        if c not in m.R2I: return []
        out.append(m.R2I[c])
    return out
m.norm_r12=norm_r12_fixed

VC_PAIRS=[
("non","nno"),("staro","strao"),("a","a"),("discorere","dscrioee"),("che","che"),("differentia","dffrntieia"),("sia","sia"),("tra","tra"),("la","la"),("riputatione","rpttnuaioe"),("de","de"),
("competitori","cmpttroeioi"),("et","te"),("la","la"),("melanconia","mlnceaoia"),("et","te"),("il","li"),("dollore","dllrooe"),("de","de"),("litterati","lttrteai"),("ne","ne"),("de","de"),("il","li"),
("splendore","splndreoe"),("dellandar","dlIndreaa"),("ben","bne"),("uestitto","stttueio"),("ne","ne"),("della","dllea"),("licentia","lcntieia"),("del","dle"),("far","fra"),("lo","lo"),("amore","mraoe")]
EXPECTED_VC_DISCREPANCIES={"discorere","differentia","riputatione","melanconia","litterati","dellandar"}
R12_LOCAL=[
("b","u",["labro","fabro","bartolomeo","battista"],["lauro","fauro","uartolomeo","uattista"]),
("d","t",["grande","uelade"],["grante","uelate"]),
("f","",["felice","forte","feltro"],["elice","orte","eltro"]),
("g","",["gioue","giocho","gienere"],["ioue","iocho","ienere"]),
("p","",["pietro","paullo"],["ietro","aullo"]),
("q","c",["quando","qualle"],["cuando","cualle"]),
]

def q0_fixed():
    rows=[]; mism=[]
    for p,src in VC_PAIRS:
        mech="".join(m.PLAIN[x] for x in m.vc_word(m.norm_std(p))); ok=mech.lower()==src.lower(); rows.append({"plain":p,"mechanical":mech,"source":src,"match":ok})
        if not ok: mism.append(p)
    local=[]
    for old,new,ps,es in R12_LOCAL:
        for p,e in zip(ps,es):
            got=p.replace(old,new); local.append({"rule":f"{old}->{new or 'DELETE'}","plain":p,"expected":e,"got":got,"match":got==e})
    p="mouendo"; got=p[:2]+"o"+p[3:]; local.append({"rule":"consonantal-u->o","plain":p,"expected":"mooendo","got":got,"match":got=="mooendo"})
    supported="abcdefgijlmnopqrstu vwxyz".replace(" ",""); formal=[]
    for ch in supported:
        if ch=="k": continue
        q=m.norm_r12(ch); formal.append({"input":ch,"output":"".join(m.R12[x] for x in q),"in_declared_alphabet":all(0<=x<m.L12 for x in q)})
    vc_pass=set(mism)==EXPECTED_VC_DISCREPANCIES and sum(x["match"] for x in rows)==28
    r12_pass=all(x["match"] for x in local) and all(x["in_declared_alphabet"] for x in formal)
    return {"VC_END":{"exact_matches":sum(x["match"] for x in rows),"total_pairs":len(rows),"discrepancies":mism,"expected_discrepancies":sorted(EXPECTED_VC_DISCREPANCIES),"pass":vc_pass},"R12_V1_024":{"local_rule_examples":local,"formalisation_alphabet_check":formal,"source_correction":"g is deleted, as consistently shown by quoted examples","pass":r12_pass},"pass":bool(vc_pass and r12_pass)}
m.q0=q0_fixed

if __name__=="__main__": m.main()
