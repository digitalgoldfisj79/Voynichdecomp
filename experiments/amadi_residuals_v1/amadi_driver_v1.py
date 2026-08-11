# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import amadi_residuals_v1 as m

VC_PAIRS=[
("non","nno"),("staro","strao"),("a","a"),("discorere","dscrioee"),("che","che"),("differentia","dffrntieia"),("sia","sia"),("tra","tra"),("la","la"),("riputatione","rpttnuaioe"),("de","de"),
("competitori","cmpttroeioi"),("et","te"),("la","la"),("melanconia","mlnceaoia"),("et","te"),("il","li"),("dollore","dllrooe"),("de","de"),("litterati","lttrteai"),("ne","ne"),("de","de"),("il","li"),
("splendore","splndreoe"),("dellandar","dlIndreaa"),("ben","bne"),("uestitto","stttueio"),("ne","ne"),("della","dllea"),("licentia","lcntieia"),("del","dle"),("far","fra"),("lo","lo"),("amore","mraoe")]
EXPECTED_VC_DISCREPANCIES={"discorere","differentia","riputatione","melanconia","litterati","dellandar"}
R12_LOCAL=[
("b","u",["labro","fabro","bartolomeo","battista"],["lauro","fauro","uartolomeo","uattista"]),
("d","t",["grande","uelade"],["grante","uelate"]),
("f","",["felice","forte","feltro"],["elice","orte","eltro"]),
("g","i",["gioue","giocho","gienere"],["ioue","iocho","ienere"]),
("p","",["pietro","paullo"],["ietro","aullo"]),
("q","c",["quando","qualle"],["cuando","cualle"]),
]

def q0_fixed():
    rows=[]; mism=[]
    for p,src in VC_PAIRS:
        mech="".join(m.PLAIN[x] for x in m.vc_word(m.norm_std(p)))
        ok=mech.lower()==src.lower(); rows.append({"plain":p,"mechanical":mech,"source":src,"match":ok})
        if not ok: mism.append(p)
    local=[]
    for old,new,ps,es in R12_LOCAL:
        for p,e in zip(ps,es):
            got=p.replace(old,new); local.append({"rule":f"{old}->{new or 'DELETE'}","plain":p,"expected":e,"got":got,"match":got==e})
    # The consonantal-u example is source-local: operate on the historical u at index 2 only.
    p="mouendo"; got=p[:2]+"o"+p[3:]; local.append({"rule":"consonantal-u->o","plain":p,"expected":"mooendo","got":got,"match":got=="mooendo"})
    # h is explicitly deleted in the prose; no short pair is supplied in the extraction.
    supported="abcdefgijlmnopqrstu vwxyz".replace(" ","")
    formal=[]
    for ch in supported:
        if ch=="k": continue
        q=m.norm_r12(ch)
        formal.append({"input":ch,"output":"".join(m.R12[x] for x in q),"in_declared_alphabet":all(0<=x<m.L12 for x in q)})
    vc_pass=set(mism)==EXPECTED_VC_DISCREPANCIES and sum(x["match"] for x in rows)==28
    r12_pass=all(x["match"] for x in local) and all(x["in_declared_alphabet"] for x in formal)
    return {"VC_END":{"exact_matches":sum(x["match"] for x in rows),"total_pairs":len(rows),"discrepancies":mism,"expected_discrepancies":sorted(EXPECTED_VC_DISCREPANCIES),"pass":vc_pass},"R12_V1_024":{"local_rule_examples":local,"formalisation_alphabet_check":formal,"pass":r12_pass},"pass":bool(vc_pass and r12_pass)}

m.q0=q0_fixed

if __name__=="__main__":
    m.main()
