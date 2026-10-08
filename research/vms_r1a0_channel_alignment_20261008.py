#!/usr/bin/env python3
# VMS-R1A0 — structural alignment checkpoint for observable-channel ladder.
import collections,hashlib,json,urllib.request

R4_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d1daa5877fc7d8ba2be9bfa98ec69150d95af8ee/research/vms_residrec4b_recipes_qualification_20261008.py"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"

# import R4 definitions only, before CLI execution
src=urllib.request.urlopen(R4_URL,timeout=120).read().decode()
r4={"__name__":"r4"}
exec(compile(src.split("\nap=argparse.ArgumentParser()")[0],R4_URL,"exec"),r4)

lsrc=urllib.request.urlopen(LAT_URL,timeout=120).read().decode()
lat={"__name__":"lat"}
exec(compile(lsrc.split("\n# annotate chunks")[0],LAT_URL,"exec"),lat)

# We need canonical rows with pieces annotations; reproduce exact deterministic annotation.
occ_url=lat["BASE"]
ons={"__name__":"occ"}
exec(compile(urllib.request.urlopen(occ_url,timeout=120).read().decode(),occ_url,"exec"),ons)
rows,folds=ons["load_rows"]()
segment=lat["segment"]; ST=lat["ST"]

for rr in rows:
    z=segment(rr["token"])
    rr["pieces"]=z
    rr["start"]=ST[z[0]]
    rr["final_piece"]=z[-1]
    rr["final_class"]=ST[z[-1]]
    rr["fold"]=folds[rr["bifolium"]]

def plenbin(n):
    return 1 if n<=1 else (2 if n==2 else (3 if n==3 else 4))
def charbin(n):
    return 0 if n<=2 else (1 if n<=4 else (2 if n<=6 else 3))

raw=collections.OrderedDict()
for rr in rows:
    n=int(rr["folio"][1:].split("r")[0].split("v")[0])
    if not (103<=n<=116): continue
    raw.setdefault((rr["folio"],rr["line"]),[]).append(rr)

parent=r4["real_lines"]()
audit=[]
family=collections.Counter(); finalc=collections.Counter(); pcb=collections.Counter(); lnb=collections.Counter()
mapped_events=0
for seq in parent:
    fol=seq[0]["folio"]; line=seq[0]["line"][1] if isinstance(seq[0].get("line"),tuple) else None
    # parent line key is preserved as (folio,line)
    if line is None:
        raise RuntimeError(("missing_parent_line_key",fol))
    rr=raw.get((fol,line))
    if rr is None:
        raise RuntimeError(("raw_line_missing",fol,line))
    scored=rr[1:]
    if len(scored)!=len(seq):
        raise RuntimeError(("line_length_mismatch",fol,line,len(scored),len(seq)))
    # history includes opener rr[0], then each current scored token.
    hist=[rr[0]]
    for t,(e,cur) in enumerate(zip(seq,scored)):
        prev=hist[-1]
        if int(e["y"])!=int(cur["start"]):
            raise RuntimeError(("y_mismatch",fol,line,t,e["y"],cur["start"]))
        if int(e["prev"])!=int(prev["start"]):
            raise RuntimeError(("prev_mismatch",fol,line,t,e["prev"],prev["start"]))
        pbin=plenbin(len(prev["pieces"])); cbin=charbin(len(prev["token"]))
        fam=f'{prev["start"]}:{prev["final_class"]}:{cbin}'
        finalc[prev["final_class"]]+=1; pcb[pbin]+=1; lnb[cbin]+=1; family[fam]+=1
        rec={
          "folio":fol,"line":line,"t":t,"fold":int(cur["fold"]),
          "y":int(cur["start"]),"prev_start":int(prev["start"]),
          "prev_final_class":int(prev["final_class"]),
          "prev_piece_count_bin":int(pbin),"prev_char_len_bin":int(cbin),
          "prev_family":fam
        }
        audit.append(rec); mapped_events+=1
        hist.append(cur)

blob=json.dumps(audit,sort_keys=True,separators=(",",":")).encode()
out={
 "programme":"VMS-R1A0",
 "status":"complete",
 "n_parent_lines":len(parent),
 "mapped_events":mapped_events,
 "sha256":hashlib.sha256(blob).hexdigest(),
 "final_class_cardinality":len(finalc),
 "piece_count_bins":dict(sorted(pcb.items())),
 "char_length_bins":dict(sorted(lnb.items())),
 "coarse_family_cardinality":len(family),
 "top_families":family.most_common(20),
 "fold_events":dict(sorted(collections.Counter(x["fold"] for x in audit).items()))
}
print("R1A0_ALIGNMENT="+json.dumps(out,separators=(",",":")),flush=True)
