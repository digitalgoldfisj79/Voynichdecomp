#!/usr/bin/env python3
"""Historical Yiddish-German L calibration transfer v0.2.

Prospectively frozen transfer test after v0.1b. This wrapper reuses the audited
v0.1 solver machinery but:
- uses the source-only frozen 8+8 L-unseen transfer panel;
- hard-gates all selected source hashes/counts;
- uses fresh v0.2 seeds/keys/aliases and no prior solver outputs;
- aggregates with the preregistered independence-conservative contrast
  Z_ind = [(sY-muY)-(sG-muG)] / sqrt(sdY^2+sdG^2).

Voynich/C7 is never loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import pickle
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_RUNNER = HERE.parent / "yiddish_l_recovery_v01" / "run_l_recovery_v01.py"

spec = importlib.util.spec_from_file_location("lrec_v01_parent_for_v02", PARENT_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen parent runner: {PARENT_RUNNER}")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

VERSION = "yiddish_german_l_calibration_transfer_v02_20260912"
PANEL_FREEZE_COMMIT = "7533f2a92b8a8d4d7f29f84349a1caa4ec30aae3"
PRE_CENSUS_PROTOCOL_COMMIT = "8d4534fc427767d224a4ee8ae8f2b7a3524665ee"
CENSUS_RUN_ID = 34717339316
CENSUS_ARTIFACT_ID = 10305995314
CENSUS_SELECTION_SHA256 = "e74e0296446f01124c4ee339f58053b8902927725484046a37b6f3a6aa16fc3a"
REF_ARCHIVE_SHA256 = "288a478d02de5796d14faa0b669879dc6bf67c47e2d7ee261815e15120b99f0e"

YID_BUILD = {
    "shir_1579": ["1579e-shir-preface.psd", "1579e-shir.psd"],
    "ester_1589": ["1589e-ester-preface.psd", "1589e-ester.psd"],
}
YID_DEV = {
    "magid_1600": ["1600e-magid-preface.psd", "1600e-magid.psd"],
    "magen_1624": ["1624e-magen.psd"],
    "messiah_1666": ["1666w-messiah.psd"],
    "ashkenaz_un_polak_1675": ["1675e-ashkenaz-un-polak.psd"],
    "vilna_1692": ["1692e-vilna.psd"],
    "purim_1697": ["1697e-purim.psd"],
    "glikl_1705": ["1705w-glikl.psd"],
    "moses_1750": ["1750w-moses.psd"],
}
GER_BUILD = ("F014", "F015")
GER_DEV = ("F004", "F028", "F057", "F128", "F166", "F249", "F300", "F313")
GROUP_ALIASES = tuple(f"G{i:02d}" for i in range(16))

PENN_EXPECTED = {
    "shir_1579": {
        "1579e-shir-preface.psd": "1e4cd69062c47ea0753e7e1185d58c1bcd9bbcf71e1b3534564c159def1161bd",
        "1579e-shir.psd": "f3f9e2a5625d35ef1efa2d552986b771ff4cadacd95f17857b557cc02119bf17",
    },
    "ester_1589": {
        "1589e-ester-preface.psd": "1a4d0bde424daa8d0519ca756a7d8376a1d3c1b3c9c0c5c4fd83270cf7cef0a0",
        "1589e-ester.psd": "e8a88bfeff5e50e92f1d88050aba9492abfc13a07121093c07d5b7dd49cbd114",
    },
    "magid_1600": {
        "1600e-magid-preface.psd": "a2a616f82b267bfec0de2fb487a576071fce1bbc8d4d1e04b413c06f2f2dd1dd",
        "1600e-magid.psd": "e0df6d1421bee337bf142b3e8c46c3eea7e8454c21b057d55fe724580e7a819b",
    },
    "magen_1624": {"1624e-magen.psd": "fdd2360685bbb387faec51ad3a297d7736a70d62410923f5e4c49dfe4124dae7"},
    "messiah_1666": {"1666w-messiah.psd": "676bcb99e5247c1a992bfc815d42c11ad5554539a785656e63f1b9cae364a2de"},
    "ashkenaz_un_polak_1675": {"1675e-ashkenaz-un-polak.psd": "6f69a7ce05b121d0b573fe94397e74da217ce6d92b594c01a846f19a8d6d29e9"},
    "vilna_1692": {"1692e-vilna.psd": "a6cf77dee968028746bbb474893fdff08f9e4de24500a485a7f91c1c89594919"},
    "purim_1697": {"1697e-purim.psd": "5b9a61cd91bd5d19666429cc9860373d048268759b8bd0055d475538ff3c3e98"},
    "glikl_1705": {"1705w-glikl.psd": "478aca9fdc19edb06839abda3a774bb9e8e76fec4895fb4602040f770798313d"},
    "moses_1750": {"1750w-moses.psd": "82c232d550f55a1875b2fe3687d0fd3b4532226eadb30ca22e6cabf75c345a62"},
}
YID_EXPECTED_WORDS = {
    "magid_1600": 1208, "magen_1624": 1904, "messiah_1666": 2821,
    "ashkenaz_un_polak_1675": 3052, "vilna_1692": 2207, "purim_1697": 5859,
    "glikl_1705": 2713, "moses_1750": 2416,
}
REF_EXPECTED = {
    "F014": {"sha256":"059211745db8c12b96d9ac4538cd94201c758f47cbc756b5fca70eeb59872f76","words":19097},
    "F015": {"sha256":"29b07d566cc8f3ceedbac16a5a0f98a9a4d8059b748c53d336d97aaffcd250f9","words":18162},
    "F004": {"sha256":"1ef0d95e2af0d6825bdcf38bfaeca5a4b7731d544a8de08f1465106e0ed026b7","words":2950},
    "F028": {"sha256":"a9954ad2f7e2f6310663856c0b6f9b9bf1713ba072a6edcb181773c62153bfa2","words":2047},
    "F057": {"sha256":"88240f0bf2e4d632f5aefcca82ff75ef0917be8cb0f2c93d2be6a0ad76660d95","words":2875},
    "F128": {"sha256":"00bbbce794c0d4c2aad43cc6e8dab5acccb57d9ce0c973526a38eec1e664d060","words":5865},
    "F166": {"sha256":"57c819335d820a0d8af14f50363ca4b9de8c90bf38a58f9736a1d0506fefe5c9","words":1231},
    "F249": {"sha256":"b7d70581a93096ca21c41818f94983e9e5ff4ba8c54f5055cdcbfd91b2e89392","words":2452},
    "F300": {"sha256":"30af3d1d375f73bb5385490faf5ca2cad6ee658d6c77e094ecf691cc55535d71","words":2018},
    "F313": {"sha256":"a51c23e24af368b7f39246c82e50b18c0d098649051bf731405d87bf68d51576","words":3325},
}

def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def ref_words_v02(root: Path, wid: str):
    if wid not in REF_EXPECTED:
        raise RuntimeError(f"unregistered ReF ID: {wid}")
    exp = REF_EXPECTED[wid]
    candidates = sorted(p for p in Path(root).rglob("*.xml") if p.name.lower() == f"{wid.lower()}.xml")
    exact = [p for p in candidates if core.sha_file(p) == exp["sha256"]]
    if len(exact) != 1:
        seen = [{"path":str(p),"sha256":core.sha_file(p)} for p in candidates]
        raise RuntimeError(f"{wid}: exact hash-pinned XML count={len(exact)} candidates={seen}")
    p = exact[0]
    xmlroot = ET.parse(p).getroot()
    out = []
    for tok in xmlroot.iter():
        if localname(tok.tag) != "token":
            continue
        raw = "".join(
            (child.attrib.get("utf") or (child.text or ""))
            for child in list(tok) if localname(child.tag) == "tok_dipl"
        )
        w = core.normalize_latin(raw)
        if w:
            out.append(w)
    if len(out) != exp["words"]:
        raise RuntimeError(f"{wid}: {len(out)} words != frozen {exp['words']}")
    return out, p

core.VERSION = VERSION
core.HERE = HERE
core.__file__ = str(Path(__file__).resolve())
core.YID_BUILD = YID_BUILD
core.YID_DEV = YID_DEV
core.GER_BUILD = GER_BUILD
core.GER_DEV = GER_DEV
core.GROUP_ALIASES = GROUP_ALIASES
core.ref_words = ref_words_v02

_parent_preflight = core.preflight

def preflight_v02(args):
    _parent_preflight(args)
    pub = Path(args.out) / "public"
    manifest_path = pub / "preflight_manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    obs_penn = m["source_hashes"]["penn"]
    for fam, files in PENN_EXPECTED.items():
        rows = obs_penn.get(fam)
        if rows is None:
            raise RuntimeError(f"{fam}: missing PPCHY source hashes")
        flat = {}
        for row in rows:
            flat.update(row)
        if flat != files:
            raise RuntimeError(f"{fam}: PPCHY hash gate failed observed={flat} expected={files}")
    dev_counts = m["development_counts"]["yiddish"]
    for fam, n in YID_EXPECTED_WORDS.items():
        if dev_counts.get(fam) != n:
            raise RuntimeError(f"{fam}: normalized word count {dev_counts.get(fam)} != frozen {n}")
    obs_ref = m["source_hashes"]["ref"]
    for wid, exp in REF_EXPECTED.items():
        row = obs_ref.get(wid)
        if row is None or row.get("sha256") != exp["sha256"] or row.get("words") != exp["words"]:
            raise RuntimeError(f"{wid}: ReF frozen gate failed: {row}")
    if any(r["shared_8gram_types"] for r in m["overlap_yiddish"] + m["overlap_german"]):
        raise RuntimeError("frozen BUILD↔TRANSFER 8-word leakage gate failed")
    m.update({
        "status": "V02_PREFLIGHT_FROZEN_BEFORE_SOLVER_OUTCOMES",
        "transfer_protocol": "yiddish_l_calibration_v02/PROTOCOL.md",
        "panel_freeze_commit": PANEL_FREEZE_COMMIT,
        "pre_census_protocol_commit": PRE_CENSUS_PROTOCOL_COMMIT,
        "census_run_id": CENSUS_RUN_ID,
        "census_artifact_id": CENSUS_ARTIFACT_ID,
        "census_selection_json_sha256": CENSUS_SELECTION_SHA256,
        "ref_archive_sha256_expected": REF_ARCHIVE_SHA256,
        "transfer_statistic": "Z_ind=((sY-muY)-(sG-muG))/sqrt(sdY^2+sdG^2)",
        "family_call_abs_z_ind": 1.0,
        "prior_solver_outputs_reused": False,
        "fresh_v02_seed_namespace": VERSION,
        "scope": "L_UNSEEN_TRANSFER_ONLY__SECONDARY_NORMALIZED_REPRESENTATION",
        "target_loaded": False,
        "voynich_access_allowed": False,
    })
    core.atomic_json(m, manifest_path)
    print(json.dumps({
        "status":"V02_PREFLIGHT_OK",
        "groups":len(GROUP_ALIASES),
        "source_hash_count_yiddish":sum(len(x) for x in PENN_EXPECTED.values()),
        "source_hash_count_ref":len(REF_EXPECTED),
        "prior_solver_outputs_reused":False,
        "target_loaded":False,
    }, indent=2))

core.preflight = preflight_v02

def z_ind_from_arm(arm: dict, ymid: str, gmid: str):
    y, g = arm[ymid], arm[gmid]
    vals = [y.get("score"),y.get("null_mean"),y.get("null_sd"),g.get("score"),g.get("null_mean"),g.get("null_sd")]
    if any(v is None or not math.isfinite(v) for v in vals):
        return {"D":None,"denom":None,"z_ind":None}
    den = math.sqrt(y["null_sd"]**2 + g["null_sd"]**2)
    if not math.isfinite(den) or den <= 0:
        return {"D":None,"denom":den,"z_ind":None}
    d = (y["score"]-y["null_mean"]) - (g["score"]-g["null_mean"])
    return {"D":d,"denom":den,"z_ind":d/den}

def call(z: float | None, threshold: float = 1.0):
    if z is None or not math.isfinite(z):
        return "abstain"
    if z >= threshold:
        return "yiddish"
    if z <= -threshold:
        return "german"
    return "abstain"

def exact_label_null(fams: list[dict], threshold: float = 1.0):
    vals=[x["median_z_ind"] for x in fams]
    labels=[x["truth"] for x in fams]
    ny=sum(x=="yiddish" for x in labels)
    n=len(labels)
    preds=[call(v,threshold) for v in vals]
    obs=sum(p==t for p,t in zip(preds,labels))/n
    null=[]
    for comb in itertools.combinations(range(n),ny):
        S=set(comb)
        labs=["yiddish" if i in S else "german" for i in range(n)]
        null.append(sum(p==t for p,t in zip(preds,labs))/n)
    mu=statistics.mean(null)
    sd=statistics.pstdev(null)
    p=sum(v>=obs-1e-15 for v in null)/len(null)
    return {
        "accuracy":obs,"null_mean":mu,"null_sd":sd,
        "effect_over_null_mean":obs-mu,
        "effect_over_null_sd":(obs-mu)/sd if sd else None,
        "exact_one_sided_p":p,"assignments":len(null),
    }

def aggregate_v02(args):
    pub=Path(args.public); priv=Path(args.private); sol=Path(args.solutions); out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    truth=pickle.loads((priv/"truth.pkl").read_bytes())
    model_truth=truth["model_truth"]; group_truth=truth["group_truth"]
    ymid=next(k for k,v in model_truth.items() if v=="yiddish")
    gmid=next(k for k,v in model_truth.items() if v=="german")
    allrows=[]
    for group in GROUP_ALIASES:
        p=sol/group/"rows.jsonl"
        if not p.exists():
            cand=list(sol.rglob(f"{group}/rows.jsonl"))
            if cand: p=cand[0]
        if not p.exists():
            raise RuntimeError(f"missing solver rows: {group}")
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows)!=core.N_KEYS:
            raise RuntimeError(f"{group}: {len(rows)} rows != {core.N_KEYS}")
        pcases={x["key_index"]:x for x in json.loads((pub/f"{group}.json").read_text())["cases"]}
        prows={x["key_index"]:x for x in truth["cases"][group]["rows"]}
        lang=group_truth[group]["language"]
        cm=ymid if lang=="yiddish" else gmid
        wm=gmid if lang=="yiddish" else ymid
        for r in rows:
            ki=r["key_index"]; pc=pcases[ki]; pr=prows[ki]
            for arm,cipherkey,truthkey in [
                ("primary","audit_cipher","audit_truth"),
                ("n2","n2_audit_cipher","n2_audit_truth"),
            ]:
                for mid in ("A","B"):
                    rr=core.rec_score(pr[truthkey],pc[cipherkey],r[arm][mid]["mapping"])
                    r[arm][mid].update(rr)
                    r[arm][mid]["m0_pass"]=rr["atom_recovery"]>=core.ATOM_PASS and rr["word_recovery"]>=core.WORD_PASS
                r[arm]["v02_contrast"]=z_ind_from_arm(r[arm],ymid,gmid)
            r["n1"]["v02_contrast"]=z_ind_from_arm(r["n1"],ymid,gmid)
            r["truth_language"]=lang
            r["family"]=group_truth[group]["family"]
            r["correct_model"]=cm
            r["wrong_model"]=wm
            allrows.append(r)
    core.atomic_pickle({"version":VERSION,"rows":allrows,"truth":truth},out/"checkpoint.pkl")
    with (out/"scored_rows.jsonl").open("w",encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r,sort_keys=True)+"\n")
    families=[]
    for group in GROUP_ALIASES:
        rs=[r for r in allrows if r["group"]==group]
        if len(rs)!=core.N_KEYS: raise RuntimeError(f"{group}: incomplete aggregate cell")
        lang=rs[0]["truth_language"]; family=rs[0]["family"]
        cm=rs[0]["correct_model"]; wm=rs[0]["wrong_model"]
        fr={"group":group,"truth":lang,"family":family}
        for arm in ("primary","n1","n2"):
            zs=[r[arm]["v02_contrast"]["z_ind"] for r in rs]
            med=None if any(z is None or not math.isfinite(z) for z in zs) else statistics.median(zs)
            fr[arm]={"median_z_ind":med,"call":call(med),"correct":call(med)==lang}
        fr["primary"]["correct_recovery_passes"]=sum(r["primary"][cm]["m0_pass"] for r in rs)
        fr["primary"]["wrong_recovery_passes"]=sum(r["primary"][wm]["m0_pass"] for r in rs)
        fr["primary"]["mean_correct_atom"]=statistics.mean(r["primary"][cm]["atom_recovery"] for r in rs)
        fr["primary"]["mean_wrong_atom"]=statistics.mean(r["primary"][wm]["atom_recovery"] for r in rs)
        families.append(fr)
    def arm_summary(arm:str, threshold:float=1.0):
        ff=[]
        for x in families:
            z=x[arm]["median_z_ind"]; c=call(z,threshold)
            ff.append({"truth":x["truth"],"family":x["family"],"median_z_ind":z,"call":c})
        acc=sum(x["call"]==x["truth"] for x in ff)/len(ff)
        return {
            "accuracy":acc,
            "errors":sum(x["call"]!=x["truth"] for x in ff),
            "abstentions":sum(x["call"]=="abstain" for x in ff),
            "families":ff,
            "null":exact_label_null(ff,threshold),
        }
    primary=arm_summary("primary"); n1=arm_summary("n1"); n2=arm_summary("n2")
    complete=all(r[arm]["v02_contrast"]["z_ind"] is not None for r in allrows for arm in ("primary","n1","n2"))
    gates={
        "correct_language_recovery":all(x["primary"]["correct_recovery_passes"]>=core.CELL_PASS for x in families),
        "wrong_language_selectivity":all(x["primary"]["wrong_recovery_passes"]<core.CELL_PASS for x in families),
        "deployable_accuracy":primary["accuracy"]>=0.80,
        "matched_label_null_2sd":primary["null"]["effect_over_null_sd"] is not None and primary["null"]["effect_over_null_sd"]>=2.0,
        "beats_unigram_by_2_errors":primary["errors"]+2<=n1["errors"],
        "beats_shuffle_by_2_errors":primary["errors"]+2<=n2["errors"],
        "complete_non_degenerate":complete,
    }
    loo=[]
    for drop in range(len(families)):
        subset=[x for i,x in enumerate(families) if i!=drop]
        ff=[{"truth":x["truth"],"family":x["family"],"median_z_ind":x["primary"]["median_z_ind"]} for x in subset]
        for x in ff: x["call"]=call(x["median_z_ind"])
        acc=sum(x["call"]==x["truth"] for x in ff)/len(ff)
        ns=exact_label_null(ff,1.0)
        ok=acc>=0.80 and ns["effect_over_null_sd"] is not None and ns["effect_over_null_sd"]>=2.0
        loo.append({"dropped":families[drop]["family"],"accuracy":acc,"null":ns,"primary_call_gates_pass":ok})
    gates["leave_one_family_out_stable"]=all(x["primary_call_gates_pass"] for x in loo) if gates["deployable_accuracy"] and gates["matched_label_null_2sd"] else False
    status="L_RECOVERY_V02_TRANSFER_PASS__FRESH_CONFIRMATION_STILL_REQUIRED" if all(gates.values()) else "L_RECOVERY_V02_TRANSFER_NOT_RESOLVED"
    sensitivity={str(t):arm_summary("primary",t) for t in (0.5,1.0,1.5)}
    pre=json.loads((pub/"preflight_manifest.json").read_text())
    summary={
        "status":status,"gates":gates,"primary":primary,
        "nuisance_unigram":n1,"nuisance_within_word_shuffle":n2,
        "families":families,"leave_one_out":loo,"sensitivity":sensitivity,
        "model_truth_revealed_after_solver_outputs":model_truth,
        "group_truth_revealed_after_solver_outputs":group_truth,
        "plant_root_commitment_sha256":pre["plant_root_commitment_sha256"],
        "plant_root_reveal_hex":truth["plant_root_hex"],
        "preflight_manifest_sha256":core.sha_file(pub/"preflight_manifest.json"),
        "protocol_sha256":pre["protocol_sha256"],
        "runner_sha256":pre["runner_sha256"],
        "statistic":"Z_ind=((sY-muY)-(sG-muG))/sqrt(sdY^2+sdG^2)",
        "scope":"L_UNSEEN_TRANSFER__SECONDARY_NORMALIZED__NO_TARGET",
        "voynich_loaded":False,
    }
    core.atomic_json(summary,out/"summary.json")
    ns=primary["null"]
    lines=[
        "# Yiddish–German L calibration transfer v0.2 — result","",
        "## RETRACTIONS / CONTROLLING LIMITS","",
        "- v0.1 remains permanently quarantined for the ReF token-unit defect.",
        "- v0.1b remains `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`; this result does not rewrite it.",
        "- The post-v0.1b same-map paired-null 10/10 diagnostic is non-controlling and was not used here.",
        "- This is an L-unseen transfer panel, not fresh whole-programme confirmation. Voynich was never loaded.","",
        f"**Status: `{status}`**","","## Frozen-gate outcome","",
    ]
    for k,v in gates.items(): lines.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")
    esd=ns["effect_over_null_sd"]
    lines += [
        "","## Primary headline","",
        f"Primary family accuracy = {primary['accuracy']:.3f}; effect over exact label null = {ns['effect_over_null_mean']:+.3f}, null SD = {ns['null_sd']:.6f}, effect/nullSD = {esd if esd is not None else 'NA'}; exact p = {ns['exact_one_sided_p']:.6f}.",
        f"Unigram nuisance accuracy = {n1['accuracy']:.3f}; within-word-shuffle nuisance accuracy = {n2['accuracy']:.3f}.",
        "","## Family results","",
        "| truth | family | median Z_ind | call | correct M0 passes | wrong-model M0 passes | unigram call | shuffle call |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]
    for x in sorted(families,key=lambda z:(z["truth"],z["family"])):
        z=x["primary"]["median_z_ind"]; zs="NA" if z is None else f"{z:+.4f}"
        lines.append(f"| {x['truth']} | {x['family']} | {zs} | {x['primary']['call']} | {x['primary']['correct_recovery_passes']}/32 | {x['primary']['wrong_recovery_passes']}/32 | {x['n1']['call']} | {x['n2']['call']} |")
    lines += [
        "","## Interpretation boundary","",
        "A full transfer pass still requires genuinely fresh external historical-Yiddish confirmation before L can be issued. A failure leaves L unresolved. No result here licenses target/Voynich scoring."
    ]
    (out/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":status,"gates":gates,"primary_accuracy":primary["accuracy"],
        "primary_null":primary["null"],"n1_accuracy":n1["accuracy"],"n2_accuracy":n2["accuracy"]
    },indent=2))

core.aggregate = aggregate_v02

if __name__ == "__main__":
    core.main()
