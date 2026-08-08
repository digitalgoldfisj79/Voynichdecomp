#!/usr/bin/env python3
from __future__ import annotations
import ast, csv, hashlib, json, os, re, time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
HERE = Path(__file__).resolve().parent.parent
COMP_MANIFESTS = [HERE / "stage5_confound_manifest.tsv", HERE / "stage5_singlepage_manifest.tsv"]
TARGET_MANIFEST = HERE / "stage5_vms_text_refs.tsv"
PROMPT = (
    'Describe only the visible morphology and geometry of this isolated manuscript illustration crop. '
    'Do not identify or guess the manuscript, place, date, artist, language, subject name, species, zodiac sign, '
    'culture, or relationship to any known manuscript. Do not use proper nouns. Ignore text content and colour '
    'unless colour itself defines a visible boundary. Focus on topology and arrangement: shapes, branching, '
    'leaves/stems/roots if present, circles/rings/spokes/connectors, enclosures, towers/roofs/walls/flags, '
    'figures/vessels, repeated elements, symmetry and relative layout. Return JSON only: '
    '{"description":"8–35 neutral words","tags":["up to 8 short morphology tags"]}.'
)
FORBIDDEN = [
    "voynich","beinecke","yale","vatican","vatlib","bodleian","oxford","munich","münchen","bavaria",
    "lombardy","milan","pavia","venice","venetian","padua","paduan","trento","trent","walsperger",
    "fontana","fra mauro","pizzig","bianco","carrara","casanatense","sloane","roccabonella","de virga",
    "regensburg","german","italian","latin","medieval"
]
HEADERS = {"User-Agent":"VoynichCorridorResearch/0.1 uniform-blind-description"}


def load_rows():
    out=[]
    for path in COMP_MANIFESTS:
        with path.open(newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                bbox=ast.literal_eval(r["bbox_1000"])
                key=hashlib.sha256((r["candidate_key"]+"|"+r["image_url"]+"|"+json.dumps(bbox,separators=(",",":"))).encode()).hexdigest()[:24]
                out.append({"item_key":key,"source_key":r["candidate_key"],"is_target":False,"object_class":None,"image_url":r["image_url"],"bbox":bbox})
    with TARGET_MANIFEST.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            bbox=None if r["bbox_1000"]=="FULL" else ast.literal_eval(r["bbox_1000"])
            key="vms_"+r["reference_key"]
            out.append({"item_key":key,"source_key":r["reference_key"],"is_target":True,"object_class":r["object_class"],"image_url":r["image_url"],"bbox":bbox})
    # Exact duplicate check; comparator 78 + target 10 expected.
    assert len(out)==88, len(out)
    assert len({x["item_key"] for x in out})==88
    return out


def fetch(url, tries=4):
    last=None
    for k in range(tries):
        try:
            rr=requests.get(url,headers=HEADERS,timeout=45)
            rr.raise_for_status()
            return Image.open(BytesIO(rr.content)).convert("RGB")
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def crop(im,bbox):
    if bbox is None: return im
    w,h=im.size; x0,y0,x1,y1=bbox
    px=(max(0,round(x0*w/1000)),max(0,round(y0*h/1000)),min(w,round(x1*w/1000)),min(h,round(y1*h/1000)))
    if px[2]<=px[0] or px[3]<=px[1]: raise ValueError(f"empty crop {px}")
    return im.crop(px)


def prepare(im):
    w,h=im.size; s=max(w,h)
    sq=Image.new("RGB",(s,s),(255,255,255)); sq.paste(im,((s-w)//2,(s-h)//2))
    # Fixed bounded presentation. Upscale small crops; downscale huge crops.
    if s != 768:
        sq=sq.resize((768,768), Image.Resampling.LANCZOS)
    return sq


def parse_json(txt):
    txt=txt.strip()
    m=re.search(r"\{.*\}",txt,re.S)
    if not m: raise ValueError("no JSON object")
    obj=json.loads(m.group(0))
    d=str(obj.get("description","")).strip()
    tags=obj.get("tags",[])
    if not d or not isinstance(tags,list): raise ValueError("bad fields")
    tags=[str(x).strip().lower() for x in tags[:8] if str(x).strip()]
    leak=[x for x in FORBIDDEN if re.search(r"\b"+re.escape(x)+r"\b",d.lower())]
    return d,tags,leak


def describe(im,processor,model,device):
    messages=[{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":PROMPT}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    image_inputs,video_inputs=process_vision_info(messages)
    inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=120,do_sample=False)
    trimmed=[o[len(i):] for i,o in zip(inputs["input_ids"],generated)]
    return processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]


def main():
    rows=load_rows()
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor=AutoProcessor.from_pretrained(MODEL,token=token,min_pixels=224*224,max_pixels=1024*1024)
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,token=token,torch_dtype=dtype,device_map="auto")
    model.eval(); device=next(model.parameters()).device
    print("CONFIG="+json.dumps({"model":MODEL,"n":len(rows),"prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),"device":str(device)}),flush=True)
    cache={}; results=[]; failures=[]
    for idx,r in enumerate(rows):
        try:
            if r["image_url"] not in cache: cache[r["image_url"]]=fetch(r["image_url"])
            im=prepare(crop(cache[r["image_url"]],r["bbox"]))
            raw=describe(im,processor,model,device)
            d,tags,leak=parse_json(raw)
            status="LEAK" if leak else "OK"
            rec={"item_key":r["item_key"],"source_key":r["source_key"],"is_target":r["is_target"],"object_class":r["object_class"],"status":status,"description":d,"tags":tags,"leak_terms":leak}
            results.append(rec)
            print("DESC="+json.dumps(rec,ensure_ascii=False,separators=(",",":")),flush=True)
        except Exception as e:
            rec={"item_key":r["item_key"],"source_key":r["source_key"],"is_target":r["is_target"],"error":repr(e)}
            failures.append(rec); print("DESC_ERROR="+json.dumps(rec,separators=(",",":")),flush=True)
    good=[x for x in results if x["status"]=="OK"]
    comp_good=sum(not x["is_target"] for x in good); target_good=sum(x["is_target"] for x in good)
    canon="\n".join(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")) for x in sorted(results,key=lambda z:z["item_key"]))
    summary={"requested":len(rows),"results":len(results),"failures":len(failures),"ok":len(good),"comparator_ok":comp_good,"target_ok":target_good,"leaks":sum(x["status"]=="LEAK" for x in results),"result_sha256":hashlib.sha256(canon.encode()).hexdigest()}
    print("SUMMARY="+json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__": main()
