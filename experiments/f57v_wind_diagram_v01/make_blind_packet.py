#!/usr/bin/env python3
"""Create a blinded filename packet and sealed key.

Copies image files under random opaque IDs without altering pixels.
Input CSV columns: id,path
"""
import argparse, csv, random, secrets, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input_csv"); ap.add_argument("out_dir"); ap.add_argument("--key",default="blind_key.csv"); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    with open(args.input_csv,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    random.SystemRandom().shuffle(rows); key=[]
    for i,r in enumerate(rows,1):
        src=Path(r["path"]); ext=src.suffix.lower() or ".img"; opaque=f"IMG_{i:04d}_{secrets.token_hex(3)}"; dst=out/(opaque+ext)
        shutil.copy2(src,dst); key.append({"opaque_id":opaque,"original_id":r["id"],"original_path":str(src),"blind_path":str(dst)})
    if not key: raise ValueError("no images")
    with open(args.key,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=key[0].keys()); w.writeheader(); w.writerows(key)
    print(f"wrote {len(key)} blinded images to {out}"); print(f"sealed key: {args.key}")
if __name__=="__main__": main()
