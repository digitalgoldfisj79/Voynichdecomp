"""Phase 2 measurement. One pickle per page (atomic, resumable). Usage: measure_pages.py <worker> <nworkers>"""
import sys, os, re, json, gzip, pickle, hashlib, subprocess, numpy as np
sys.path.insert(0,"/home/claude/inkdirect"); sys.path.insert(0,"/home/claude/spatial")
import instrument as ins, sharp, reg, reg2
OUT="/home/claude/inkdirect/pages"; TIF="/home/claude/tif"; URL="https://zodiackillerciphers.com/voynich-sid-tif/"
PAD=6; BAD=1|2|4|8|16|256|512
def main():
    wk,nw=int(sys.argv[1]),int(sys.argv[2]); os.makedirs(OUT,exist_ok=True)
    from vmsspatial import ink
    B,pages=reg.bundle(); I=ink.load_ink(); clean=set()
    Uu=I["units"]
    for fl,a,b in zip(Uu["flags"],Uu["first"],Uu["last"]):
        if (int(fl)&BAD)==0: clean.update(range(int(a),int(b)+1))
    files={}
    for f in open(f"{TIF}/listing.txt").read().split():
        m=re.match(r"^(f\d+[rv]\d*)(\.sid)?\.tif$",f)
        if m: files[m.group(1)]=f
    todo=sorted(p for p in files if p in pages and reg2.leaf(p))
    for k,p in enumerate(todo):
        if k%nw!=wk or os.path.exists(f"{OUT}/{p}.pkl"): continue
        path=f"{TIF}/{files[p]}"; keep=os.path.exists(path)
        try:
            if not keep: subprocess.run(["curl","-sS","-m","300","-o",path,URL+files[p]],check=True)
            sha=hashlib.sha256(open(path,"rb").read()).hexdigest(); L=reg.load_L(path)
            toks=pages[p]; bx=[g for _,_,g,_ in toks]; mm=reg2.leaf(p)["mm"]
            r=reg2.register(L,bx,mm); c=reg2.check(L,bx,r); rows=[]; H,W=L.shape
            for i,w,g,loc in toks:
                if i not in clean: continue
                x0,y0,x1,y1=[int(round(v)) for v in (g[0]*r["scale"]+r["dx"]-PAD,g[1]*r["scale"]+r["dy"]-PAD,g[2]*r["scale"]+r["dx"]+PAD,g[3]*r["scale"]+r["dy"]+PAD)]
                if x0<0 or y0<0 or x1>W or y1>H or x1-x0<12 or y1-y0<12: rows.append(dict(i=i,word=w,ok=False)); continue
                crop=L[y0:y1,x0:x1]; m=ins.measure(crop); s=sharp.gauge(crop)
                Bg=np.median(crop[crop>=np.median(crop)]); nf=float(((1-crop/Bg)>0.25).mean()) if Bg>0 else np.nan
                d=dict(i=i,word=w,locus=loc,ok=m is not None,box_h=float(g[3]-g[1]),box_w=float(g[2]-g[0]),n_frac=nf)
                if m: d.update(m)
                if s: d.update(sigma_hat=s["sigma_hat"])
                rows.append(d)
            res=dict(page=p,file=files[p],sha256=sha,shape=L.shape,mm=mm,reg={k:float(v) for k,v in r.items()},chk={k:float(v) for k,v in c.items()},n_tokens=len(toks),rows=rows)
        except Exception as e:
            res=dict(page=p,file=files[p],error=repr(e),rows=[])
        pickle.dump(res,open(f"{OUT}/{p}.pkl.tmp","wb")); os.replace(f"{OUT}/{p}.pkl.tmp",f"{OUT}/{p}.pkl")
        if not keep and os.path.exists(path): os.remove(path)
        print(p,res.get("error") or (round(res["reg"]["scale"],3),res["reg"]["dx"],res["reg"]["dy"],round(res["chk"]["ratio_halfline"],2),sum(x["ok"] for x in rows),len(rows)),flush=True)
main()
