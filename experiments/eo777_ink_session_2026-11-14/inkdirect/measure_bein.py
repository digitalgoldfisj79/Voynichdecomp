"""Phase 5 measurement on the 2014 Yale IIIF images. One pickle per page. Usage: measure_bein.py <worker> <nworkers>"""
import sys, os, json, gzip, pickle, hashlib, subprocess, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,"/home/claude/inkdirect"); import instrument as ins, sharp, reg, reg2
OUT="/home/claude/inkdirect/pages_bein"; TMP="/home/claude/bein/full"; UP=1.5; PAD=6; BAD=1|2|4|8|16|256|512
def main():
    wk,nw=int(sys.argv[1]),int(sys.argv[2]); os.makedirs(OUT,exist_ok=True); os.makedirs(TMP,exist_ok=True)
    from vmsspatial import ink
    B,pages=reg.bundle(); I=ink.load_ink(); clean=set(); Uu=I["units"]
    for fl,a,b in zip(Uu["flags"],Uu["first"],Uu["last"]):
        if (int(fl)&BAD)==0: clean.update(range(int(a),int(b)+1))
    S=json.load(gzip.open("/home/claude/ext/vms-spatial/data/manuscript/registrations.json.gz"))["sources"]["beinecke"]
    done2004={f[:-4] for f in os.listdir("/home/claude/inkdirect/pages")}
    todo=sorted(p for p in pages if p in S and S[p].get("ok") and p in done2004 and reg2.leaf(p))
    for k,p in enumerate(todo):
        if k%nw!=wk or os.path.exists(f"{OUT}/{p}.pkl"): continue
        e=S[p]; path=f"{TMP}/{e['image_id']}.png"
        try:
            if not os.path.exists(path): subprocess.run(["curl","-sS","-f","-m","300","-o",path,f"https://collections.library.yale.edu/iiif/2/{e['image_id']}/full/full/0/default.png"],check=True)
            sha=hashlib.sha256(open(path,"rb").read()).hexdigest(); L0=reg.load_L(path); fx=L0.shape[1]/e["image_size"][0]; L=ndi.zoom(L0,UP,order=3); H,W=L.shape
            M=np.array(e["matrix"],float); toks=pages[p]; mm=reg2.leaf(p)["mm"]
            def tb(g):
                c=np.array([[g[0],g[1],1],[g[2],g[1],1],[g[0],g[3],1],[g[2],g[3],1]],float)@M.T; c=c[:,:2]/c[:,2:3]*fx*UP
                return c[:,0].min(),c[:,1].min(),c[:,0].max(),c[:,1].max()
            T=reg2.thin_map(L,4); bxs=[tb(g) for _,_,g,_ in toks]; hh=np.median([b[3]-b[1] for b in bxs])
            def val(oy):
                v=[]
                for a,t,c,d in bxs:
                    x0,y0,x1,y1=[int(round(q/4)) for q in (a,t+oy,c,d+oy)]
                    if x0<0 or y0<0 or x1>T.shape[1] or y1>T.shape[0] or x1<=x0 or y1<=y0: continue
                    v.append(T[y0:y1,x0:x1].mean())
                return float(np.mean(v)) if v else np.nan
            ratio=val(0)/np.mean([val(-hh*0.5),val(hh*0.5)]); rows=[]
            for (i,w,g,loc),b in zip(toks,bxs):
                if i not in clean: continue
                x0,y0,x1,y1=[int(round(v)) for v in (b[0]-PAD,b[1]-PAD,b[2]+PAD,b[3]+PAD)]
                if x0<0 or y0<0 or x1>W or y1>H or x1-x0<12 or y1-y0<12: rows.append(dict(i=i,word=w,ok=False)); continue
                crop=L[y0:y1,x0:x1]; m=ins.measure(crop); s=sharp.gauge(crop); Bg=np.median(crop[crop>=np.median(crop)])
                d=dict(i=i,word=w,locus=loc,ok=m is not None,box_h=float(g[3]-g[1]),box_w=float(g[2]-g[0]),n_frac=float(((1-crop/Bg)>0.25).mean()) if Bg>0 else np.nan)
                if m: d.update(m)
                if s: d.update(sigma_hat=s["sigma_hat"])
                rows.append(d)
            res=dict(page=p,file=e["image_id"],sha256=sha,shape=L.shape,mm=mm,pxmm=float(mm*e["scale"]*fx*UP),reg=dict(scale=float(e["scale"]*fx*UP),dx=0.0,dy=0.0),chk=dict(ratio_halfline=float(ratio)),n_tokens=len(toks),rows=rows)
        except Exception as ex:
            res=dict(page=p,error=repr(ex),rows=[])
        pickle.dump(res,open(f"{OUT}/{p}.pkl.tmp","wb")); os.replace(f"{OUT}/{p}.pkl.tmp",f"{OUT}/{p}.pkl")
        if os.path.exists(path): os.remove(path)
        print(p,res.get("error") or (round(res["pxmm"],2),round(res["chk"]["ratio_halfline"],2),sum(x["ok"] for x in rows),len(rows)),flush=True)
main()
