#!/usr/bin/env python3
"""Candidate-blind physical-image audit for Stage F.
Implements vms_edge_physical_image_20260907_v01.
"""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

from vms_spill_topology import PAGES,FOLIOS,BIFOLIA,fetch,edge_feature,vec
from vms_topology_marginalization import CORPORA,fetch_text,parse_ivtff

OUT=Path('artifacts/vms_edge_physical_image_v01'); OUT.mkdir(parents=True,exist_ok=True)
SALT='vms-edge-physical-v01-20260907'
CANDS=[('E1','q01_b3_6','q03_b17_24'),('E2','q06_b42_47','q01_b1_8')]

def opaque(s): return hashlib.sha256((SALT+'|'+s).encode()).hexdigest()[:12]

def scaler(X):
    X=np.asarray(X,float); med=np.median(X,axis=0); mad=np.median(np.abs(X-med),axis=0)
    sd=np.where(mad>1e-8,1.4826*mad,np.std(X,axis=0)+1e-8); sd=np.where(sd>1e-8,sd,1.0)
    return med,sd

def unitdist(leavesA,leavesB,Z):
    a1,a2=leavesA; b1,b2=leavesB
    def d(i,j): return float(np.sqrt(np.mean((Z[i]-Z[j])**2)))
    return min((d(a1,b1)+d(a2,b2))/2,(d(a1,b2)+d(a2,b1))/2)

def sig_for_unit(unit,fm,keys):
    uid,a,b=unit; out=[]
    for k in keys:
        vals=[]
        for f in (a,b):
            for side in ('r','v'):
                m=fm.get(f,{}); v=m.get(k)
                if v is not None: vals.append(v)
        out.append(tuple(sorted(set(vals))))
    return tuple(out)

def pclass(i,j,sigs): return tuple(sorted((repr(sigs[i]),repr(sigs[j]))))

def choose_alts(i,j,units,fm):
    n=len(units); allp=[(a,b) for a in range(n) for b in range(a+1,n) if a not in (i,j) and b not in (i,j)]
    for name,keys in [('A_LHI',('L','H','I')),('B_LH',('L','H')),('C_LI',('L','I'))]:
        sig=[sig_for_unit(u,fm,keys) for u in units]; c=pclass(i,j,sig); al=[p for p in allp if pclass(p[0],p[1],sig)==c]
        if len(al)>=20:return name,keys,al
    return 'D_ALL',(),allp

def lowstat(obs,vals):
    a=np.asarray(vals,float); m=float(a.mean()); sd=float(a.std(ddof=1)); z=(m-obs)/sd if sd>0 else None
    p=float((1+np.sum(a<=obs))/(len(a)+1))
    return {'observed':float(obs),'null_mean':m,'null_sd':sd,'effect':float(obs-m),'z_lower_closeness':None if z is None else float(z),'p_lower':p,'n_null':len(a)}

def main():
    # Metadata only for reveal/null matching, never image scoring.
    body=fetch_text(*CORPORA['ZL']); ft,fm,pt,pm=parse_ivtff(body)

    # Complete early-Q1-Q7 bifolia only.
    units=[]
    for q,pairs in BIFOLIA.items():
        for a,b in pairs:
            if a in FOLIOS and b in FOLIOS:
                uid=f'{q}_b{a}_{b}'; units.append((uid,a,b))
    units.sort(key=lambda u:opaque(u[0]))

    # Compute all page features BEFORE reveal.
    rows={}; hashes={}
    for label,iid in PAGES:
        im,hh=fetch(label,iid,900); hashes[label]=hh
        rows[label]={'top':edge_feature(im,'top'),'bottom':edge_feature(im,'bottom')}
        print('PAGE',opaque(label),flush=True)

    # Physical leaf records. Folio number is used only internally to join its two sides;
    # saved blind scores contain opaque node ids.
    leafids=sorted(FOLIOS)
    li={f:k for k,f in enumerate(leafids)}
    top=[]; bot=[]; ts=[]; bs=[]; shape=[]
    # Also save side components for generic leave-one-side-out sensitivity.
    comps={}
    for f in leafids:
        tr,tv=rows[f'f{f}r']['top'],rows[f'f{f}v']['top']; br,bv=rows[f'f{f}r']['bottom'],rows[f'f{f}v']['bottom']
        tvr=np.concatenate([tv['grid'][:,::-1].ravel(),tv['profile'][::-1]]); bvr=np.concatenate([bv['grid'][:,::-1].ravel(),bv['profile'][::-1]])
        trv=vec(tr); brv=vec(br)
        top.append((trv+tvr)/2); bot.append((brv+bvr)/2)
        sr=np.array([tr[x] for x in ('mean','p90','area015','area025')],float); sv=np.array([tv[x] for x in ('mean','p90','area015','area025')],float)
        qr=np.array([br[x] for x in ('mean','p90','area015','area025')],float); qv=np.array([bv[x] for x in ('mean','p90','area015','area025')],float)
        ts.append((sr+sv)/2); bs.append((qr+qv)/2)
        shape.append(np.array([tr['edge_y_mean'],tr['edge_y_sd'],tv['edge_y_mean'],tv['edge_y_sd'],br['edge_y_mean'],br['edge_y_sd'],bv['edge_y_mean'],bv['edge_y_sd']],float))
        comps[f]={'top_r':trv,'top_v':tvr,'bottom_r':brv,'bottom_v':bvr,'ts_r':sr,'ts_v':sv,'bs_r':qr,'bs_v':qv}
    top=np.asarray(top); bot=np.asarray(bot); ts=np.asarray(ts); bs=np.asarray(bs); shape=np.asarray(shape)
    mt,st=scaler(top); mb,sb=scaler(bot); mts,sts=scaler(ts); mbs,sbs=scaler(bs); msh,ssh=scaler(shape)
    Zt=(top-mt)/st; Zb=(bot-mb)/sb; Zts=(ts-mts)/sts; Zbs=(bs-mbs)/sbs; Zsh=(shape-msh)/ssh

    unit_leaves=[(li[a],li[b]) for uid,a,b in units]; n=len(units)
    mats={k:np.zeros((n,n)) for k in ('top','bottom','topspec','topscalar','bottomscalar','topscalar_spec','shape')}
    for i in range(n):
        for j in range(i+1,n):
            dt=unitdist(unit_leaves[i],unit_leaves[j],Zt); db=unitdist(unit_leaves[i],unit_leaves[j],Zb)
            dts=unitdist(unit_leaves[i],unit_leaves[j],Zts); dbs=unitdist(unit_leaves[i],unit_leaves[j],Zbs); ds=unitdist(unit_leaves[i],unit_leaves[j],Zsh)
            vals={'top':dt,'bottom':db,'topspec':dt-db,'topscalar':dts,'bottomscalar':dbs,'topscalar_spec':dts-dbs,'shape':ds}
            for k,v in vals.items():mats[k][i,j]=mats[k][j,i]=v

    # blind all-pairs file before mapping reveal
    blind=[]
    for i in range(n):
        for j in range(i+1,n):
            blind.append({'a':opaque(units[i][0]),'b':opaque(units[j][0]),**{k:float(M[i,j]) for k,M in mats.items()}})
    (OUT/'blind_pair_scores.json').write_text(json.dumps({'protocol':'vms_edge_physical_image_20260907_v01','pairs':blind},indent=2,sort_keys=True))

    reveal={'units':[{'opaque':opaque(uid),'unit_id':uid,'folios':[a,b]} for uid,a,b in units]}
    (OUT/'reveal_map.json').write_text(json.dumps(reveal,indent=2,sort_keys=True))
    uidix={u[0]:i for i,u in enumerate(units)}

    results=[]
    for eid,ua,ub in CANDS:
        i,j=uidix[ua],uidix[ub]; level,keys,alts=choose_alts(i,j,units,fm)
        prof=lowstat(mats['topspec'][i,j],[mats['topspec'][p] for p in alts])
        scal=lowstat(mats['topscalar_spec'][i,j],[mats['topscalar_spec'][p] for p in alts])
        bottom=lowstat(mats['bottom'][i,j],[mats['bottom'][p] for p in alts])
        shp=lowstat(mats['shape'][i,j],[mats['shape'][p] for p in alts])

        # leave each candidate page side out; scaler fixed from full eligible dataset.
        dels=[]
        endpoints=[units[i],units[j]]
        for U in endpoints:
            uid,a,b=U
            for f in (a,b):
                for side in ('r','v'):
                    # copy baseline standardized arrays, replace affected physical leaf by surviving side only.
                    T=Zt.copy(); B=Zb.copy(); leaf=li[f]
                    tv=comps[f]['top_v'] if side=='r' else comps[f]['top_r']; bv=comps[f]['bottom_v'] if side=='r' else comps[f]['bottom_r']
                    T[leaf]=(tv-mt)/st; B[leaf]=(bv-mb)/sb
                    obs=unitdist(unit_leaves[i],unit_leaves[j],T)-unitdist(unit_leaves[i],unit_leaves[j],B)
                    # null remains endpoint-disjoint but recalculated in same T/B space (unchanged for endpoint-disjoint alts).
                    nv=[unitdist(unit_leaves[x],unit_leaves[y],T)-unitdist(unit_leaves[x],unit_leaves[y],B) for x,y in alts]
                    zz=lowstat(obs,nv); dels.append({'deleted':f'f{f}{side}','effect':zz['effect'],'z':zz['z_lower_closeness'],'p':zz['p_lower']})
        same_dir=all(d['effect']<0 for d in dels); n_ge2=sum(d['z'] is not None and d['z']>=2 for d in dels)
        p2=bool(prof['z_lower_closeness'] is not None and prof['z_lower_closeness']>=2 and prof['p_lower']<=.01 and scal['effect']<0 and same_dir and n_ge2>=6 and not(bottom['z_lower_closeness'] is not None and bottom['z_lower_closeness']>=2))
        results.append({'edge_id':eid,'unit_a':ua,'unit_b':ub,'null_level':level,'null_keys':list(keys),'n_alternatives':len(alts),'top_specific_profile':prof,'top_specific_scalar':scal,'bottom_raw_closeness':bottom,'edge_shape_compatibility':shp,'leave_one_surface_out':dels,'n_deletions_z_ge2':n_ge2,'all_deletions_same_direction':same_dir,'P2_QUANT_SIGNAL':p2})

    out={'protocol':'vms_edge_physical_image_20260907_v01','n_units':n,'n_page_sides':len(PAGES),'source_hashes':hashes,'results':results,'rule':'P2 requires profile z>=2,p<=.01; scalar same direction; all 8 deletions same direction with >=6/8 z>=2; bottom raw closeness <2 SD'}
    (OUT/'physical_reveal_results.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    lines=['# Physical-image Stage F closeout','', '## RETRACTIONS / BOUNDS','- This channel measures upper-edge state compatibility, not direct chronological adjacency.','- Existing f32v/f33r calibration is not candidate evidence.','']
    for r in results:
        p=r['top_specific_profile']; s=r['top_specific_scalar']; b=r['bottom_raw_closeness']
        lines.append(f"## {r['edge_id']} {r['unit_a']} ↔ {r['unit_b']}")
        lines.append(f"- null {r['null_level']} n={r['n_alternatives']}")
        lines.append(f"- TOP_SPECIFIC_PROFILE: effect {p['effect']:.6g}, null SD {p['null_sd']:.6g}, closeness z={p['z_lower_closeness']}, p={p['p_lower']}")
        lines.append(f"- TOP_SPECIFIC_SCALAR: effect {s['effect']:.6g}, null SD {s['null_sd']:.6g}, closeness z={s['z_lower_closeness']}, p={s['p_lower']}")
        lines.append(f"- BOTTOM raw: effect {b['effect']:.6g}, null SD {b['null_sd']:.6g}, closeness z={b['z_lower_closeness']}, p={b['p_lower']}")
        lines.append(f"- leave-one-surface: same direction={r['all_deletions_same_direction']}; z>=2 in {r['n_deletions_z_ge2']}/8")
        lines.append(f"- P2_QUANT_SIGNAL={r['P2_QUANT_SIGNAL']}")
        lines.append('')
    (OUT/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'results':[{'edge_id':r['edge_id'],'P2_QUANT_SIGNAL':r['P2_QUANT_SIGNAL'],'profile_z':r['top_specific_profile']['z_lower_closeness'],'profile_p':r['top_specific_profile']['p_lower']} for r in results]},sort_keys=True),flush=True)

if __name__=='__main__':main()
