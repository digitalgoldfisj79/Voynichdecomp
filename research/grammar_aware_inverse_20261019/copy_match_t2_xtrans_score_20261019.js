/* GAI T2 cross-quire matched-copy vs association: pure reproducible scorer. Input rows: [quire,folio,paragraphNo,pos,token].
 * Output is a pilot diagnostic, NOT an inferred source language.
 * Run: node lf_transfer_score_20261019.js data.json > results.json
 * data.json may be either raw rows or {"rows":[...]} from lf_transfer_export_20261019.sql.
 */
function LFTransfer(rows) {
  const docsMap = new Map();
  for (const [quire, folio, para, pos, tok] of rows) {
    if (!docsMap.has(folio)) docsMap.set(folio, {folio, quire, x:[]});
    docsMap.get(folio).x.push({para,pos,w:tok});
  }
  const docs = Array.from(docsMap.values());
  for (const d of docs) d.x.sort((a,b)=>a.para-b.para||a.pos-b.pos);
  const groups = [...new Set(docs.map(d=>d.quire))].sort();
  if (groups.join('') !== 'ABCDEFG') throw Error('Cohort not seven Herbal-A quires A-G');
  if (rows.length < 1 || docs.length !== 80)
    throw Error('Unexpected source population: tokens='+rows.length+' folios='+docs.length);
  function rng32(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return(x>>>0)/4294967296;};}
  const random=rng32(20261019);
  function ed1(a,b){
    if(a===b)return false;
    const n=a.length,m=b.length;if(Math.abs(n-m)>1)return false;
    let j=0,k=0,err=0;
    while(j<n&&k<m){
      if(a[j]===b[k]){j++;k++;continue;}
      if(++err>1)return false;
      if(n>m)j++;else if(n<m)k++;else{j++;k++;}
    }
    return err+(n-j)+(m-k)===1;
  }
  function fitFold(held){
    const tr=docs.filter(d=>d.quire!==held),te=docs.filter(d=>d.quire===held);
    const freq=new Map();
    for(const d of tr)for(const t of d.x)freq.set(t.w,(freq.get(t.w)||0)+1);
    const vocab=[...freq.entries()].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,96).map(x=>x[0]);
    const vmap=new Map(vocab.map((w,i)=>[w,i])); const OTHER=vocab.length,K=OTHER+1;
    const cat=w=>vmap.get(w)??OTHER;
    const count=new Float64Array(K),contexts=new Map(); let nt=0;
    for(const d of tr)for(let i=0;i<d.x.length;i++){
      const x=d.x[i],y=cat(x.w), key=(i?d.x[i-1].w.slice(0,2):'^')+'|'+(x.pos===1?'^':'_');
      count[y]++;nt++;
      if(!contexts.has(key))contexts.set(key,{n:0,c:new Float64Array(K)});
      const z=contexts.get(key);z.n++;z.c[y]++;
    }
    const pg=Array.from(count,x=>(x+.5)/(nt+.5*K));
    const kernel=Array.from({length:K},(_,src)=>{
      const out=new Float64Array(K);
      if(src===OTHER){out[OTHER]=1;return out;}
      const near=[];for(let t=0;t<K-1;t++)if(ed1(vocab[src],vocab[t]))near.push(t);
      out[src]=near.length?.85:1;
      for(const x of near)out[x]+=.15/near.length;
      return out;
    });
    const anchorsOf=d=>[...new Set(d.x.slice(0,15).map(x=>x.w))]
      .filter(w=>w.length>=4&&(freq.get(w)||0)>=2&&(freq.get(w)||0)<=35)
      .sort((a,b)=>freq.get(a)-freq.get(b)||a.localeCompare(b)).slice(0,10);
    const candidates=[];
    for(const d of tr){
      const hist=new Float64Array(K);
      for(const t of d.x.slice(15))hist[cat(t.w)]++;
      candidates.push({quire:d.quire,folio:d.folio,len:d.x.length-15,anchors:anchorsOf(d),hist});
    }
    const test=[];let s0=0,s1=0,ntest=0,hasAnchor=0;
    for(const d of te){
      const anchors=anchorsOf(d), past=[];
      for(let i=0;i<d.x.length;i++){
        const x=d.x[i],y=cat(x.w),key=(i?d.x[i-1].w.slice(0,2):'^')+'|'+(x.pos===1?'^':'_');
        const z=contexts.get(key);
        const pcond=z?(z.c[y]+30*pg[y])/(z.n+30):pg[y];
        const p0=.4*pg[y]+.6*pcond;
        let weight=0,acc=0,decay=1;
        for(let j=past.length-1;j>=0;j--){
          acc+=decay*kernel[past[j]][y];weight+=decay;decay*=.98;
        }
        const pc=(12*p0+acc)/(12+weight);
        const p1=.75*p0+.25*pc;
        if(i>=15){test.push({y,p0,p1,anchors,folio:d.folio});s0+=Math.log2(p0);s1+=Math.log2(p1);ntest++;if(anchors.length)hasAnchor++;}
        past.push(y);
      }
    }
    // TRAIN donor allocation and matched feature capacity.
    const testPages=new Map();
    for(const x of test){
      if(!testPages.has(x.folio))testPages.set(x.folio,{anchors:x.anchors,targets:[]});
      testPages.get(x.folio).targets.push(x);
    }
    for(const page of testPages.values()){
      page.donors=[];
      const aset=new Set(page.anchors);
      for(let k=0;k<candidates.length;k++){
        const d=candidates[k];
        let similarity=0;
        for(const a of d.anchors)if(aset.has(a))similarity+=1/freq.get(a);
        if(similarity>0&&d.len>0)page.donors.push({k,w:similarity});
      }
    }
    // Edit kernel consumes the SAME late-token histograms in both families.
    const changed=candidates.map(d=>{
      const c=new Float64Array(K);
      for(let i=0;i<K;i++)if(d.hist[i]>0){
        const val=d.hist[i],row=kernel[i];
        for(let j=0;j<K;j++)if(row[j])c[j]+=val*row[j];
      }
      return c;
    });
    function fitTopic(shuffle){
      const ix=candidates.map((_,i)=>i);
      if(shuffle){
        const by=new Map();
        for(let k=0;k<candidates.length;k++){
          const quire=candidates[k].quire;
          if(!by.has(quire))by.set(quire,[]);
          by.get(quire).push(k);
        }
        for(const ids of by.values()){
          ids.sort((a,b)=>candidates[a].len-candidates[b].len);
          for(let a=0;a<ids.length;a+=4){
            const block=ids.slice(a,a+4),donors=block.slice();
            for(let j=donors.length-1;j>0;j--){
              const k=Math.floor(random()*(j+1));
              [donors[j],donors[k]]=[donors[k],donors[j]];
            }
            for(let j=0;j<block.length;j++)ix[block[j]]=donors[j];
          }
        }
      }
      const co=new Map();
      for(let k=0;k<candidates.length;k++){
        const d=candidates[k],mapped=ix[k],h=changed[mapped],length=candidates[mapped].len;
        for(const a of d.anchors){
          if(!co.has(a))co.set(a,{c:new Float64Array(K),n:0});
          const z=co.get(a);
          for(let y=0;y<K;y++)z.c[y]+=h[y];
          z.n+=length;
        }
      }
      return {co,ix};
    }
    function score(model){
      let primary=0,gainCopy=0,gainAssoc=0,gainHybrid=0,nDonor=0;
      const co=model.co,ix=model.ix;
      for(const page of testPages.values()){
        const assoc=new Float64Array(K);
        let nmatch=0;
        for(const a of page.anchors){
          const z=co.get(a);if(!z)continue;
          nmatch++;
          for(let y=0;y<K;y++)assoc[y]+=(z.c[y]+35*pg[y])/(z.n+35);
        }
        if(nmatch)for(let y=0;y<K;y++)assoc[y]/=nmatch;
        let copy=null;
        if(page.donors.length){
          copy=new Float64Array(K);
          let totalWeight=0,N=0;
          for(const {k,w} of page.donors){
            const mapped=ix[k],length=candidates[mapped].len;
            if(length<=0)continue;
            totalWeight+=w;
            N+=length;
            for(let y=0;y<K;y++)copy[y]+=w*changed[mapped][y]/length;
          }
          if(totalWeight>0){
            for(let y=0;y<K;y++)
              copy[y]=((copy[y]/totalWeight)*N+35*pg[y])/(N+35);
          } else copy=null;
        }
        for(const x of page.targets){
          const pc=copy?(.8*x.p1+.2*copy[x.y]):x.p1;
          const pa=nmatch?(.8*x.p1+.2*assoc[x.y]):x.p1;
          const ph=.8*pc+.2*pa;
          primary+=Math.log2(pa/pc);
          gainCopy+=Math.log2(pc/x.p1);
          gainAssoc+=Math.log2(pa/x.p1);
          gainHybrid+=Math.log2(ph/pc);
          if(copy)nDonor++;
        }
      }
      return {primary:primary/ntest,gainCopy:gainCopy/ntest,
              gainAssoc:gainAssoc/ntest,gainHybrid:gainHybrid/ntest,
              donorScored:nDonor};
    }
    const obs=score(fitTopic(false));
    return {held,train:tr.length,test:te.length,targets:ntest,anchors:hasAnchor,
      donorScored:obs.donorScored,
      entropy0:-s0/ntest,entropy1:-s1/ntest,primary:obs.primary,
      gainCopy:obs.gainCopy,gainAssoc:obs.gainAssoc,gainHybrid:obs.gainHybrid,
      nullScore:()=>score(fitTopic(true)).primary};
  }
  const folds=groups.map(fitFold),N=folds.reduce((s,f)=>s+f.targets,0);
  const weighted=(key)=>folds.reduce((s,f)=>s+f[key]*f.targets,0)/N;
  const observed=weighted('primary'), copy=weighted('gainCopy'),assoc=weighted('gainAssoc'),hybrid=weighted('gainHybrid');
  const nulls=[];
  for(let j=0;j<199;j++){
    let value=0;
    for(const f of folds)value+=f.nullScore()*f.targets;
    nulls.push(value/N);
  }
  const mean=nulls.reduce((a,b)=>a+b,0)/nulls.length;
  const sd=Math.sqrt(nulls.reduce((a,b)=>a+(b-mean)**2,0)/(nulls.length-1));
  const p=(1+nulls.filter(x=>x>=observed-1e-14).length)/(nulls.length+1);
  const base=weighted('entropy1');
  return {
    experiment:'GAI_COPY_MATCH_T2',status:'PILOT_NOT_SOURCE_IDENTIFICATION',
    rows:rows.length,folios:docs.length,quires:groups,targets:N,
    grammar_m0_bits:weighted('entropy0'),local_M1_bits:base,
    cross_quire_copy_bits:base-copy,cross_quire_assoc_bits:base-assoc,
    hybrid_bits:base-copy-hybrid,
    copy_over_local_bits:copy,assoc_over_local_bits:assoc,
    assoc_over_copy_bits:observed,
    matched_null_mean_bits:mean,matched_null_sd_bits:sd,
    effect_vs_matched_null_bits:observed-mean,z:(observed-mean)/sd,
    permutation_upper_p:p,n_nulls:nulls.length,
    null_quantiles:[.05,.5,.95].map(q=>[...nulls].sort((a,b)=>a-b)[Math.floor((nulls.length-1)*q)]),
    positive_quires:folds.filter(f=>f.primary>0).length,
    gate_pass:observed>=.02 && observed-mean>=2*sd && p<=.05 &&
      folds.filter(f=>f.primary>0).length>=5,
    leave_one_quire_out_descriptive:folds.map(f=>({omit:f.held,
      assoc_over_copy_bits:(observed*N-f.primary*f.targets)/(N-f.targets)})),
    folds:folds.map(f=>({quire:f.held,test_folios:f.test,train_folios:f.train,
      scored:f.targets,donor_scored:f.donorScored,copy_over_local_bits:f.gainCopy,
      assoc_over_local_bits:f.gainAssoc,
      assoc_over_copy_bits:f.primary,hybrid_over_copy_bits:f.gainHybrid})),
    null_values:nulls,
    caution:'Observational early-late feature contrast, not distinguishability of source semantics and cross-quire copying. Synthetic full renderer not qualified.'
  };
}
if(typeof module!=='undefined'&&module.exports){
  module.exports=LFTransfer;
  if(require.main===module){
    const fs=require('fs');
    if(process.argv.length<3)throw Error('Pass data.json from export SQL');
    const obj=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
    process.stdout.write(JSON.stringify(LFTransfer(Array.isArray(obj)?obj:obj.rows),null,2)+'\n');
  }
}
