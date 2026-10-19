/* GAI LF-T1: pure reproducible scorer. Input rows: [quire,folio,paragraphNo,pos,token].
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
  if (rows.length !== 4580 || docs.length !== 80)
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
        const d=candidates[k],h=candidates[ix[k]].hist;
        for(const a of d.anchors){
          if(!co.has(a))co.set(a,{c:new Float64Array(K),n:0});
          const z=co.get(a);
          for(let y=0;y<K;y++){z.c[y]+=h[y];z.n+=h[y];}
        }
      }
      return co;
    }
    function score(co){
      let gain=0,active=0;
      for(const x of test){
        let pt=0,na=0;
        for(const a of x.anchors){
          const z=co.get(a);if(!z)continue;
          pt+=(z.c[x.y]+35*pg[x.y])/(z.n+35);na++;
        }
        if(!na)pt=x.p1;else pt/=na;
        gain+=Math.log2((.8*x.p1+.2*pt)/x.p1);
        if(na)active++;
      }
      return {gain:gain/ntest,active};
    }
    const obs=score(fitTopic(false));
    return {held,train:tr.length,test:te.length,targets:ntest,anchors:hasAnchor,active:obs.active,
      entropy0:-s0/ntest,entropy1:-s1/ntest,obsGain:obs.gain,
      nullScore:()=>score(fitTopic(true)).gain};
  }
  const folds=groups.map(fitFold),N=folds.reduce((s,f)=>s+f.targets,0);
  const obs=folds.reduce((s,f)=>s+f.obsGain*f.targets,0)/N;
  const nulls=[];
  for(let j=0;j<199;j++){
    let g=0;for(const f of folds)g+=f.nullScore()*f.targets;
    nulls.push(g/N);
  }
  const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const nullMean=avg(nulls),sd=Math.sqrt(nulls.reduce((s,x)=>s+(x-nullMean)**2,0)/(nulls.length-1));
  const tailP=(1+nulls.filter(x=>x>=obs-1e-14).length)/(nulls.length+1);
  const m0=folds.reduce((s,f)=>s+f.entropy0*f.targets,0)/N;
  const m1=folds.reduce((s,f)=>s+f.entropy1*f.targets,0)/N;
  return {
    experiment:'GAI_LF_T1_PILOT', decision:'PILOT_FAIL_NOT_IDENTIFIABLE',
    corpus_rows:rows.length, corpus_folios:docs.length, groups, heldout_scored_tokens:N,
    m0_entropy_bits:m0,m1_entropy_bits:m1,m2_entropy_bits:m1-obs,
    m2_minus_m1_bits:obs,null_mean_bits:nullMean,null_sd_bits:sd,
    effect_minus_matched_null_bits:obs-nullMean,effect_to_null_sd:(obs-nullMean)/sd,
    null_upper_p:tailP,null_reps:nulls.length,
    null_percentiles:[.05,.5,.95].map(p=>[...nulls].sort((a,b)=>a-b)[Math.floor((nulls.length-1)*p)]),
    positive_quires:folds.filter(f=>f.obsGain>0).length,
    prereg_gate_pass:obs>=.02 && tailP<=.05 && folds.filter(f=>f.obsGain>0).length>=5,
    folds:folds.map(f=>({quire:f.held,scored:f.targets,folios:f.test,trained_folios:f.train,gain_bits:f.obsGain,active:f.active}))
  };
}
if(typeof module!=='undefined' && module.exports){
  module.exports=LFTransfer;
  if(require.main===module){
    const fs=require('fs');
    if(process.argv.length<3)throw Error('Pass data.json from export SQL');
    const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
    process.stdout.write(JSON.stringify(LFTransfer(Array.isArray(input)?input:input.rows),null,2)+'\n');
  }
}