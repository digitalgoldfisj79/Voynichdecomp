#!/usr/bin/env node
'use strict';
// Reproduction script for ZL3b-n.txt (13 May 2025); run: node currier_herbal_transfer.js ZL3b-n.txt > results.json
// Scope: paragraph (P) loci with certain lowercase transcription tokens, section I=H only.
// It tests raw trigram SUPPORT, not historical cross-word carry or source-mechanism identification.
const fs = require('fs');
if (process.argv.length < 3) { console.error('Usage: node currier_herbal_transfer.js ZL3b-n.txt'); process.exit(2); }
const source = fs.readFileSync(process.argv[2], 'utf8');
let meta={}, data={A:{},B:{}};
let all={A:{words:0,folios:new Set(),sections:{},hands:{}},B:{words:0,folios:new Set(),sections:{},hands:{}}};
let skipped={A:0,B:0};
for (const line of source.split(/\r?\n/)) {
  let h=line.match(/^<(f[^.>\s]+)>\s+<!([^>]+)>/);
  if (h) {const m={};for(const z of h[2].matchAll(/\$([A-Z])=([^\s>]+)/g))m[z[1]]=z[2];meta[h[1]]=m;continue;}
  let p=line.match(/^<(f[^.>\s]+)\.[^,>]+,([^>]+)>\s*(.*)$/);
  if (!p || !/[=@+*]P/.test(p[2])) continue;
  const m=meta[p[1]]||{};
  if(!['A','B'].includes(m.L))continue;
  const raw=p[3].replace(/<->/g,'?').replace(/<[^>]*>/g,'').replace(/\[[^\]]*\]/g,'?').replace(/\{[^}]*\}/g,'?').replace(/@\d+;/g,'?');
  const arr=raw.split(/[.,\s]+/).filter(Boolean);
  const words=arr.filter(w=>/^[a-z]+$/.test(w));
  skipped[m.L]+=arr.length-words.length;
  const x=all[m.L]; x.words+=words.length;x.folios.add(p[1]);
  x.sections[m.I]=(x.sections[m.I]||0)+words.length; x.hands[m.H]=(x.hands[m.H]||0)+words.length;
  if(m.I==='H')(data[m.L][m.Q]??=[]).push(...words);
}
function rng(seed){let x=seed|0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}
function sample(a,n,next){const b=a.slice();for(let i=0;i<n;i++){const j=i+Math.floor(next()*(b.length-i));[b[i],b[j]]=[b[j],b[i]];}return b.slice(0,n);}
function rate(test,train){const support=new Set();for(const w of train)for(let i=0;i+2<w.length;i++)support.add(w.slice(i,i+3));let bad=0;for(const w of test){let v=0;for(let i=0;i+2<w.length;i++)if(!support.has(w.slice(i,i+3))){v=1;break;}bad+=v;}return bad/test.length;}
function reva(w){return (w.match(/cth|ckh|cph|cfh|ch|sh|./g)||[]).reverse().join('');}
const output={source:'ZL3b-n.txt',scope:'P loci; certain lowercase tokens; Herbal I=H for transfer', all_scope:Object.fromEntries(['A','B'].map(k=>[k,{words:all[k].words,folios:all[k].folios.size,sections:all[k].sections,hands:all[k].hands,skipped:skipped[k]}])), n_train_per_arm:1500,n_draws:100,
  warning:'Rates are descriptive cross-repertoire diagnostics. No qualified matched-null SD. A/B confounded with hand; trained codebook pertains to JoJo Currier B, not independent Voynich tests.'};
for(const repr of ['EVA','REVA']) {
 const d={A:{},B:{}};for(const lang of ['A','B'])for(const [q,ws] of Object.entries(data[lang]))d[lang][q]=repr==='EVA'?ws:ws.map(reva);
 const ans={};for(const lang of ['A','B']){
  const other=lang==='A'?'B':'A', rows=[];
  for(const [q,test] of Object.entries(d[lang])){
   const ownTrain=Object.entries(d[lang]).filter(([k])=>k!==q).flatMap(([,x])=>x);
   const otherTrain=Object.entries(d[other]).filter(([k])=>k!==q).flatMap(([,x])=>x);
   if(ownTrain.length<1500||otherTrain.length<1500){rows.push({quire:q,status:'insufficient',ownTrain:ownTrain.length,otherTrain:otherTrain.length});continue;}
   let own=0,cross=0;
   for(let r=0;r<100;r++){const next=rng((r+1)*39917+q.charCodeAt(0)*757+(lang==='A'?199:907));own+=rate(test,sample(ownTrain,1500,next));cross+=rate(test,sample(otherTrain,1500,next));}
   rows.push({quire:q,n_test:test.length,own:own/100,cross:cross/100,delta:(cross-own)/100});
  }
  const ok=rows.filter(x=>x.delta!==undefined),t=ok.reduce((a,x)=>a+x.n_test,0);
  ans[lang]={by_quire:rows,combined:{test_tokens:t,own:ok.reduce((a,x)=>a+x.own*x.n_test,0)/t,cross:ok.reduce((a,x)=>a+x.cross*x.n_test,0)/t,delta:ok.reduce((a,x)=>a+x.delta*x.n_test,0)/t}};
 }
 output[repr]=ans;
}
console.log(JSON.stringify(output,null,2));
