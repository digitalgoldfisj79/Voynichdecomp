#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
using Map=std::array<int,27>;
struct Events {
 int n;const int *ids;const double *weights;const double *lp;
 std::vector<int> affected[26][26];
 Events(int n_,const int*i,const double*w,const double*l):n(n_),ids(i),weights(w),lp(l){
  for(int a=0;a<26;a++)for(int b=a+1;b<26;b++)for(int j=0;j<n;j++){
   bool hit=false;for(int k=0;k<3;k++)if(ids[3*j+k]==a||ids[3*j+k]==b)hit=true;
   if(hit)affected[a][b].push_back(j);
  }
 }
 double event(int j,const Map&m)const{return lp[(m[ids[3*j]]*27+m[ids[3*j+1]])*27+m[ids[3*j+2]]];}
 double score(const Map&m)const{double s=0;for(int j=0;j<n;j++)s+=weights[j]*event(j,m);return s;}
 double delta(const Map&m,int a,int b)const{
  if(a>b)std::swap(a,b);Map mm=m;std::swap(mm[a],mm[b]);double s=0;
  for(int j:affected[a][b])s+=weights[j]*(event(j,mm)-event(j,m));return s;
 }
};
extern "C" double trigram_solve(int n,const int*ids,const double*weights,const double*lp,const int*initial,uint64_t seed,int restarts,int steps,int greedy,int*out){
 Events ev(n,ids,weights,lp);Map best,init;for(int i=0;i<26;i++)init[i]=initial[i];init[26]=26;best=init;
 double bs=ev.score(best);std::mt19937_64 rng(seed);std::uniform_int_distribution<int>pick(0,25);std::uniform_real_distribution<double>unit(0,1);
 for(int r=0;r<restarts;r++){
  Map m=init;for(int k=0;k<8*r;k++){int a=pick(rng),b=pick(rng);std::swap(m[a],m[b]);}double s=ev.score(m);
  if(s>bs){bs=s;best=m;}
  for(int t=0;t<steps;t++){
   int a=pick(rng),b=pick(rng);while(b==a)b=pick(rng);
   double d=ev.delta(m,a,b),temp=.006*(1.-double(t)/std::max(1,steps-1))+.00003;
   if(d>=0||unit(rng)<std::exp(std::max(-50.,d/temp))){std::swap(m[a],m[b]);s+=d;if(s>bs){bs=s;best=m;}}
  }
  m=best;s=bs;
  for(int t=0;t<greedy;t++){
   double bd=0;int ba=-1,bb=-1;
   for(int a=0;a<26;a++)for(int b=a+1;b<26;b++){double d=ev.delta(m,a,b);if(d>bd+1e-12){bd=d;ba=a;bb=b;}}
   if(ba<0)break;std::swap(m[ba],m[bb]);s+=bd;if(s>bs){bs=s;best=m;}
  }
 }
 for(int i=0;i<26;i++)out[i]=best[i];return ev.score(best);
}
extern "C" double trigram_delta(int n,const int*ids,const double*weights,const double*lp,const int*map,int a,int b){
 Events ev(n,ids,weights,lp);Map m;for(int i=0;i<26;i++)m[i]=map[i];m[26]=26;return ev.delta(m,a,b);
}
