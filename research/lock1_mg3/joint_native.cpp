// Minimal native library for MG1: identical distance_raw/dist_many to GPT's joint_native.cpp; other symbols are stubs.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
using namespace std;
extern "C" {
int distance_raw(const char* aa,const char* bb){
 string a(aa),b(bb);vector<int> r(b.size()+1),s(b.size()+1);
 for(int j=0;j<=int(b.size());j++)r[j]=j;
 for(int i=0;i<int(a.size());i++){s[0]=i+1;for(int j=0;j<int(b.size());j++)s[j+1]=min({s[j]+1,r[j+1]+1,r[j]+(a[i]!=b[j])});r.swap(s);}return r[b.size()];
}
void dist_many(const char* a,const char** bs,int n,uint8_t* out){for(int i=0;i<n;i++)out[i]=min(4,distance_raw(a,bs[i]));}
void set_vocab(const char** w,int n){}
void set_atom_trie(const int* w,const int* l,int n){}
long long close_masses(const char* s,int A,int S,int i,int m,const char** a,const double* P,const int* ns,double* o,long long x){return -1;}
void shell_sample(const char* s,int A,int i,int c,const char** a,const double* P,const int* ns,unsigned long long seed,int n,double* o){}
int draw_conditioned(const char* s,int b,int A,int i,int c,const char** a,const double* P,const int* ns,const double* f,unsigned long long seed,int bud,char* o){return -1;}
long long draw_close_exact(const char* s,int b,int A,int i,int c,const char** a,const double* P,const int* ns,const double* f,unsigned long long seed,char* o,double* z){return -1;}
}
