"""Checks on npcompl33t/eo777 thread-6299 crease claim, using ONLY the golf row order
transcribed by eye from his chart 'Bottom creases / warping - MS vs Golf' (rows 1-22, 30-37 shown).
Question: does Golf order cluster quire-CENTRE bifolia, independently of any crease?"""
import itertools, math, numpy as np
rows = {1:"1-8",2:"27-30",3:"42-47",4:"49-56",5:"2-7",6:"25-32",7:"20-21",8:"35-38",9:"4-5",
10:"28-29",11:"10-15",12:"36-37",13:"9-16",14:"19-22",15:"11-14",16:"12-13",17:"18-23",18:"44-45",
19:"52-53",20:"3-6",21:"17-24",22:"51-54",30:"58-65",31:"26-31",32:"57-66",33:"41-48",34:"43-46",
35:"34-39",36:"33-40",37:"50-55"}
# quire-centre bifolia of Q1-Q7 (8-leaf quires f1-8, f9-16, ..., f49-56)
centre = {"4-5","12-13","20-21","28-29","36-37","44-45","52-53"}
order = sorted(rows)                      # 30 shown rows, ranks 0..29
rank = {rows[r]:i for i,r in enumerate(order)}
c_ranks = sorted(rank[b] for b in centre)
N, k = len(order), len(centre)
span = c_ranks[-1]-c_ranks[0]+1
print("centre bifolia ranks:", c_ranks, "| span", span, "of", N)
# exact P(k random ranks fit in a window <= span)
tot = math.comb(N,k)
fav = sum(math.comb(w-2,k-2)*(N-w+1) for w in range(k,span+1))
print(f"P(all {k} centre bifolia within span<={span} | random order) = {fav/tot:.5f}")
# damage set: which listed leaves are on a centre bifolium?
listed = {"f5":"4-5","f9":"9-16","f15":"10-15","f21":"20-21","f29":"28-29","f30":"27-30",
          "f32":"25-32","f37":"36-37","f52":"52-53"}
hits = [f for f,b in listed.items() if b in centre]
p = 2/8   # 2 of 8 leaves per quire sit on the centre bifolium
x, n = len(hits), len(listed)
pb = sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(x,n+1))
print(f"damaged leaves on centre bifolium: {hits} = {x}/{n}; P(>= {x} | p=0.25) = {pb:.4f}")
