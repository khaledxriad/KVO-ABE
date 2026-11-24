import math, csv, json
from model import min_beta, bits
LAM=128; ELL=64; SIGE=3.2; TT=14
B0=math.ceil(TT*SIGE)

def worst_noise(K,n,L,d,V):
    m=n*K
    sig_r=1.6*math.sqrt(m); tau=math.ceil(TT*sig_r)
    inb=m*B0
    Nf=L*(m+V)**d*inb
    T=B0+B0*m*tau+Nf*m*tau
    return T, m, tau

from functools import lru_cache
@lru_cache(None)
def n_for_security(K,target=128):
    lo,hi=64,40000
    while lo<hi:
        mid=(lo+hi)//2
        b=min_beta(mid,K,SIGE); b=3000 if b is None else b
        if bits(b)>=target: hi=mid
        else: lo=mid+1
    return lo

sets=[('P-I',8,1,16),('P-II',16,2,256),('P-III',32,3,65536)]
rows=[]
for name,L,d,V in sets:
    best=None
    for K in range(44,200,4):
        n=n_for_security(K)
        T,m,tau=worst_noise(K,n,L,d,V)
        if T< 2.0**K/4:
            # smallest K feasible; further K only increases size, but n may shrink: keep min size
            size=(ELL+2)*n*m*K/8
            if best is None or size<best[0]: best=(size,K,n,m,tau,T)
    size,K,n,m,tau,T=best
    beta=min_beta(n,K,SIGE) or 3000
    g=2**d
    mult=g-1
    row=dict(set=name,L=L,depth=d,V=V,K=K,n=n,m=m,tau=tau,beta=beta,sec_bits=round(bits(beta),1),
        log2_noise=round(math.log2(T),1),
        mpk_GB=size/1e9,
        ct_MB=((ELL+1)*m+LAM)*K/8/1e6,
        download_MB=(2*m+LAM)*K/8/1e6,
        aux_per_mulgate_GB=m*m/8/1e9, mulgates=mult,
        sk_MB=2*m*LAM*math.ceil(math.log2(tau)+1)/8/1e6,
        cloud_ops=mult*m*m, user_ops=2*m*LAM, ratio=mult*m*m/(2*m*LAM))
    rows.append(row); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()})
with open('data/params_sets.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
