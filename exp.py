import numpy as np, time, csv, sys, statistics as st
from kvo import *

def med(f, reps):
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); r=f(); ts.append(time.perf_counter()-t)
    return st.median(ts), r

def sat_x(rng, g, w):
    L=g*w
    x=rng.integers(0,2,L)
    # make at least one group all zero (satisfying); groups are consecutive blocks of w
    gi=rng.integers(0,g)
    x[gi*w:(gi+1)*w]=0
    return x

def center(v):
    v=modq(v); return np.where(v>=(Q>>1), v-Q, v)

# ------------------------------------------------------------- E1 correctness
def e1():
    rows=[]
    for n,g,w,trials in [(16,1,8,60),(16,2,4,60),(16,4,4,40),(32,2,4,30),(32,4,4,20)]:
        rng=np.random.default_rng(100+n+g)
        L=g*w; P=Params(n,L); A,Bs=setup(P,rng); f=make_policy(g,w)
        ax=eval_pk(P,f,Bs); R,U=keygen_shortcut(P,A,ax,rng)
        ok=0; worst=0.0; nb=0
        for t in range(trials):
            x=sat_x(rng,g,w); assert eval_x(f,x)==0
            msg=rng.bytes(48)
            ct,cs=full_encrypt(P,A,Bs,U,x,msg,rng)
            psi=eval_ct(P,f,ax,x,cs)
            out=finalize(R,ct,psi)
            ok+= (out==msg); nb=max(nb,noise_bound(P,f,x))
            # actual noise: recompute from seed
            rho=recover_seed(R,ct['c0'],psi,ct['cU'])
            v=center(ct['cU']-partial_inner(R,ct['c0'],psi)-rho.astype(np.int64)*(Q>>1))
            worst=max(worst,float(np.abs(v).max())/(Q/4))
        wc=(P.B0+P.B0*P.m*P.Rb+nb*P.m*P.Rb)/(Q/4)
        rows.append(dict(n=n,m=P.m,g=g,w=w,L=L,depth=mdepth(f),mulgates=g-1,trials=trials,success=ok,
                         max_noise_over_qdiv4=worst,worstcase_bound_over_qdiv4=wc))
        print(rows[-1]); sys.stdout.flush()
    with open('data/e1_correctness.csv','w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

# ------------------------------------------------------------- E2 timing
def e2_once():
    rows=[]
    for n in (16,32,64):
        for g in (1,2,4,8):
            w=4; L=g*w
            if n==64 and g==8: continue
            rng=np.random.default_rng(7+n*10+g)
            P=Params(n,L); A,Bs=setup(P,rng); f=make_policy(g,w)
            t_pk,ax=med(lambda: eval_pk(P,f,Bs),1)
            R,U=keygen_shortcut(P,A,ax,rng)
            x=sat_x(rng,g,w); msg=rng.bytes(1024)
            t_enc,(ct,cs)=med(lambda: full_encrypt(P,A,Bs,U,x,msg,rng),1)
            eval_ct(P,f,ax,x,cs); t_cloud,psi=med(lambda: eval_ct(P,f,ax,x,cs),9)
            finalize(R,ct,psi); t_user,out=med(lambda: finalize(R,ct,psi),201)
            assert out==msg
            t_local=t_cloud+t_user
            aux=sum(a.size for a in ax.gm)/8.0      # bits -> bytes (1 bit per entry)
            ctb=((L+1)*P.m+LAM)*K/8+len(ct['csym'])+TAGB
            dlb=(2*P.m+LAM)*K/8+len(ct['csym'])+TAGB
            rows.append(dict(n=n,m=P.m,g=g,L=L,mulgates=g-1,t_evalpk_s=t_pk,t_enc_s=t_enc,t_cloud_s=t_cloud,
                             t_user_out_s=t_user,t_user_local_s=t_local,speedup=t_local/t_user,
                             aux_bytes=aux,ct_bytes=ctb,user_download_bytes=dlb))
            print(rows[-1]); sys.stdout.flush()
            del ax,Bs
    return rows

def e2(reps=5):
    """Run the whole timing experiment `reps` times and report per-cell medians.
    Raw runs go to data/e2_timing_runs.csv, medians to data/e2_timing.csv."""
    allruns=[]
    for r in range(reps):
        for row in e2_once():
            row=dict(row); row['run']=r+1; allruns.append(row)
    with open('data/e2_timing_runs.csv','w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=allruns[0].keys()); wr.writeheader(); wr.writerows(allruns)
    keys=[(a['n'],a['g']) for a in allruns if a['run']==1]
    rows=[]
    for (n,g) in keys:
        cell=[a for a in allruns if a['n']==n and a['g']==g]
        agg=dict(n=n,m=cell[0]['m'],g=g,L=cell[0]['L'],mulgates=cell[0]['mulgates'])
        for k in ('t_evalpk_s','t_enc_s','t_cloud_s','t_user_out_s','t_user_local_s','speedup'):
            agg[k]=st.median([c[k] for c in cell])
        for k in ('aux_bytes','ct_bytes','user_download_bytes'): agg[k]=cell[0][k]
        sp=[c['speedup'] for c in cell]
        agg['speedup_min']=min(sp); agg['speedup_max']=max(sp); agg['runs']=len(cell)
        rows.append(agg)
    with open('data/e2_timing.csv','w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

# ------------------------------------------------------------- E3 cheating cloud
def e3():
    n,g,w=16,2,4; L=g*w
    rng=np.random.default_rng(2026)
    P=Params(n,L); A,Bs=setup(P,rng); f=make_policy(g,w)
    ax=eval_pk(P,f,Bs); R,U=keygen_shortcut(P,A,ax,rng)
    NCT=20; PER=500
    cts=[]
    for _ in range(NCT):
        x=sat_x(rng,g,w); msg=rng.bytes(64)
        ct,cs=full_encrypt(P,A,Bs,U,x,msg,rng)
        psi=eval_ct(P,f,ax,x,cs); cts.append((ct,msg,psi))
    kinds=['honest','random_vector','all_zero','replay_other_ct','one_coord_random','sparse_noise_2^40','linear_shift_forgery']
    res={k:[0,0,0] for k in kinds}   # accepted_correct, accepted_wrong, rejected
    def judge(k,out,msg):
        if out is None: res[k][2]+=1
        elif out==msg: res[k][0]+=1
        else: res[k][1]+=1
    z=np.zeros(n,dtype=np.int64); z[0]=Q>>1
    for ci,(ct,msg,psi) in enumerate(cts):
        for _ in range(PER):
            judge('honest',finalize(R,ct,psi),msg)
            judge('random_vector',finalize(R,ct,rng.integers(0,Q,P.m,dtype=np.int64)),msg)
            judge('all_zero',finalize(R,ct,np.zeros(P.m,dtype=np.int64)),msg)
            o=cts[(ci+1+rng.integers(0,NCT-1))%NCT][2]
            judge('replay_other_ct',finalize(R,ct,o),msg)
            p=psi.copy(); p[rng.integers(0,P.m)]=rng.integers(0,Q); judge('one_coord_random',finalize(R,ct,p),msg)
            p=modq(psi+rng.integers(-(1<<40),(1<<40),P.m,dtype=np.int64)); judge('sparse_noise_2^40',finalize(R,ct,p),msg)
        # linear shift forgery: adversary serves tampered c0 as well
        ct2=dict(ct); ct2['c0']=modq(ct['c0']+z@A); 
        for _ in range(PER):
            judge('linear_shift_forgery',finalize(R,ct2,modq(psi+z@ax.Bf)),msg)
    rows=[dict(attack=k,trials=sum(v),accepted_correct=v[0],accepted_wrong=v[1],rejected=v[2]) for k,v in res.items()]
    for r in rows: print(r)
    with open('data/e3_cheating.csv','w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

    # tolerance curve: uniform perturbation of magnitude 2^e on every coordinate
    ct,msg,psi=cts[0]
    rows=[]
    for e in range(40,62):
        ok=wr_=rej=0
        for _ in range(300):
            p=modq(psi+rng.integers(-(1<<e),(1<<e)+1,P.m,dtype=np.int64))
            out=finalize(R,ct,p)
            if out is None: rej+=1
            elif out==msg: ok+=1
            else: wr_+=1
        rows.append(dict(log2_magnitude=e,trials=300,accepted_correct=ok,accepted_wrong=wr_,rejected=rej))
        print(rows[-1])
    with open('data/e3_tolerance.csv','w',newline='') as fh:
        w2=csv.DictWriter(fh,fieldnames=rows[0].keys()); w2.writeheader(); w2.writerows(rows)

if __name__=='__main__':
    for a in sys.argv[1:]:
        {'e1':e1,'e2':e2,'e3':e3}[a]()
