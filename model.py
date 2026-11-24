import math, json, csv

def delta(beta):
    return ((math.pi*beta)**(1.0/beta)*beta/(2*math.pi*math.e))**(1.0/(2*(beta-1)))

def log2delta(b): return math.log2(delta(b))

def min_beta(n, K, sigma):
    """primal uSVP / Kannan embedding, core-SVP; best over number of samples m'"""
    best = None
    for mp in sorted(set(int(n*f) for f in [1.0,1.1,1.2,1.3,1.5,1.7,2,2.5,3,4,5,6])):
        d = mp + 1
        for beta in range(60, 3000):
            lhs = math.log2(sigma) + 0.5*math.log2(beta)
            rhs = (2*beta - d - 1)*log2delta(beta) + (mp - n)*K/d
            if lhs <= rhs:
                if best is None or beta < best: best = beta
                break
    return best

def bits(beta): return 0.292*beta

if __name__ == '__main__':
    # calibration vs published tables (uniform secret, sigma=3.2)
    for n,K in [(1024,27),(2048,54),(2048,47),(2300,62)]:
        b=min_beta(n,K,3.2); print(n,K,b,round(bits(b),1))
