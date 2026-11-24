"""
KVO-ABE prototype: keyless verifiable outsourced decryption for a
lattice-based key-policy ABE (BGG+-style, power-of-two modulus).

All arithmetic is done modulo 2^64 with int64 wrap-around and then masked
to q = 2^K, which is exact because q divides 2^64.

NOTE (prototype shortcuts, stated in the paper):
  * key generation picks the short matrix R first and defines U = [A|B_f] R
    (no trapdoor sampler); this is only used to test Enc / Eval / Dec.
  * discrete Gaussians are produced by rounding continuous Gaussians and
    resampling values outside the truncation bound.
"""
import hashlib
import numpy as np

K = 62                    # log2 q
Q = 1 << K
MASK = np.int64(Q - 1)
LAM = 128                 # seed length in bits
TAGB = 32                 # tag length in bytes


def modq(a):
    return a & MASK


def tgauss(rng, sigma, size, tail=6.0):
    """truncated rounded Gaussian, |x| <= ceil(tail*sigma)"""
    bound = int(np.ceil(tail * sigma))
    x = np.rint(rng.normal(0, sigma, size)).astype(np.int64)
    bad = np.abs(x) > bound
    while bad.any():
        x[bad] = np.rint(rng.normal(0, sigma, int(bad.sum()))).astype(np.int64)
        bad = np.abs(x) > bound
    return x


def ginv(B, n, m):
    """bit decomposition: returns (n*K) x m 0/1 matrix with G @ ginv(B) = B"""
    B = modq(B)
    out = np.empty((n, K, m), dtype=np.int64)
    for t in range(K):
        out[:, t, :] = (B >> t) & 1
    return out.reshape(n * K, m)


def gmat(n):
    g = (np.int64(1) << np.arange(K, dtype=np.int64))
    G = np.zeros((n, n * K), dtype=np.int64)
    for a in range(n):
        G[a, a * K:(a + 1) * K] = g
    return G


# ---------------------------------------------------------------- formulas
# a formula is a nested tuple: ('leaf', i) | ('add', f1, f2) | ('mul', f1, f2)

def leaves(f):
    if f[0] == 'leaf':
        return 1
    return leaves(f[1]) + leaves(f[2])


def mdepth(f):
    if f[0] == 'leaf':
        return 0
    a, b = mdepth(f[1]), mdepth(f[2])
    return max(a, b) + (1 if f[0] == 'mul' else 0)


def make_policy(g, w, start=0):
    """product of g groups, each the sum of w leaves (AND inside, OR across)"""
    idx = start
    groups = []
    for _ in range(g):
        node = ('leaf', idx); idx += 1
        for _ in range(w - 1):
            node = ('add', node, ('leaf', idx)); idx += 1
        groups.append(node)
    while len(groups) > 1:
        nxt = []
        for i in range(0, len(groups) - 1, 2):
            nxt.append(('mul', groups[i], groups[i + 1]))
        if len(groups) % 2:
            nxt.append(groups[-1])
        groups = nxt
    return groups[0]


def eval_x(f, x):
    if f[0] == 'leaf':
        return int(x[f[1]])
    a, b = eval_x(f[1], x), eval_x(f[2], x)
    return a + b if f[0] == 'add' else a * b


# ---------------------------------------------------------------- scheme
class Params:
    def __init__(self, n, ell, sig_e=3.2, sig_r=None):
        self.n, self.ell = n, ell
        self.m = n * K
        self.sig_e = sig_e
        self.B0 = int(np.ceil(6 * sig_e))
        self.sig_r = sig_r if sig_r else 1.6 * np.sqrt(self.m)
        self.Rb = int(np.ceil(6 * self.sig_r))


def setup(P, rng):
    n, m = P.n, P.m
    A = rng.integers(0, Q, (n, m), dtype=np.int64)
    Bs = [rng.integers(0, Q, (n, m), dtype=np.int64) for _ in range(P.ell)]
    return A, Bs


class Aux:
    """public per-policy data: B_f and the G^{-1}(.) matrices of mul gates"""
    pass


def eval_pk(P, f, Bs):
    """returns (B_f, list of ginv matrices in post-order of mul gates)"""
    n, m = P.n, P.m
    gm = []

    def rec(node):
        if node[0] == 'leaf':
            return Bs[node[1]]
        Bi = rec(node[1])
        Bj = rec(node[2])
        if node[0] == 'add':
            return modq(Bi + Bj)
        Gi = ginv(Bj, n, m)
        gm.append(Gi)
        node_aux[id(node)] = len(gm) - 1
        return modq(-(Bi @ Gi))

    node_aux = {}
    Bf = rec(f)
    ax = Aux()
    ax.Bf, ax.gm, ax.map = Bf, gm, node_aux
    return ax


def keygen_shortcut(P, A, ax, rng):
    n, m = P.n, P.m
    R = tgauss(rng, P.sig_r, (2 * m, LAM))
    AB = np.concatenate([A, ax.Bf], axis=1)          # n x 2m
    U = modq(AB @ R)
    return R, U


def enc_seed(P, A, Bs, U, x, rho_bits, rng):
    """BGG+ style encryption of a lambda-bit seed under attribute vector x."""
    n, m = P.n, P.m
    G = gmat(n)
    s = rng.integers(0, Q, n, dtype=np.int64)
    e0 = tgauss(rng, P.sig_e, m)
    c0 = modq(s @ A + e0)
    cs = []
    for i in range(P.ell):
        Ri = rng.integers(0, 2, (m, m), dtype=np.int8) * 2 - 1
        ei = e0 @ Ri.astype(np.int64)
        Bi = modq(Bs[i] + int(x[i]) * G)
        cs.append(modq(s @ Bi + ei))
    eU = tgauss(rng, P.sig_e, LAM)
    cU = modq(s @ U + eU + rho_bits.astype(np.int64) * (Q >> 1))
    return c0, cs, cU


def eval_ct(P, f, ax, x, cs):
    """the heavy, key-free part (run by the cloud)."""
    def rec(node):
        if node[0] == 'leaf':
            return cs[node[1]], int(x[node[1]])
        pi, xi = rec(node[1])
        pj, xj = rec(node[2])
        if node[0] == 'add':
            return modq(pi + pj), xi + xj
        Gi = ax.gm[ax.map[id(node)]]
        return modq(xi * pj - pi @ Gi), xi * xj
    return rec(f)[0]


def round_bits(v):
    v = modq(v)
    return ((v >= (Q >> 2)) & (v < 3 * (Q >> 2))).astype(np.uint8)


def partial_inner(R, c0, psi):
    """user-side work: lambda inner products of length 2m"""
    cc = np.concatenate([c0, psi])
    return modq(cc @ R)


def recover_seed(R, c0, psi, cU):
    return round_bits(cU - partial_inner(R, c0, psi))


# ------------------------------------------------------------- hybrid layer
def H(tag, *parts):
    h = hashlib.sha3_256()
    h.update(tag)
    for p in parts:
        h.update(len(p).to_bytes(8, 'big'))
        h.update(p)
    return h.digest()


def kdf_pad(rho_bytes, ln):
    return hashlib.shake_256(b'KVO-pad' + rho_bytes).digest(ln)


def xbytes(x):
    return bytes(int(v) for v in x)


def make_tag(rho_bytes, x, csym):
    return H(b'KVO-tag', rho_bytes, xbytes(x), csym)


def finalize(R, ct, psi):
    """verification + message recovery. ct = dict(c0,cU,x,csym,tag)"""
    rho = recover_seed(R, ct['c0'], psi, ct['cU'])
    rb = np.packbits(rho).tobytes()
    if make_tag(rb, ct['x'], ct['csym']) != ct['tag']:
        return None
    pad = kdf_pad(rb, len(ct['csym']))
    return bytes(a ^ b for a, b in zip(ct['csym'], pad))


def full_encrypt(P, A, Bs, U, x, msg, rng):
    rho = rng.integers(0, 2, LAM).astype(np.uint8)
    rb = np.packbits(rho).tobytes()
    c0, cs, cU = enc_seed(P, A, Bs, U, x, rho, rng)
    pad = kdf_pad(rb, len(msg))
    csym = bytes(a ^ b for a, b in zip(msg, pad))
    tag = make_tag(rb, x, csym)
    return dict(c0=c0, cU=cU, x=np.array(x, dtype=np.int64), csym=csym, tag=tag), cs


# ---------------------------------------------------------- noise accounting
def noise_bound(P, f, x):
    """worst-case infinity-norm bound of e_f (Lemma 2 recursion with actual wire values)"""
    inb = P.m * P.B0            # |e_i| <= m*B0 (e_i = R_i^T e_0)

    def rec(node):
        if node[0] == 'leaf':
            return inb, int(x[node[1]])
        ni, xi = rec(node[1])
        nj, xj = rec(node[2])
        if node[0] == 'add':
            return ni + nj, xi + xj
        return abs(xi) * nj + P.m * ni, xi * xj
    return rec(f)[0]
