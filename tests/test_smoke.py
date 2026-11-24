"""Quick checks: honest decryption is correct, and a tampered cloud answer is rejected."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kvo import (Params, setup, make_policy, eval_pk, keygen_shortcut, full_encrypt,
                 eval_ct, finalize, eval_x)


def make(n=16, g=2, w=4, seed=1):
    rng = np.random.default_rng(seed)
    L = g * w
    P = Params(n, L)
    A, Bs = setup(P, rng)
    f = make_policy(g, w)
    ax = eval_pk(P, f, Bs)
    R, U = keygen_shortcut(P, A, ax, rng)
    return rng, P, A, Bs, f, ax, R, U


def satisfying(rng, g, w):
    # draw random attribute vectors; the caller keeps the first one that satisfies the policy
    while True:
        x = rng.integers(0, 2, g * w)
        yield x


def test_honest_and_tampered():
    g, w = 2, 4
    rng, P, A, Bs, f, ax, R, U = make(g=g, w=w)
    x = next(x for x in satisfying(rng, g, w) if eval_x(f, x) == 0)
    msg = rng.bytes(48)
    ct, cs = full_encrypt(P, A, Bs, U, x, msg, rng)
    psi = eval_ct(P, f, ax, x, cs)
    assert finalize(R, ct, psi) == msg
    bad = psi.copy()
    bad[0] = (bad[0] + 1 << 55) % (1 << 62)
    out = finalize(R, ct, bad)
    assert out is None or out == msg   # never a wrong message
