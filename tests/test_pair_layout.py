"""Host-only check of the symmetric pair layout / tile scatter used by gpu_ao2mo.

Mirrors ``gpu_ao2mo._strip_task`` in numpy with a random symmetric kernel W in
place of the Coulomb operator: every strip width must reproduce the brute-force
physicist tensor exactly and write every element exactly once per tile set.
"""
from __future__ import annotations

import numpy as np

from lib_pprpa.pair_layout import PairLayout, write_tile


def _reference(phiA, phiB, W):
    # out[a, b, c, d] = <ab|cd> = (ac|bd) = sum_gh phiA_a(g) phiB_c(g) W[g,h] phiA_b(h) phiB_d(h)
    rho = np.einsum("ag,cg->acg", phiA, phiB)                     # (nA, nB, Ng)
    E = np.einsum("acg,gh,bdh->acbd", rho, W, rho)                # (ac|bd)
    return E.transpose(0, 2, 1, 3)                                # [a, b, c, d]


def _strip_loop(phiA, phiB, W, layout, blk):
    nA, nB = layout.nA, layout.nB
    pidx, qidx = layout.index_arrays()
    assert len(pidx) == layout.npair
    out = np.full((nA, nA, nB, nB), np.nan)

    def codensity(P0, P1):
        return phiA[pidx[P0:P1]] * phiB[qidx[P0:P1]]

    ntile = 0
    for pa, pb in layout.split(0, nA, blk):
        P0, P1 = layout.pairs(pa, pb)
        assert P1 - P0 <= max(blk, layout.min_blk)
        vR = codensity(P0, P1) @ W
        for ra, rb in layout.split(0, pb, blk):
            Q0, Q1 = layout.pairs(ra, rb)
            tile = vR @ codensity(Q0, Q1).T
            write_tile(out, tile, layout, pa, pb, ra, rb)
            ntile += 1
    assert ntile == layout.tiles(layout.split(0, nA, blk), blk)
    return out


def _check(nA, nB, compact, seed=0):
    rng = np.random.default_rng(seed)
    ng = 37
    phiA = rng.standard_normal((nA, ng))
    phiB = phiA if compact else rng.standard_normal((nB, ng))
    W = rng.standard_normal((ng, ng))
    W = W + W.T
    ref = _reference(phiA, phiB, W)
    layout = PairLayout(nA, nB, compact)
    for blk in sorted({1, layout.min_blk, layout.min_blk + 1, 2 * layout.min_blk + 3,
                       layout.npair // 3 + 1, layout.npair, 10 ** 9}):
        out = _strip_loop(phiA, phiB, W, layout, blk)
        assert not np.isnan(out).any(), (nA, nB, compact, blk, "unwritten elements")
        err = np.abs(out - ref).max()
        assert err < 1e-10 * max(1.0, np.abs(ref).max()), (nA, nB, compact, blk, err)


def test_compact_same_mo_set():
    for n in (1, 2, 3, 5, 8):
        _check(n, n, compact=True, seed=n)


def test_mixed_mo_sets():
    for nA, nB in ((1, 1), (1, 4), (3, 4), (5, 2), (6, 6)):
        _check(nA, nB, compact=False, seed=nA * 10 + nB)


def test_split_and_counts():
    lay = PairLayout(300, 300, compact=True)
    assert lay.npair == 300 * 301 // 2 and lay.min_blk == 300
    strips = lay.split(0, 300, 600)
    assert strips[0][0] == 0 and strips[-1][1] == 300
    assert all(b > a for a, b in strips)
    assert all(lay.pairs(a, b)[1] - lay.pairs(a, b)[0] <= 600 for a, b in strips)
    # every row appears exactly once
    assert sorted(r for a, b in strips for r in range(a, b)) == list(range(300))
    # compact + lower triangle: about 1/8 of the full-index all-tiles flop count
    full = 2.0 * (300 ** 2) ** 2
    assert 0.11 < lay.gemm_flop(1, 600) / full < 0.15
    mixed = PairLayout(128, 300, compact=False)
    assert mixed.npair == 128 * 300 and mixed.min_blk == 300
    full_m = 2.0 * (128 * 300) ** 2
    assert 0.5 < mixed.gemm_flop(1, 600) / full_m < 0.56


if __name__ == "__main__":
    test_compact_same_mo_set()
    print("OK  test_compact_same_mo_set")
    test_mixed_mo_sets()
    print("OK  test_mixed_mo_sets")
    test_split_and_counts()
    print("OK  test_split_and_counts")
