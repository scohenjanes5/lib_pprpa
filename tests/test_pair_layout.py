"""Host-only check of the symmetric pair layout / tile scatter used by gpu_ao2mo.

Mirrors ``gpu_ao2mo._strip_task`` in numpy with a random symmetric kernel W in
place of the Coulomb operator: every strip width must reproduce the brute-force
physicist tensor exactly and write every element exactly once per tile set.
"""
from __future__ import annotations

import numpy as np

from lib_pprpa.pair_layout import PairLayout, split_diagonal, write_tile


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

    for pa, pb in layout.split(0, nA, blk):
        P0, P1 = layout.pairs(pa, pb)
        assert P1 - P0 <= max(blk, layout.min_blk)
        vR = codensity(P0, P1) @ W
        for ra, rb in layout.split(0, pb, blk):
            for oa, ob, ia, ib in split_diagonal(pa, pb, ra, rb):
                Q0, Q1 = layout.pairs(ia, ib)
                R0, R1 = layout.pairs(oa, ob)
                tile = vR[R0 - P0:R1 - P0] @ codensity(Q0, Q1).T
                write_tile(out, tile, layout, oa, ob, ia, ib)
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


def test_split_diagonal_partitions_the_canonical_cells():
    rng = np.random.default_rng(7)
    for _ in range(200):
        pa = int(rng.integers(0, 20)); pb = pa + int(rng.integers(1, 12))
        ra = int(rng.integers(0, 20)); rb = ra + int(rng.integers(1, 12))
        want = {(p, r) for p in range(pa, pb) for r in range(ra, rb) if r <= p}
        got = []
        for oa, ob, ia, ib in split_diagonal(pa, pb, ra, rb, depth=int(rng.integers(0, 4))):
            assert pa <= oa < ob <= pb and ra <= ia < ib <= rb
            got.extend((p, r) for p in range(oa, ob) for r in range(ia, ib) if r <= p)
        assert len(got) == len(set(got)) == len(want) and set(got) == want, (pa, pb, ra, rb)
    # below the diagonal: untouched; straddling: the unused triangle shrinks
    assert split_diagonal(10, 20, 0, 10) == [(10, 20, 0, 10)]
    computed = sum((ob - oa) * (ib - ia) for oa, ob, ia, ib in split_diagonal(0, 10, 0, 10, depth=2))
    assert computed == 63            # 55 canonical cells; depth 2 computes 8 of the 45 unused


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


def test_symmetric_half_kernel_matches_c2c_real():
    """R2C with the symmetrised half kernel == C2C followed by .real, even when
    the kernel is not even in G (Nyquist planes of an even mesh)."""
    from lib_pprpa.pair_layout import symmetric_half_kernel
    rng = np.random.default_rng(3)
    for mesh in ((6, 8, 10), (7, 9, 11), (8, 8, 8), (5, 6, 7)):
        w = rng.random(int(np.prod(mesh)))                    # generic, not even in G
        rho = rng.standard_normal((3, *mesh))
        c2c = np.fft.ifftn(np.fft.fftn(rho, axes=(1, 2, 3)) * w.reshape(mesh), axes=(1, 2, 3)).real
        half = symmetric_half_kernel(w, mesh)
        assert half.shape == (mesh[0], mesh[1], mesh[2] // 2 + 1)
        r2c = np.fft.irfftn(np.fft.rfftn(rho, axes=(1, 2, 3)) * half, s=mesh, axes=(1, 2, 3))
        assert np.abs(r2c - c2c).max() < 1e-13, (mesh, np.abs(r2c - c2c).max())
        # the naive (unsymmetrised) half kernel differs on even meshes
        naive = w.reshape(mesh)[:, :, :mesh[2] // 2 + 1]
        r2c_naive = np.fft.irfftn(np.fft.rfftn(rho, axes=(1, 2, 3)) * naive, s=mesh, axes=(1, 2, 3))
        assert np.abs(r2c_naive - c2c).max() > 1e-6


if __name__ == "__main__":
    test_split_diagonal_partitions_the_canonical_cells()
    print("OK  test_split_diagonal_partitions_the_canonical_cells")
    test_compact_same_mo_set()
    print("OK  test_compact_same_mo_set")
    test_mixed_mo_sets()
    print("OK  test_mixed_mo_sets")
    test_split_and_counts()
    print("OK  test_split_and_counts")
    test_symmetric_half_kernel_matches_c2c_real()
    print("OK  test_symmetric_half_kernel_matches_c2c_real")
