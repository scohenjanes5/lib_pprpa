"""Pair-index layout and symmetric tile scatter for the FFT ao2mo Gram kernel.

The chemist ERI of one block is the Gram matrix ``E = rho W rho^T`` over a pair
index ``P = (p, q)`` (``gpu_ao2mo``).  Two exact symmetries cut its cost:

* ``E`` is symmetric, so only tiles ``E[I, J]`` with ``J`` at or below the
  diagonal are computed; the ``(J, I)`` tile is the transpose (2x fewer GEMM
  flops for every block).
* With real orbitals and the same MO set on both sides of the pair (vvvv,
  oooo) the codensity of ``(p, q)`` equals that of ``(q, p)``, so the pair
  index runs over ``p >= q`` only (``compact``): half the rows and half the
  columns of ``E`` (another 4x, 8x in total for those blocks).

Both are pure index bookkeeping -- no approximation.  This module is
array-library agnostic (numpy or cupy for ``out`` and ``tile``), so the
scatter can be unit-tested on the host without a GPU.
"""
from __future__ import annotations

import numpy as np


class PairLayout:
    """Row-aligned strips over the pair index of one ERI block.

    Rows are the first MO index ``p``; row ``p`` holds ``p + 1`` pairs when
    ``compact`` (``q <= p``) and ``nB`` pairs otherwise.  Strips are whole
    rows, so the physicist write of a tile stays a set of slice assignments.
    """

    def __init__(self, nA, nB, compact):
        self.nA, self.nB, self.compact = int(nA), int(nB), bool(compact)
        if self.compact:
            if self.nA != self.nB:
                raise ValueError("compact pair layout needs nA == nB")
            row_len = np.arange(1, self.nA + 1, dtype=np.int64)
        else:
            row_len = np.full(self.nA, self.nB, dtype=np.int64)
        self.row_off = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int64)
        self.npair = int(self.row_off[-1])
        # a strip must hold at least one whole row
        self.min_blk = int(row_len.max())
        self.nrows = self.nA

    def pairs(self, r0, r1):
        """Pair offsets [P0, P1) covered by rows [r0, r1)."""
        return int(self.row_off[r0]), int(self.row_off[r1])

    def split(self, r0, r1, blk):
        """Whole-row sub-strips of rows [r0, r1), each of at most ``blk`` pairs
        (a single row is always emitted even if it exceeds ``blk``)."""
        out = []
        a = int(r0)
        r1 = int(r1)
        while a < r1:
            b = a + 1
            while b < r1 and self.row_off[b + 1] - self.row_off[a] <= blk:
                b += 1
            out.append((a, b))
            a = b
        return out

    def index_arrays(self):
        """``(pidx, qidx)`` int32: pair ``P`` -> its two MO indices."""
        if self.compact:
            p = np.repeat(np.arange(self.nA), np.arange(1, self.nA + 1))
            q = np.concatenate([np.arange(k + 1) for k in range(self.nA)])
        else:
            p = np.repeat(np.arange(self.nA), self.nB)
            q = np.tile(np.arange(self.nB), self.nA)
        return p.astype(np.int32), q.astype(np.int32)

    def tiles(self, strips, blk):
        """Number of (outer, inner) Gram tiles for these outer strips, with
        inner strips of at most ``blk`` pairs restricted to rows below the
        outer strip's end."""
        return sum(len(self.split(0, pb, blk)) for _pa, pb in strips)

    def gemm_flop(self, ngrid, blk):
        """fp64 flops of the tile GEMMs for outer/inner strips of ``blk`` pairs."""
        strips = self.split(0, self.nA, blk)
        flop = 0.0
        for pa, pb in strips:
            for ra, rb in self.split(0, pb, blk):
                for oa, ob, ia, ib in split_diagonal(pa, pb, ra, rb):
                    P0, P1 = self.pairs(oa, ob)
                    Q0, Q1 = self.pairs(ia, ib)
                    flop += 2.0 * (P1 - P0) * (Q1 - Q0) * ngrid
        return flop


def split_diagonal(pa, pb, ra, rb, depth=2):
    """Sub-tiles ``(oa, ob, ia, ib)`` (outer rows, inner rows) of the strip pair
    outer [pa, pb) x inner [ra, rb) that together cover its canonical ``r <= p``
    elements exactly once.

    A pair entirely below the diagonal (``rb <= pa``) is one tile.  One that
    straddles it is halved ``depth`` times in both directions and the
    sub-tiles with every ``r > p`` (``ia >= ob``) are dropped, so the GEMM
    computes only ~1/2**depth of the unused upper triangle instead of all of
    it (about 9% of the block's flops at wide strips).
    """
    pa, pb, ra, rb = int(pa), int(pb), int(ra), int(rb)
    if rb <= pa:
        return [(pa, pb, ra, rb)]
    if ra >= pb:
        return []
    if depth <= 0 or pb - pa < 2 or rb - ra < 2:
        return [(pa, pb, ra, rb)]
    pm = (pa + pb) // 2
    rm = (ra + rb) // 2
    tiles = []
    for oa, ob in ((pa, pm), (pm, pb)):
        for ia, ib in ((ra, rm), (rm, rb)):
            tiles.extend(split_diagonal(oa, ob, ia, ib, depth - 1))
    return tiles


def symmetric_half_kernel(w_flat, mesh):
    """The flat real-space-FFT kernel ``w(G)`` on the R2C half mesh,
    ``(nx, ny, nz//2 + 1)``, symmetrised as ``(w(G) + w(-G)) / 2``.

    ``ifft(fft(rho) * w).real`` (pyscf's and the C2C path's convention) only
    sees the even part of ``w``: the odd part yields a purely imaginary
    contribution that ``.real`` drops.  ``w(G) = 4 pi / |G|^2`` is even except on
    the Nyquist planes of an even mesh dimension in a non-orthogonal cell,
    where ``-G`` is not representable and the fftfreq index aliases it.  The
    C2R transform assumes a Hermitian spectrum, i.e. an even kernel, so feeding
    it the symmetrised kernel reproduces the C2C result exactly (to rounding)
    on every mesh instead of only on odd ones.  Works for numpy and cupy.
    """
    nx, ny, nz = (int(m) for m in mesh)
    w = w_flat.reshape(nx, ny, nz)
    ix, iy, iz = ((-np.arange(n)) % n for n in (nx, ny, nz))     # k -> -k mod n
    w_mirror = w[ix][:, iy][:, :, iz]
    ws = 0.5 * (w + w_mirror)
    return ws[:, :, :nz // 2 + 1].copy()


def write_tile(out, tile, layout, pa, pb, ra, rb):
    """Scatter the Gram tile ``E[I, J]`` into the physicist tensor ``out``.

    ``I`` = outer rows [pa, pb), ``J`` = inner rows [ra, rb); ``tile`` is the
    chemist block ``(pq|rs)`` over the pairs of those rows.  Every element the
    tile fixes by symmetry is written, but only for canonical row pairs
    ``r <= p``, so two tiles never write the same element and redoing a strip
    after an OOM is idempotent.  ``out`` and ``tile`` must live in the same
    array library (both numpy or both cupy).

    Physicist convention: ``out[a, b, c, d] = <ab|cd> = (ac|bd)``.
    """
    ro = layout.row_off
    if not layout.compact:
        nB = layout.nB
        phys = tile.reshape(pb - pa, nB, rb - ra, nB).transpose(0, 2, 1, 3)
        if rb <= pa:                       # tile entirely below the diagonal
            out[pa:pb, ra:rb] = phys
            out[ra:rb, pa:pb] = phys.transpose(1, 0, 3, 2)
            return
        for p in range(pa, pb):            # straddles the diagonal: keep r <= p
            r1 = min(rb, p + 1)
            if r1 <= ra:
                continue
            blk = phys[p - pa, :r1 - ra]   # (r1-ra, nB, nB) = out[p, ra:r1]
            out[p, ra:r1] = blk
            out[ra:r1, p] = blk.transpose(0, 2, 1)
        return

    # compact: pairs (p, q<=p) and (r, s<=r); T[q, s] = (pq|rs) is one
    # (p+1) x (r+1) sub-block of the tile, and its 8 permutation images are
    #   (pq|rs) (qp|rs) (pq|sr) (qp|sr) (rs|pq) (sr|pq) (rs|qp) (sr|qp)
    # each landing on out[x, z, y, w] for (xy|zw).
    for p in range(pa, pb):
        rows = slice(int(ro[p] - ro[pa]), int(ro[p + 1] - ro[pa]))
        for r in range(ra, min(rb, p + 1)):
            T = tile[rows, int(ro[r] - ro[ra]):int(ro[r + 1] - ro[ra])]
            TT = T.T
            out[p, r, :p + 1, :r + 1] = T        # (pq|rs)
            out[:p + 1, r, p, :r + 1] = T        # (qp|rs)
            out[:p + 1, :r + 1, p, r] = T        # (qp|sr)
            out[r, :p + 1, :r + 1, p] = T        # (rs|qp)
            out[p, :r + 1, :p + 1, r] = TT       # (pq|sr)
            out[r, p, :r + 1, :p + 1] = TT       # (rs|pq)
            out[:r + 1, p, r, :p + 1] = TT       # (sr|pq)
            out[:r + 1, :p + 1, r, p] = TT       # (sr|qp)
