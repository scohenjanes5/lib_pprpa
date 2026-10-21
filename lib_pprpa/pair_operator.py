"""The ERI part of the pp-RPA matrix as the pair-space operator of one multiplicity (NumPy).

The trial vectors of ppRPA_Davidson live on the pairs p >= q (singlet) or p > q
(triplet) of the occupied and of the virtual orbitals, with 1/sqrt(2) on p = q.  In
that basis the ERI part of the product is one symmetric matrix,

    M[(pq), (rs)] = f_pq f_rs (<pq|rs> + s <pq|sr>),  <pq|rs> = (pr|qs),
    f_pp = 1/sqrt(2), s = +1 (singlet) / -1 (triplet),

with blocks M_oo, M_vv and B = M[oo, vv], kept as row strips of the lower triangles of
M_oo and M_vv (the diagonal block of a strip in full and halved) and of B.  It is built
from the packed Gram blocks of ``pack_eri`` / ``gpu_ao2mo.gpu_ao2mo_blocks``.
"""

import numpy as np

__all__ = ['STRIP_ROWS', 'PairOperator', 'apply_pair_operator', 'operator_strips',
           'pack_eri', 'pack_offset', 'pair_offset', 'pair_operator', 'pair_operator_bytes',
           'row_strips']

# Pair rows per strip.  The diagonal block of a strip is stored in full, which adds
# rows / npair to the storage of a block; the GEMMs are bandwidth-bound at any width.
STRIP_ROWS = 1024


def pack_offset(n):
    '''Offset of row ``n`` of a packed lower triangle, n (n + 1) / 2: the number
    of pairs p >= q of n orbitals, and the length of an n-row triangle.'''
    n = int(n)
    return n * (n + 1) // 2


def pair_offset(p, multi):
    """Index of the first pair (p, .) in the trial-vector order (np.tril_indices)."""
    return pack_offset(p) if multi == "s" else pack_offset(p - 1)


def pair_operator_bytes(nocc, nvir, multi):
    """Bytes of M (lower triangles of M_oo, M_vv and all of B), before the diagonal blocks."""
    noo, nvv = pair_offset(nocc, multi), pair_offset(nvir, multi)
    return 8 * (pack_offset(noo) + pack_offset(nvv) + noo * nvv)


class PairOperator:
    """The pair-space operator of one multiplicity as row strips [(kind, R0, R1, S)],
    kind 'oo', 'vv' or 'ov' (B); see the module docstring."""

    def __init__(self, nocc, nvir, multi, strips):
        self.nocc, self.nvir, self.multi, self.strips = int(nocc), int(nvir), multi, strips

    @property
    def nbytes(self):
        return sum(strip[3].nbytes for strip in self.strips)


def row_strips(n, multi, rows, pa=0, pb=None):
    """[(pa, pb, R0, R1)]: whole first orbitals within [pa, pb) of an n-orbital pair
    space and their pair rows, at most ``rows`` orbitals or STRIP_ROWS pairs each (at
    least one orbital); empty ranges (orbital 0 of a triplet) are skipped."""
    pb = n if pb is None else pb
    out = []
    while pa < pb:
        R0 = pair_offset(pa, multi)
        if rows is not None:
            pc = min(pb, pa + max(1, int(rows)))
        else:
            pc = pa + 1
            while pc < pb and pair_offset(pc + 1, multi) - R0 <= STRIP_ROWS:
                pc += 1
        R1 = pair_offset(pc, multi)
        if R1 > R0:
            out.append((pa, pc, R0, R1))
        pa = pc
    return out


def operator_strips(nocc, nvir, multi, rows=None):
    """[(kind, pa, pb, R0, R1, nbytes)] of the whole operator; B runs over occupied rows."""
    out = []
    for kind, n in (("oo", nocc), ("vv", nvir), ("ov", nocc)):
        for pa, pb, R0, R1 in row_strips(n, multi, rows):
            ncol = pair_offset(nvir, multi) if kind == "ov" else R1
            out.append((kind, pa, pb, R0, R1, 8 * (R1 - R0) * ncol))
    return out


def _pack_rows(rows, P0, out):
    """Rows P0, P0 + 1, ... of the packed lower triangle ``out`` from the leading
    columns of ``rows``."""
    m = rows.shape[0]
    keep = np.arange(P0 + m)[None, :] <= P0 + np.arange(m)[:, None]
    out[pack_offset(P0):pack_offset(P0 + m)] = rows[:, :P0 + m][keep]


def pack_eri(vvvv, oovv, oooo):
    """Packed Gram triangles (the format of ``gpu_ao2mo_blocks``) from physicist blocks
    vvvv[a,b,c,d] = <ab|cd>, oovv[i,j,a,b] = <ij|ab>, oooo: (pq|rs) for Q <= P at
    P (P + 1) / 2 + Q, P = (p, r) over the compact pairs p >= r of vvvv and oooo and
    over i nvir + a of oovv.  One first orbital at a time, so no block is copied whole."""
    def compact(block):
        n = block.shape[0]
        p, q = np.tril_indices(n)
        out = np.empty(pack_offset(pack_offset(n)))
        for a in range(n):
            # rows (a, c), c <= a: (ac|bd) = <ab|cd> over the compact (b, d)
            _pack_rows(np.asarray(block[a, :, :a + 1]).transpose(1, 0, 2)[:, p, q],
                       pack_offset(a), out)
        return out

    no, nv = oovv.shape[0], oovv.shape[2]
    ov = np.empty(pack_offset(no * nv))
    for i in range(no):
        # rows (i, a): (ia|jb) = <ij|ab> over (j, b)
        _pack_rows(np.asarray(oovv[i]).transpose(1, 0, 2).reshape(nv, no * nv), i * nv, ov)
    return compact(vvvv), ov, compact(oooo)


def _tri(a, b):
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return hi * (hi + 1) // 2 + lo


def _fpair(p, q):
    return np.where(p == q, np.sqrt(0.5), 1.0)


def _sym_strip(G, pp, pq, R0, R1, sgn):
    """Rows [R0, R1) and columns [0, R1) of M_oo / M_vv from the compact Gram triangle
    G, <pq|rs> = (pr|qs) at tri(tri(p, r), tri(q, s)); the columns from R0 on form the
    diagonal block, which both GEMMs of a product use: halved."""
    p, q, r, s = pp[R0:R1, None], pq[R0:R1, None], pp[None, :R1], pq[None, :R1]
    S = G[_tri(_tri(p, r), _tri(q, s))] + sgn * G[_tri(_tri(p, s), _tri(q, r))]
    S *= _fpair(p, q) * _fpair(r, s)
    S[:, R0:] *= 0.5
    return S


def _b_strip(G, io, jo, av, bv, R0, R1, nvir, sgn):
    """Rows [R0, R1) of B from the oovv Gram triangle G, <ij|ab> = (ia|jb), Gram pair
    index i nvir + a."""
    i, j = io[R0:R1, None], jo[R0:R1, None]
    S = G[_tri(i * nvir + av, j * nvir + bv)] + sgn * G[_tri(i * nvir + bv, j * nvir + av)]
    S *= _fpair(i, j) * _fpair(av, bv)
    return S


def pair_operator(vvvv, oovv, oooo, nocc, nvir, multi, rows=None):
    """The pair-space operator of multiplicity ``multi`` ('s' or 't') from the packed
    Gram blocks of ``pack_eri``, as NumPy strips (``rows`` as for ``row_strips``)."""
    if multi not in ("s", "t"):
        raise ValueError(f"multi must be 's' or 't', got {multi!r}")
    sgn = 1.0 if multi == "s" else -1.0
    k = 0 if multi == "s" else -1
    pairs = {"oo": np.tril_indices(nocc, k), "vv": np.tril_indices(nvir, k)}
    gram = {"oo": np.asarray(oooo), "vv": np.asarray(vvvv), "ov": np.asarray(oovv)}
    strips = []
    for kind, _, _, R0, R1, _ in operator_strips(nocc, nvir, multi, rows):
        if kind == "ov":
            S = _b_strip(gram[kind], *pairs["oo"], *pairs["vv"], R0, R1, nvir, sgn)
        else:
            S = _sym_strip(gram[kind], *pairs[kind], R0, R1, sgn)
        strips.append((kind, R0, R1, S))
    return PairOperator(nocc, nvir, multi, strips)


def apply_pair_operator(op, X):
    """M X for trial vectors X (full_dim, ntri) in the pair basis of ``op.multi``."""
    noo = pair_offset(op.nocc, op.multi)
    Y = np.zeros_like(X)
    Xo, Xv, Yo, Yv = X[:noo], X[noo:], Y[:noo], Y[noo:]
    for kind, R0, R1, S in op.strips:
        if kind == "ov":
            Yo[R0:R1] += S @ Xv
            Yv += S.T @ Xo[R0:R1]
        else:
            Xs, Ys = (Xo, Yo) if kind == "oo" else (Xv, Yv)
            Ys[R0:R1] += S @ Xs[:R1]
            Ys[:R1] += S.T @ Xs[R0:R1]
    return Y
