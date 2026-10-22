"""GPU (cupy) Davidson products for the pp-RPA use_eri path with the pair-space operator
of ``pair_operator`` (M_oo, M_vv and B as row strips; see that module).

A product is two GEMMs per strip, Y[R] += S X[:R1] and Y[:R1] += S^T X[R] (for B:
Y_oo[R] += S X_vv and Y_vv += S^T X_oo[R]), with no gather and no zeros.  Both terms of
a row whose first orbital is p sit in Gram rows of first orbital max(p, r) with r <= p,
so a strip over the first orbitals [pa, pb) is converted on the device from the Gram
rows of those orbitals alone.  ``gpu_ao2mo_pair_operator`` does that to every strip of
the FFT ao2mo as it is built, so the Gram blocks are never stored and only M (about 2/3
of them; B is half of oovv) reaches the host.  ``gdf_pair_operator`` does the same from
three-index factors, whose Gram rows are one GEMM per strip.  ``attach_gpu_eri_contraction``
takes that operator, or Gram blocks, which it converts at the first product for the
multiplicity the solver then has.
"""
import math

import numpy as np
import cupy as cp
from cupy import cublas
from pyscf.lib import logger

from lib_pprpa.gpu_ao2mo import PAIR_BLK_CAP, _pack_tile, _PairLayout, gpu_ao2mo_strips
from lib_pprpa.gpu_emulation import emulation_pays, fp64_emulation
from lib_pprpa.gpu_mem import empty_pinned, free_bytes, pack_offset, plan_budget
from lib_pprpa.gpu_multi import default_devices, each
from lib_pprpa.pair_operator import (STRIP_ROWS, PairOperator, operator_strips, pair_offset,
                                     pair_operator_bytes, row_strips)

__all__ = ['attach_gpu_eri_contraction', 'gdf_pair_operator', 'gpu_ao2mo_pair_operator',
           'release_gpu_eri']

# (ntri, full_dim) float64 arrays one product holds besides the strips: T, X, Y and
# the two of Y^T + orb T.
_PRODUCT_ARRAYS = 5

_PREAMBLE = r'''
__device__ long long tri(long long a, long long b) {
    return a > b ? a * (a + 1) / 2 + b : b * (b + 1) / 2 + a;
}
__device__ double fpair(long long p, long long q) {
    return p == q ? 0.70710678118654752440 : 1.0;
}
'''

# Rows [R0, R0 + n) and columns [0, ncol) of M_oo / M_vv from the compact Gram rows
# starting at packed offset ``base``: <pq|rs> = (pr|qs) at tri(tri(p, r), tri(q, s)).
# The columns from R0 on form the diagonal block, which both GEMMs use: halved.
_SYM_STRIP = cp.ElementwiseKernel(
    'raw float64 E, raw int32 pp, raw int32 pq, int64 base, int64 R0, int64 ncol, float64 sgn',
    'float64 out',
    '''
    const long long P = R0 + i / ncol, Q = i % ncol;
    const long long p = pp[P], q = pq[P], r = pp[Q], s = pq[Q];
    const double v = E[tri(tri(p, r), tri(q, s)) - base]
                     + sgn * E[tri(tri(p, s), tri(q, r)) - base];
    out = (Q >= R0 ? 0.5 : 1.0) * fpair(p, q) * fpair(r, s) * v;
    ''',
    'pprpa_pair_operator_sym', preamble=_PREAMBLE)

# Rows [R0, R0 + n) of B from the oovv Gram rows starting at ``base``: <ij|ab> =
# (ia|jb), Gram pair index i nv + a.
_B_STRIP = cp.ElementwiseKernel(
    'raw float64 E, raw int32 op, raw int32 oq, raw int32 vp, raw int32 vq, int64 base, '
    'int64 R0, int64 ncol, int64 nv, float64 sgn',
    'float64 out',
    '''
    const long long P = R0 + i / ncol, Q = i % ncol;
    const long long ii = op[P], j = oq[P], a = vp[Q], b = vq[Q];
    const double v = E[tri(ii * nv + a, j * nv + b) - base]
                     + sgn * E[tri(ii * nv + b, j * nv + a) - base];
    out = fpair(ii, j) * fpair(a, b) * v;
    ''',
    'pprpa_pair_operator_b', preamble=_PREAMBLE)


def gpu_ao2mo_pair_operator(cell, cocc, cvir, mesh, multi, rows=None, pair_blk=None,
                            devices=None, emulate=True, verbose=None):
    """The pair-space operator of multiplicity ``multi`` ('s' or 't') straight from the
    FFT ao2mo (``gpu_ao2mo.gpu_ao2mo_strips``): each Gram strip is converted on its
    device and only the operator strips reach the host.

    Kwargs:
        rows : first orbitals per operator strip; default up to STRIP_ROWS pairs.
        pair_blk, devices, emulate : as for ``gpu_ao2mo.gpu_ao2mo_blocks``.
    Returns:
        A PairOperator in pinned host memory, for ``attach_gpu_eri_contraction``.
    """
    if multi not in ("s", "t"):
        raise ValueError(f"multi must be 's' or 't', got {multi!r}")
    nocc, nvir = np.asarray(cocc).shape[1], np.asarray(cvir).shape[1]
    sink = _OperatorSink(nocc, nvir, multi, rows)
    gpu_ao2mo_strips(cell, cocc, cvir, mesh, sink, pair_blk, devices, emulate, verbose)
    return PairOperator(nocc, nvir, multi, sink.strips)


def gdf_pair_operator(lpq, nocc, multi, rows=None, pair_blk=None, resident=None, emulate=True,
                      verbose=None):
    """The pair-space operator of multiplicity ``multi`` ('s' or 't') from the factors
    lpq[L, p, q] of the active orbitals (occupied first; NumPy, pinned or CuPy), without
    the Gram blocks: per block the factor columns of its pairs, F[P, L] = lpq[L, p, q],
    go to the device once and each Gram row strip F[P0:P1] F[:P1]^T is converted as it
    is formed, as ``gpu_ao2mo_pair_operator`` does for the FFT ao2mo.

    Kwargs:
        rows : first orbitals per operator strip; default up to STRIP_ROWS pairs.
        pair_blk : force the Gram strip width in pairs (default: from free memory).
        resident : keep the strips on the device instead of pinned host memory;
            default when the operator and the largest block's build fit the free memory.
        emulate : run the strip GEMMs that gain from it as emulated fp64
            (``gpu_emulation.emulation_pays``); False keeps every GEMM native.
    Returns:
        A PairOperator, for ``attach_gpu_eri_contraction``.
    """
    if multi not in ("s", "t"):
        raise ValueError(f"multi must be 's' or 't', got {multi!r}")
    log = logger.new_logger(None, logger.NOTE if verbose is None else verbose)
    t0 = log.init_timer()
    nocc = int(nocc)
    naux, nvir = lpq.shape[0], lpq.shape[1] - nocc
    blocks = (("vvvv", (nocc, nvir), (nocc, nvir)), ("oooo", (0, nocc), (0, nocc)),
              ("oovv", (0, nocc), (nocc, nvir)))
    layouts = [_PairLayout(nA, nB, compact=name != "oovv") for name, (_, nA), (_, nB) in blocks]
    sink = _OperatorSink(nocc, nvir, multi, rows)
    need = sum(s[-1] for s in operator_strips(nocc, nvir, multi, rows))   # bound: strips only shrink
    if resident is None:
        work = max(_gdf_block_bytes(layout, naux) + sink.device_bytes(name)
                   for (name, _, _), layout in zip(blocks, layouts))
        resident = need + work <= plan_budget(free_bytes())
    sink.device = bool(resident)
    log.info("GDF pair operator (%s): naux=%d nocc=%d nvir=%d, operator <= %.2f GB %s", multi,
             naux, nocc, nvir, need / 1e9, "on the device" if resident else "in pinned host memory")
    for (name, (oa, _), (ob, _)), layout in zip(blocks, layouts):
        _gdf_block(lpq, oa, ob, layout, name, sink, pair_blk, emulate, log)
        t0 = log.timer(f"GDF pair operator {name}", *t0)
    return PairOperator(nocc, nvir, multi, sink.strips)


def _gdf_block_bytes(layout, naux):
    """Device bytes of one block's build: F, an aux slice of at most its size, and the
    Gram strip buffers at the widest strip (PAIR_BLK_CAP pairs)."""
    b = min(PAIR_BLK_CAP, layout.npair)
    return 16 * layout.npair * naux + 8 * b * (layout.npair + b)


def _pair_factor(lpq, oa, ob, layout):
    """F[P, L] = lpq[L, oa + p, ob + q] over the pairs P = (p, q) of ``layout``, on the
    device; lpq is uploaded in aux slices within the free memory and no larger than F,
    so the upload at most doubles the block's footprint."""
    naux, n = lpq.shape[0], lpq.shape[1]
    p, q = layout.index_arrays()
    p += oa
    q += ob
    F = cp.empty((layout.npair, naux))
    # a slice, its gathered pairs and their transposed copy into F
    per_aux = 8 * (n * n + 2 * layout.npair)
    step = max(1, min(naux, plan_budget(free_bytes()) // per_aux, F.nbytes // per_aux))
    for a0 in range(0, naux, step):
        a1 = min(naux, a0 + step)
        F[:, a0:a1] = cp.asarray(lpq[a0:a1])[:, p, q].T
    return F


def _gram_strip_width(layout, pair_blk, extra):
    """Gram strip width in pairs: 8 b^2 + 8 npair b (a b x b tile and the packed rows of
    the strip) within the free memory less ``extra``, whole rows, capped at PAIR_BLK_CAP."""
    if pair_blk is None:
        rest = max(0, plan_budget(free_bytes()) - extra)
        lin = 8 * layout.npair
        pair_blk = min(int((-lin + math.sqrt(lin * lin + 32 * rest)) / 16), PAIR_BLK_CAP)
    blk = max(layout.min_blk, int(pair_blk))
    if not layout.compact:
        blk = (blk // layout.nB) * layout.nB
    return min(blk, layout.npair)


def _gdf_block(lpq, oa, ob, layout, name, sink, pair_blk, emulate, log):
    """Gram rows of block ``name`` strip by strip from the factors, each strip handed to
    ``sink``; the tiles at or above the diagonal of a strip are dropped by _pack_tile."""
    F = _pair_factor(lpq, oa, ob, layout)
    blk = _gram_strip_width(layout, pair_blk, sink.device_bytes(name))
    tasks = layout.split(0, layout.nA, blk)
    emit = sink.prepare(name, tasks)
    log.debug("GDF pair operator %s: npair=%d naux=%d pair_blk=%d strips=%d", name,
              layout.npair, F.shape[1], blk, len(tasks))
    nbuf = max(pack_offset(layout.pairs(a, b)[1]) - pack_offset(layout.pairs(a, b)[0])
               for a, b in tasks)
    buf_all, tile_all = cp.empty(nbuf), cp.empty(blk * blk)
    for pa, pb in tasks:
        P0, P1 = layout.pairs(pa, pb)
        base = pack_offset(P0)
        buf = buf_all[:pack_offset(P1) - base]
        for ra, rb in layout.split(0, pb, blk):
            Q0, Q1 = layout.pairs(ra, rb)
            tile = tile_all[:(P1 - P0) * (Q1 - Q0)].reshape(P1 - P0, Q1 - Q0)
            with fp64_emulation('eager' if emulate and emulation_pays(P1 - P0, Q1 - Q0)
                                else None):
                cublas.gemm("N", "T", F[P0:P1], F[Q0:Q1], out=tile)
            _pack_tile(buf, tile, P0, Q0, base)
        emit(pa, pb, buf, base)
    F = buf_all = tile_all = None
    cp.get_default_memory_pool().free_all_blocks()


def attach_gpu_eri_contraction(pprpa, *eri, mode="auto", rows=None, devices=None, verbose=None):
    """Route the use_eri Davidson products through the pair-space operator on the GPU.

    Args:
        eri : a PairOperator (``gpu_ao2mo_pair_operator``), or the packed Gram
            triangles vvvv, oovv, oooo of ``gpu_ao2mo_blocks`` (NumPy or CuPy;
            ``pair_operator.pack_eri`` converts physicist tensors), which are converted
            at the first product for the solver's multiplicity and then dropped.
    Kwargs:
        mode : 'auto', 'resident', 'split' or 'streamed'.  'auto' keeps the strips on
            the device when they fit its free memory beside one more strip (the
            Gram slice being converted) and the product arrays of the largest trial
            block (max_vec + nroot vectors), divides them over ``devices`` when an
            equal share plus one strip fits each, and streams them otherwise.
        rows : first orbitals per strip when converting Gram blocks; default up to
            STRIP_ROWS pairs.
        devices : CUDA device ids for the split mode (default: the current device).
    """
    log = logger.new_logger(pprpa, logger.INFO if verbose is None else verbose)
    nocc, nvir = pprpa.nocc, pprpa.nvir
    if len(eri) == 1 and isinstance(eri[0], PairOperator):
        pre, blocks, multi, need = eri[0], None, eri[0].multi, eri[0].nbytes
        if (pre.nocc, pre.nvir) != (nocc, nvir):
            raise ValueError(f"operator for nocc={pre.nocc} nvir={pre.nvir}, solver has "
                             f"nocc={nocc} nvir={nvir}")
    elif len(eri) == 3:
        pre, blocks, multi = None, _check_packed(*eri, nocc, nvir), None
        need = pair_operator_bytes(nocc, nvir, "s")
    else:
        raise TypeError("expected a PairOperator or the vvvv, oovv, oooo Gram blocks")
    devices = default_devices(devices)
    if mode == "auto" and pre is not None and all(isinstance(S, cp.ndarray)
                                                  for *_, S in pre.strips):
        mode = "resident"                       # already on the device: nothing to place
    if mode == "auto":
        if pre is not None:
            strip = max(S.nbytes for *_, S in pre.strips)
        else:
            strip = max(s[-1] for s in operator_strips(nocc, nvir, "s", rows))
        temps = 8 * _PRODUCT_ARRAYS * (pprpa.max_vec + pprpa.nroot) * (
            pair_offset(nocc, "s") + pair_offset(nvir, "s"))
        free = each(devices, lambda rank: free_bytes())
        nshare = max(devices.count(d) for d in devices)
        if need + strip + temps <= plan_budget(free[0]):
            mode = "resident"
        elif (len(devices) > 1 and
              need / len(devices) + strip + temps <= plan_budget(min(free) // nshare)):
            mode = "split"
        else:
            mode = "streamed"
    if mode not in ("resident", "split", "streamed"):
        raise ValueError(f"unknown mode {mode!r}")
    pprpa._use_eri = True
    pprpa._gpu_eri = dict(blocks=blocks, prebuilt=pre, mode=mode, rows=rows, devices=devices,
                          multi=multi, strips=None, log=log)
    pprpa.eri_gram = pprpa._pair_op = None
    pprpa.contraction = lambda tri_vec: _contraction(pprpa, tri_vec)
    pprpa.eri_nbytes = lambda: _eri_nbytes(pprpa)
    log.info("GPU pair operator: mode=%s devices=%s operator=%.2f GB%s nocc=%d nvir=%d",
             mode, devices, need / 1e9, f" ({multi})" if pre else " (singlet, from Gram)",
             nocc, nvir)
    return pprpa


def release_gpu_eri(pprpa):
    """Drop the operator and the Gram blocks held by attach_gpu_eri_contraction and free
    the cupy pools; the solver keeps its eigenvalues and eigenvectors."""
    op = pprpa.__dict__.pop("_gpu_eri", None)
    pprpa.__dict__.pop("contraction", None)
    pprpa.__dict__.pop("eri_nbytes", None)
    if op is not None:
        devices = op["devices"]
        op.clear()
        each(devices, lambda rank: cp.get_default_memory_pool().free_all_blocks())
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _check_packed(vvvv, oovv, oooo, nocc, nvir):
    blocks = {"vvvv": (vvvv, pack_offset(pack_offset(nvir))),
              "oooo": (oooo, pack_offset(pack_offset(nocc))),
              "oovv": (oovv, pack_offset(nocc * nvir))}
    for name, (block, n) in blocks.items():
        if block.ndim != 1 or block.shape[0] != n:
            raise ValueError(f"{name}: expected a packed block of {n} elements, got shape "
                             f"{block.shape} (pack_eri converts physicist tensors)")
    return {name: block for name, (block, n) in blocks.items()}


def _pairs(n, multi):
    p, q = np.tril_indices(n, 0 if multi == "s" else -1)
    return cp.asarray(p, dtype=np.int32), cp.asarray(q, dtype=np.int32)


def _convert_strip(kind, R0, R1, E, base, pairs, nvir, sgn):
    """Operator rows [R0, R1) of ``kind`` on the device from the Gram rows ``E`` (packed
    offset ``base``) that hold their first orbitals."""
    if kind == "ov":
        ncol = pairs["vv"][0].size
        S = cp.empty((R1 - R0, ncol))
        _B_STRIP(E, *pairs["oo"], *pairs["vv"], base, R0, ncol, nvir, sgn, S)
    else:
        S = cp.empty((R1 - R0, R1))
        _SYM_STRIP(E, *pairs[kind], base, R0, R1, sgn, S)
    return S


class _OperatorSink:
    """Converts each Gram strip of ``gpu_ao2mo_strips`` into the operator strips over
    the same first orbitals and keeps those in pinned host memory, or on the device
    that converted them if ``device``."""
    _KIND = {"vvvv": "vv", "oooo": "oo", "oovv": "ov"}

    def __init__(self, nocc, nvir, multi, rows, device=False):
        self.nocc, self.nvir, self.multi, self.rows = nocc, nvir, multi, rows
        self.device = device
        self.strips = []
        self._pairs = {}

    def _ncol(self, kind, R1):
        return pair_offset(self.nvir, self.multi) if kind == "ov" else R1

    def device_bytes(self, name):
        """One converted strip (at least one orbital of pairs) and the pair index arrays."""
        kind = self._KIND[name]
        n = self.nvir if kind == "vv" else self.nocc
        npair = pair_offset(self.nocc, self.multi) + pair_offset(self.nvir, self.multi)
        return 8 * max(STRIP_ROWS, n) * self._ncol(kind, pair_offset(n, self.multi)) + 8 * npair

    def prepare(self, name, tasks):
        kind = self._KIND[name]
        n = self.nvir if kind == "vv" else self.nocc
        sgn = 1.0 if self.multi == "s" else -1.0
        plan = {}
        for pa, pb in tasks:
            plan[pa, pb] = []
            for _, _, R0, R1 in row_strips(n, self.multi, self.rows, pa, pb):
                host = None if self.device else empty_pinned((R1 - R0, self._ncol(kind, R1)))
                plan[pa, pb].append((len(self.strips), R0, R1, host))
                self.strips.append((kind, R0, R1, host))

        def emit(pa, pb, buf, base):
            pairs = self._device_pairs()
            for k, R0, R1, host in plan[pa, pb]:
                S = _convert_strip(kind, R0, R1, buf, base, pairs, self.nvir, sgn)
                if host is None:
                    self.strips[k] = (kind, R0, R1, S)
                else:
                    S.get(out=host)
        return emit

    def _device_pairs(self):
        dev = cp.cuda.Device().id
        if dev not in self._pairs:
            self._pairs[dev] = {"oo": _pairs(self.nocc, self.multi),
                                "vv": _pairs(self.nvir, self.multi)}
        return self._pairs[dev]


def _convert(blocks, plan, nocc, nvir, multi, host):
    """Build the strips of ``plan`` on the current device from uploaded Gram slices;
    pinned host copies if ``host``."""
    sgn = 1.0 if multi == "s" else -1.0
    pairs = {"oo": _pairs(nocc, multi), "vv": _pairs(nvir, multi)}
    out = []
    for kind, pa, pb, R0, R1, _ in plan:
        if kind == "ov":
            g0, g1 = pack_offset(pa * nvir), pack_offset(pb * nvir)
            E = cp.asarray(blocks["oovv"][g0:g1])
        else:
            g0, g1 = pack_offset(pack_offset(pa)), pack_offset(pack_offset(pb))
            E = cp.asarray(blocks["vvvv" if kind == "vv" else "oooo"][g0:g1])
        S = _convert_strip(kind, R0, R1, E, g0, pairs, nvir, sgn)
        E = None
        if host:
            h = empty_pinned(S.shape)
            S.get(out=h)
            S = h
        out.append((kind, R0, R1, S))
    return out


def _assign(sizes, n):
    """Indices of ``sizes`` per rank: largest first onto the least-loaded rank."""
    parts, load = [[] for _ in range(n)], [0] * n
    for i in sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True):
        rank = min(range(n), key=load.__getitem__)
        parts[rank].append(i)
        load[rank] += sizes[i]
    return parts


def _build(op, nocc, nvir):
    mode, devices, multi, pre = op["mode"], op["devices"], op["multi"], op["prebuilt"]
    t0 = op["log"].init_timer()
    if pre is not None:
        items = pre.strips
        if mode == "streamed":
            strips = items
        elif mode == "resident":
            strips = [(k, R0, R1, cp.asarray(S)) for k, R0, R1, S in items]
        else:
            parts = _assign([S.nbytes for *_, S in items], len(devices))
            strips = each(devices, lambda rank: [(k, R0, R1, cp.asarray(S)) for k, R0, R1, S
                                                 in (items[i] for i in parts[rank])])
    else:
        blocks = op["blocks"]
        plan = operator_strips(nocc, nvir, multi, op["rows"])
        if mode == "split":
            parts = _assign([s[-1] for s in plan], len(devices))
            strips = each(devices, lambda rank: _convert(
                blocks, [plan[i] for i in parts[rank]], nocc, nvir, multi, False))
        else:
            strips = _convert(blocks, plan, nocc, nvir, multi, mode == "streamed")
    op.update(strips=strips, blocks=None, prebuilt=None)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()   # dropped host blocks, if unreferenced
    flat = sum(strips, []) if mode == "split" else strips
    op["log"].timer("pair operator placement" if pre is not None else "pair operator build", *t0)
    op["log"].info("GPU pair operator (%s, %s): %d strips, %.2f GB", multi, mode, len(flat),
                   sum(strip[3].nbytes for strip in flat) / 1e9)


def _eri_nbytes(pprpa):
    """Bytes of the operator strips for the solver's multiplicity, plus the Gram blocks
    they are converted from while those are held (until the first product)."""
    op = pprpa._gpu_eri
    if op["strips"] is not None:
        flat = sum(op["strips"], []) if op["mode"] == "split" else op["strips"]
        return sum(S.nbytes for *_, S in flat)
    if op["prebuilt"] is not None:
        return op["prebuilt"].nbytes
    plan = operator_strips(pprpa.nocc, pprpa.nvir, pprpa.multi, op["rows"])
    return sum(s[-1] for s in plan) + sum(b.nbytes for b in op["blocks"].values())


def _apply(strips, X, Y, noo):
    """Y += M X over ``strips`` (device or pinned host arrays)."""
    Xo, Xv, Yo, Yv = X[:noo], X[noo:], Y[:noo], Y[noo:]
    for kind, R0, R1, S in strips:
        S = cp.asarray(S)
        if kind == "ov":
            cublas.gemm("N", "N", S, Xv, out=Yo[R0:R1], beta=1.0)
            cublas.gemm("T", "N", S, Xo[R0:R1], out=Yv, beta=1.0)
        else:
            Xs, Ys = (Xo, Yo) if kind == "oo" else (Xv, Yv)
            cublas.gemm("N", "N", S, Xs[:R1], out=Ys[R0:R1], beta=1.0)
            cublas.gemm("T", "N", S, Xs[R0:R1], out=Ys[:R1], beta=1.0)


def _contraction(pprpa, tri_vec):
    op = pprpa._gpu_eri
    nocc, nvir, multi = pprpa.nocc, pprpa.nvir, pprpa.multi
    if op["multi"] is None:
        op["multi"] = multi
    if op["multi"] != multi:
        raise ValueError(f"pair operator is for multi={op['multi']!r}, the solver has "
                         f"{multi!r}; attach the ERIs again")
    if op["strips"] is None:
        _build(op, nocc, nvir)
    noo = pair_offset(nocc, multi)
    T = cp.asarray(tri_vec)
    X = cp.ascontiguousarray(T.T)                 # (full_dim, ntri)
    if op["mode"] == "split":
        xh = X.get()

        def partial(rank):
            Xd = cp.asarray(xh)
            Yd = cp.zeros_like(Xd)
            _apply(op["strips"][rank], Xd, Yd, noo)
            return Yd.get()
        Y = cp.asarray(sum(each(op["devices"], partial)))
    else:
        Y = cp.zeros_like(X)
        _apply(op["strips"], X, Y, noo)

    # orbital-energy diagonal term: (e_p + e_q - 2 mu), hh block negated
    k = 0 if multi == "s" else -1
    me = cp.asarray(pprpa.mo_energy)
    eo, ev = me[:nocc], me[nocc:]
    orb = cp.concatenate(((eo[None, :] + eo[:, None])[cp.tril_indices(nocc, k)],
                          (ev[None, :] + ev[:, None])[cp.tril_indices(nvir, k)])) - 2.0 * pprpa.mu
    orb[:noo] *= -1.0
    return cp.asnumpy(Y.T + orb[None, :] * T)
