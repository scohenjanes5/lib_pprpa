'''Exchange potentials of a batch of codensities summed into W by five custom FFT kernels.

For codensities rho_P = A[a_P] B[b_P] and the symmetrised half kernel w of ``half_kernel``,
W[rows_t] += sum_k scale[src_k] S[src_k] V[rho_(p0 + pair_k)], V = ifft(w fft(rho)): the sum of
``gpu_coulomb.coulomb_potential`` and ``gpu_exchange``'s accumulation.  Each pass runs 1D Stockham
FFTs along the contiguous axis in shared memory and writes transposed ([x,y,z] -> [kz,x,y] ->
[ky,kz,x] -> [kz,x,ky] -> [x,y,kz] -> W): codensity and R2C along z, along y, along x with w and
the inverse along x, the inverse along y, C2R along z with the S sum.  Meshes whose sides factor
into 2, 3, 5 and 7 (``applies``); compiled per mesh and device with NVRTC.  One device per call.
'''

import threading

import numpy as np
import cupy as cp

__all__ = ['applies', 'FFTChain']

# (lines per block, threads, shared-index padding shift) of each pass: the fastest shapes shared
# by the B200 and the H200 on the 160^3 mesh, each within 4% of either card's best.
_SHAPES = ((8, 64, 4), (16, 160, 4), (8, 128, 3), (8, 160, 4), (2, 64, 0))
# Pass 1 with at least 12 resident blocks per SM: 80 registers, no spills, 4% faster on a B200.
_PASS1_MIN_BLOCKS = 12
_SMOOTH = (2, 3, 5, 7)
_LOCK = threading.Lock()
_KERNELS = {}


def _twiddle_tables():
    out = []
    for R in (16, 15, 14, 12, 10, 9, 6):
        m = np.arange(R)
        c = ', '.join(repr(float(x)) for x in np.cos(2 * np.pi * m / R))
        s = ', '.join(repr(float(x)) for x in -np.sin(2 * np.pi * m / R))
        out.append(f'template <> struct Tw<{R}> {{ static __device__ __forceinline__ double2 '
                   f'at(int m) {{ const double c[{R}] = {{{c}}}; const double s[{R}] = {{{s}}}; '
                   f'return make_double2(c[m], s[m]); }} }};')
    return '\n'.join(out)


_SRC = r'''
__device__ __forceinline__ double2 cmul(double2 a, double2 b)
{ return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ double2 conjif(double2 a, bool c) { if (c) a.y = -a.y; return a; }
__device__ __forceinline__ double2 add(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 sub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ double2 mi(double2 a) { return make_double2(a.y, -a.x); }   // -i a
__device__ __forceinline__ double2 zero2() { return make_double2(0.0, 0.0); }

// Forward radix-R DFT in registers.
template <int R> struct Dft;
template <> struct Dft<2> {
    static __device__ __forceinline__ void run(double2 *v)
    { const double2 a = v[0]; v[0] = add(a, v[1]); v[1] = sub(a, v[1]); }
};
template <> struct Dft<4> {
    static __device__ __forceinline__ void run(double2 *v)
    {
        const double2 t0 = add(v[0], v[2]), t1 = sub(v[0], v[2]);
        const double2 t2 = add(v[1], v[3]), t3 = mi(sub(v[1], v[3]));
        v[0] = add(t0, t2); v[2] = sub(t0, t2); v[1] = add(t1, t3); v[3] = sub(t1, t3);
    }
};
template <> struct Dft<8> {
    static __device__ __forceinline__ void run(double2 *v)
    {
        const double s = 0.70710678118654752440;
        double2 b[4], c[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) { b[k] = add(v[k], v[k + 4]); c[k] = sub(v[k], v[k + 4]); }
        c[1] = make_double2(s * (c[1].x + c[1].y), s * (c[1].y - c[1].x));
        c[2] = mi(c[2]);
        c[3] = make_double2(s * (c[3].y - c[3].x), -s * (c[3].x + c[3].y));
        Dft<4>::run(b);
        Dft<4>::run(c);
#pragma unroll
        for (int m = 0; m < 4; ++m) { v[2 * m] = b[m]; v[2 * m + 1] = c[m]; }
    }
};
// odd R: u_q = v0 + sum_r (v_r + v_{R-r}) cos(2 pi q r / R) - i (v_r - v_{R-r}) sin(2 pi q r / R)
template <int R> struct DftOdd {
    static __device__ __forceinline__ void run(double2 *v, const double *C, const double *S)
    {
        constexpr int h = (R - 1) / 2;
        double2 p[h], m[h], u[R];
#pragma unroll
        for (int r = 1; r <= h; ++r) {
            p[r - 1] = add(v[r], v[R - r]);
            m[r - 1] = sub(v[r], v[R - r]);
        }
        u[0] = v[0];
#pragma unroll
        for (int r = 0; r < h; ++r) u[0] = add(u[0], p[r]);
#pragma unroll
        for (int q = 1; q <= h; ++q) {
            double2 re = v[0], im = zero2();
#pragma unroll
            for (int r = 1; r <= h; ++r) {
                const int j = (q * r) % R;
                re.x += C[j] * p[r - 1].x; re.y += C[j] * p[r - 1].y;
                im.x += S[j] * m[r - 1].x; im.y += S[j] * m[r - 1].y;
            }
            u[q] = add(re, mi(im));
            u[R - q] = sub(re, mi(im));
        }
#pragma unroll
        for (int q = 0; q < R; ++q) v[q] = u[q];
    }
};
template <> struct Dft<3> {
    static __device__ __forceinline__ void run(double2 *v)
    {
        const double C[3] = {1.0, -0.5, -0.5};
        const double S[3] = {0.0, 0.86602540378443864676, -0.86602540378443864676};
        DftOdd<3>::run(v, C, S);
    }
};
template <> struct Dft<5> {
    static __device__ __forceinline__ void run(double2 *v)
    {
        const double C[5] = {1.0, 0.30901699437494742410, -0.80901699437494742410,
                             -0.80901699437494742410, 0.30901699437494742410};
        const double S[5] = {0.0, 0.95105651629515357212, 0.58778525229247312917,
                             -0.58778525229247312917, -0.95105651629515357212};
        DftOdd<5>::run(v, C, S);
    }
};
template <> struct Dft<7> {
    static __device__ __forceinline__ void run(double2 *v)
    {
        const double C[7] = {1.0, 0.62348980185873353053, -0.22252093395631440429,
                             -0.90096886790241912624, -0.90096886790241912624,
                             -0.22252093395631440429, 0.62348980185873353053};
        const double S[7] = {0.0, 0.78183148246802980871, 0.97492791218182360702,
                             0.43388373911755812048, -0.43388373911755812048,
                             -0.97492791218182360702, -0.78183148246802980871};
        DftOdd<7>::run(v, C, S);
    }
};
// R = A B by one Cooley-Tukey split; Tw<R>::at(m) = exp(-2 pi i m / R)
template <int R> struct Tw;
TWIDDLE_TABLES
template <int A, int B> struct DftCT {
    static __device__ __forceinline__ void run(double2 *v)
    {
        constexpr int R = A * B;
        double2 y[R];
#pragma unroll
        for (int n2 = 0; n2 < B; ++n2) {
            double2 t[A];
#pragma unroll
            for (int n1 = 0; n1 < A; ++n1) t[n1] = v[B * n1 + n2];
            Dft<A>::run(t);
#pragma unroll
            for (int k1 = 0; k1 < A; ++k1)
                y[n2 * A + k1] = (n2 * k1) ? cmul(t[k1], Tw<R>::at(n2 * k1)) : t[k1];
        }
#pragma unroll
        for (int k1 = 0; k1 < A; ++k1) {
            double2 t[B];
#pragma unroll
            for (int n2 = 0; n2 < B; ++n2) t[n2] = y[n2 * A + k1];
            Dft<B>::run(t);
#pragma unroll
            for (int k2 = 0; k2 < B; ++k2) v[k1 + A * k2] = t[k2];
        }
    }
};
template <> struct Dft<16> : DftCT<4, 4> {};
template <> struct Dft<15> : DftCT<3, 5> {};
template <> struct Dft<14> : DftCT<2, 7> {};
template <> struct Dft<12> : DftCT<4, 3> {};
template <> struct Dft<10> : DftCT<2, 5> {};
template <> struct Dft<9> : DftCT<3, 3> {};
template <> struct Dft<6> : DftCT<2, 3> {};

constexpr int nx = NX, ny = NY, nz = NZ, nkz = NZ / 2 + 1;
constexpr long long ng = (long long)NX * NY * NZ;
// passes 1-4: the largest radix <= 16 dividing m (fewest shared-memory round trips)
constexpr int pick(int m)
{
    constexpr int cand[13] = {16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2};
    for (int i = 0; i < 13; ++i)
        if (m % cand[i] == 0) return cand[i];
    return 2;
}
// pass 5: odd radices first, at most 8
constexpr int pick_pp(int m)
{ return m % 7 == 0 ? 7 : m % 5 == 0 ? 5 : m % 3 == 0 ? 3 : m % 8 == 0 ? 8 : m % 4 == 0 ? 4 : 2; }
// one padding slot every 2^SH entries: breaks the bank conflicts of the strided Stockham writes
template <int SH> __host__ __device__ constexpr int padi(int i) { return SH ? i + (i >> SH) : i; }
template <int SH> __host__ __device__ constexpr int ldof(int n)
{ return (padi<SH>(n - 1) + 1) | 1; }   // odd row stride

// In-place Stockham stage over NL lines of length N: each thread reads its butterflies (through
// src, global memory for the first stage) into registers, then all write buf.  rt[m] =
// exp(-2 pi i m / N); the inverse is conj(DFT(conj(x))).
template <int R, int N, int Ns, int NL, bool INV, int T, int SH, class Src>
__device__ __forceinline__ void cstage(Src src, double2 *buf, const double2 *rt)
{
    constexpr int LD = ldof<SH>(N), nb = N / R, tw = N / (Ns * R), NB = NL * nb;
    constexpr int IT = (NB + T - 1) / T;
    double2 v[IT][R];
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int t = threadIdx.x + it * T;
        if (t < NB) {
            const int l = t / nb, j = t - l * nb, k = j % Ns;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                v[it][r] = src(l, j + r * nb);
                if (Ns > 1 && r) v[it][r] = cmul(v[it][r], conjif(rt[r * k * tw], INV));
                if (INV) v[it][r].y = -v[it][r].y;
            }
            Dft<R>::run(v[it]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int t = threadIdx.x + it * T;
        if (t < NB) {
            const int l = t / nb, j = t - l * nb, k = j % Ns;
            const int o = (j / Ns) * Ns * R + k;
#pragma unroll
            for (int q = 0; q < R; ++q) {
                double2 u = v[it][q];
                if (INV) u.y = -u.y;
                buf[l * LD + padi<SH>(o + q * Ns)] = u;
            }
        }
    }
    __syncthreads();
}
template <int N, int Ns, int NL, bool INV, int T, int SH>
struct Fft {
    template <class Src>
    static __device__ __forceinline__ void run(Src src, double2 *buf, const double2 *rt)
    {
        if constexpr (Ns < N) {
            constexpr int R = pick(N / Ns);
            cstage<R, N, Ns, NL, INV, T, SH>(src, buf, rt);
            Fft<N, Ns * R, NL, INV, T, SH>::run(
                [buf](int l, int i) { return buf[l * ldof<SH>(N) + padi<SH>(i)]; }, buf, rt);
        }
    }
};
template <int N, int NL, bool INV, int T, int SH, class Src>
__device__ __forceinline__ void line_fft(Src src, double2 *buf, const double2 *rt)
{ Fft<N, 1, NL, INV, T, SH>::run(src, buf, rt); }

// Out-of-place stage a -> b (pass 5, rows of stride N + 1): no butterfly is held across a
// barrier, which keeps the pair loop of pass 5 at ~64 registers.
template <int R, int N, int Ns, int NL, bool INV>
__device__ __forceinline__ void pstage(const double2 *a, double2 *b, const double2 *rt)
{
    constexpr int LD = N + 1, nb = N / R, tw = N / (Ns * R);
    for (int t = threadIdx.x; t < NL * nb; t += blockDim.x) {
        const int l = t / nb, j = t - l * nb, k = j % Ns;
        double2 v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            v[r] = a[l * LD + j + r * nb];
            if (Ns > 1 && r) v[r] = cmul(v[r], conjif(rt[r * k * tw], INV));
            if (INV) v[r].y = -v[r].y;
        }
        Dft<R>::run(v);
        const int o = l * LD + (j / Ns) * Ns * R + k;
#pragma unroll
        for (int q = 0; q < R; ++q) {
            if (INV) v[q].y = -v[q].y;
            b[o + q * Ns] = v[q];
        }
    }
}
template <int N, int Ns, int NL, bool INV>
struct FftPP {
    static __device__ __forceinline__ double2 *run(double2 *a, double2 *b, const double2 *rt)
    {
        if constexpr (Ns == N) {
            return a;
        } else {
            constexpr int R = pick_pp(N / Ns);
            pstage<R, N, Ns, NL, INV>(a, b, rt);
            __syncthreads();
            return FftPP<N, Ns * R, NL, INV>::run(b, a, rt);
        }
    }
};

__device__ __forceinline__ void load_roots(double2 *rt, const double2 *roots, int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) rt[i] = roots[i];
}

// 1: codensity and forward R2C along z, two y lines per complex FFT, [x,y,z] -> X[kz,x,y]
template <int TL, int T, int SH, int MB>
__global__ void __launch_bounds__(T, MB) pass1(const double *A, const double *B,
    const int *aidx, const int *bidx, long long P0, double2 *X, const double2 *roots)
{
    extern __shared__ double2 sh[];
    constexpr int ld = ldof<SH>(nz), nyt = (ny + TL - 1) / TL;
    double2 *rt = sh, *buf = sh + nz;
    long long blk = blockIdx.x;
    const int y0 = (int)(blk % nyt) * TL; blk /= nyt;
    const int x = (int)(blk % nx);
    const long long p = blk / nx;
    load_roots(rt, roots, nz);
    const long long off = ((long long)x * ny + y0) * nz;
    const double *Ar = A + aidx[P0 + p] * ng + off;
    const double *Br = B + bidx[P0 + p] * ng + off;
    line_fft<nz, TL / 2, false, T, SH>([=](int c, int z) {
        const int l = 2 * c;
        double2 v;
        v.x = (y0 + l < ny) ? Ar[l * nz + z] * Br[l * nz + z] : 0.0;
        v.y = (y0 + l + 1 < ny) ? Ar[(l + 1) * nz + z] * Br[(l + 1) * nz + z] : 0.0;
        return v;
    }, buf, rt);
    // Z = F[a + i b]: F[a]_k = (Z_k + conj Z_-k) / 2, F[b]_k = (Z_k - conj Z_-k) / 2i
    for (int i = threadIdx.x; i < TL * nkz; i += T) {
        const int l = i % TL, k = i / TL;
        if (y0 + l >= ny) continue;
        const double2 Z = buf[(l >> 1) * ld + padi<SH>(k)];
        const double2 Zc = buf[(l >> 1) * ld + padi<SH>(k ? nz - k : 0)];
        double2 o;
        if (l & 1) { o.x = 0.5 * (Z.y + Zc.y); o.y = -0.5 * (Z.x - Zc.x); }
        else       { o.x = 0.5 * (Z.x + Zc.x); o.y = 0.5 * (Z.y - Zc.y); }
        X[((p * nkz + k) * nx + x) * (long long)ny + y0 + l] = o;
    }
}

// 2: forward along y, X[kz,x,y] -> Y[ky,kz,x]
template <int TL, int T, int SH>
__global__ void __launch_bounds__(T) pass2(const double2 *X, double2 *Y, const double2 *roots)
{
    extern __shared__ double2 sh[];
    constexpr int ld = ldof<SH>(ny), nxt = (nx + TL - 1) / TL;
    double2 *rt = sh, *buf = sh + ny;
    long long blk = blockIdx.x;
    const int x0 = (int)(blk % nxt) * TL; blk /= nxt;
    const int kz = (int)(blk % nkz);
    const long long p = blk / nkz;
    load_roots(rt, roots, ny);
    const double2 *Xr = X + ((p * nkz + kz) * nx + x0) * (long long)ny;
    line_fft<ny, TL, false, T, SH>([=](int l, int y) {
        return (x0 + l < nx) ? Xr[l * ny + y] : zero2();
    }, buf, rt);
    for (int i = threadIdx.x; i < TL * ny; i += T) {
        const int l = i % TL, ky = i / TL;
        if (x0 + l < nx)
            Y[((p * ny + ky) * nkz + kz) * (long long)nx + x0 + l] = buf[l * ld + padi<SH>(ky)];
    }
}

// 3: forward along x, times wT[kz,ky,kx] (1/ngrid included), inverse along x,
// X[ky,kz,x] -> Y[kz,x,ky]
template <int TL, int T, int SH>
__global__ void __launch_bounds__(T) pass3(const double2 *X, double2 *Y, const double *wT,
    const double2 *roots)
{
    extern __shared__ double2 sh[];
    constexpr int ld = ldof<SH>(nx), nyt = (ny + TL - 1) / TL;
    double2 *rt = sh, *buf = sh + nx;
    long long blk = blockIdx.x;
    const int ky0 = (int)(blk % nyt) * TL; blk /= nyt;
    const int kz = (int)(blk % nkz);
    const long long p = blk / nkz;
    load_roots(rt, roots, nx);
    line_fft<nx, TL, false, T, SH>([=](int l, int x) {
        return (ky0 + l < ny) ? X[((p * ny + ky0 + l) * nkz + kz) * (long long)nx + x] : zero2();
    }, buf, rt);
    const double *wr = wT + ((long long)kz * ny + ky0) * nx;
    for (int i = threadIdx.x; i < TL * nx; i += T) {
        const int l = i / nx, kx = i - l * nx;
        if (ky0 + l < ny) {
            const double s = wr[i];
            buf[l * ld + padi<SH>(kx)].x *= s;
            buf[l * ld + padi<SH>(kx)].y *= s;
        }
    }
    __syncthreads();
    line_fft<nx, TL, true, T, SH>([buf](int l, int i) { return buf[l * ld + padi<SH>(i)]; },
                                  buf, rt);
    for (int i = threadIdx.x; i < TL * nx; i += T) {
        const int l = i % TL, x = i / TL;
        if (ky0 + l < ny)
            Y[((p * nkz + kz) * nx + x) * (long long)ny + ky0 + l] = buf[l * ld + padi<SH>(x)];
    }
}

// 4: inverse along y, X[kz,x,ky] -> Y[x,y,kz]
template <int TL, int T, int SH>
__global__ void __launch_bounds__(T) pass4(const double2 *X, double2 *Y, const double2 *roots)
{
    extern __shared__ double2 sh[];
    constexpr int ld = ldof<SH>(ny), nkt = (nkz + TL - 1) / TL;
    double2 *rt = sh, *buf = sh + ny;
    long long blk = blockIdx.x;
    const int kz0 = (int)(blk % nkt) * TL; blk /= nkt;
    const int x = (int)(blk % nx);
    const long long p = blk / nx;
    load_roots(rt, roots, ny);
    line_fft<ny, TL, true, T, SH>([=](int l, int ky) {
        return (kz0 + l < nkz) ? X[((p * nkz + kz0 + l) * nx + x) * (long long)ny + ky] : zero2();
    }, buf, rt);
    for (int i = threadIdx.x; i < TL * ny; i += T) {
        const int l = i % TL, y = i / TL;
        if (kz0 + l < nkz)
            Y[((p * nx + x) * ny + y) * (long long)nkz + kz0 + l] = buf[l * ld + padi<SH>(y)];
    }
}

// 5: inverse C2R along z, two y lines per complex FFT (Z = X_a + i X_b over the full length,
// X[nz - m] = conj X[m]), W[rows[t]] += sum_k scale[src[k]] S[src[k]] V[pair[k]], X[x,y,kz] -> W
template <int TL>
__global__ void pass5(const double2 *X, const double *S, const double *scale, const int *rows,
    const int *ptr, const int *pair, const int *src, double *W, const double2 *roots)
{
    extern __shared__ double2 sh[];
    constexpr int ld = nz + 1, nl = TL / 2, nyt = (ny + TL - 1) / TL;
    double2 *rt = sh, *a = sh + nz, *b = a + nl * ld;
    double *acc = (double *)(b + nl * ld);
    long long blk = blockIdx.x;
    const int y0 = (int)(blk % nyt) * TL; blk /= nyt;
    const int x = (int)(blk % nx);
    const long long t = blk / nx;
    load_roots(rt, roots, nz);
    for (int i = threadIdx.x; i < TL * nz; i += blockDim.x) acc[i] = 0.0;
    const long long off = ((long long)x * ny + y0) * nz;
    for (int k = ptr[t]; k < ptr[t + 1]; ++k) {
        const double2 *Xr = X + (((long long)pair[k] * nx + x) * ny + y0) * nkz;
        for (int i = threadIdx.x; i < nl * nz; i += blockDim.x) {
            const int c = i / nz, m = i - c * nz;
            const int mm = m < nkz ? m : nz - m;
            double2 xa = zero2(), xb = zero2();
            if (y0 + 2 * c < ny) xa = Xr[(2 * c) * nkz + mm];
            if (y0 + 2 * c + 1 < ny) xb = Xr[(2 * c + 1) * nkz + mm];
            if (m >= nkz) { xa.y = -xa.y; xb.y = -xb.y; }
            a[c * ld + m] = make_double2(xa.x - xb.y, xa.y + xb.x);
        }
        __syncthreads();
        const double2 *r = FftPP<nz, 1, nl, true>::run(a, b, rt);
        const double sc = scale[src[k]];
        const double *Sr = S + src[k] * ng + off;
        for (int i = threadIdx.x; i < TL * nz; i += blockDim.x) {
            const int l = i / nz, z = i - l * nz;
            if (y0 + l < ny) {
                const double2 v = r[(l >> 1) * ld + z];
                acc[i] += sc * Sr[i] * ((l & 1) ? v.y : v.x);
            }
        }
        __syncthreads();
    }
    double *Wr = W + rows[t] * ng + off;
    for (int i = threadIdx.x; i < TL * nz; i += blockDim.x)
        if (y0 + i / nz < ny) Wr[i] += acc[i];
}
'''.replace('TWIDDLE_TABLES', _twiddle_tables())


def _names():
    (l1, t1, s1), (l2, t2, s2), (l3, t3, s3), (l4, t4, s4), (l5, _, _) = _SHAPES
    return (f'pass1<{l1}, {t1}, {s1}, {_PASS1_MIN_BLOCKS}>', f'pass2<{l2}, {t2}, {s2}>',
            f'pass3<{l3}, {t3}, {s3}>', f'pass4<{l4}, {t4}, {s4}>', f'pass5<{l5}>')


def _shared_bytes(mesh):
    '''Dynamic shared memory of each pass: the roots, the line buffers, pass 5's W rows.'''
    nx, ny, nz = mesh

    def ld(n, sh):
        i = n - 1
        return ((i + (i >> sh) if sh else i) + 1) | 1

    (l1, _, s1), (l2, _, s2), (l3, _, s3), (l4, _, s4), (l5, _, _) = _SHAPES
    return (16 * (nz + l1 // 2 * ld(nz, s1)), 16 * (ny + l2 * ld(ny, s2)),
            16 * (nx + l3 * ld(nx, s3)), 16 * (ny + l4 * ld(ny, s4)),
            16 * (nz + 2 * (l5 // 2) * (nz + 1)) + 8 * l5 * nz)


def applies(mesh):
    '''True if every side of ``mesh`` factors into 2, 3, 5 and 7 and every pass fits the
    current device's shared memory.'''
    for n in mesh:
        n = int(n)
        for p in _SMOOTH:
            while n % p == 0:
                n //= p
        if n != 1:
            return False
    limit = cp.cuda.Device().attributes['MaxSharedMemoryPerBlockOptin']
    return max(_shared_bytes(tuple(int(n) for n in mesh))) <= limit


def _kernels(mesh):
    '''The five kernels of ``mesh`` on the current device, compiled once.'''
    key = (mesh, cp.cuda.Device().id)
    with _LOCK:
        if key not in _KERNELS:
            names = _names()
            mod = cp.RawModule(code=_SRC, options=(
                '-std=c++17', f'-DNX={mesh[0]}', f'-DNY={mesh[1]}', f'-DNZ={mesh[2]}'),
                name_expressions=names)
            kernels = [mod.get_function(n) for n in names]
            for k, nbytes in zip(kernels, _shared_bytes(mesh)):
                if nbytes > 48 * 1024:
                    k.max_dynamic_shared_size_bytes = nbytes
            _KERNELS[key] = kernels
        return _KERNELS[key]


class FFTChain:
    '''The five passes on ``mesh`` with the half kernel ``w_half`` (nx, ny, nz//2 + 1) and
    device buffers for ``npair`` codensities: 2 x 16 (nz//2 + 1) / nz bytes per pair and point.'''

    def __init__(self, mesh, w_half, npair):
        self.mesh = tuple(int(n) for n in mesh)
        nx, ny, nz = self.mesh
        self.kernels = _kernels(self.mesh)
        self.shared = _shared_bytes(self.mesh)
        # [kz, ky, kx] for pass 3, with the inverse transform's 1 / ngrid
        self.wT = cp.ascontiguousarray((cp.asarray(w_half) / (nx * ny * nz)).transpose(2, 1, 0))
        self.roots = {n: cp.asarray(np.exp(-2j * np.pi * np.arange(n) / n)) for n in set(self.mesh)}
        size = npair * (nz // 2 + 1) * nx * ny
        self.X = cp.empty(size, dtype=np.complex128)
        self.Y = cp.empty(size, dtype=np.complex128)

    def add(self, W, A, B, aidx, bidx, p0, p1, S, scale, rows, ptr, pair, src):
        '''W[rows[t]] += sum_{k in [ptr[t], ptr[t+1])} scale[src[k]] S[src[k]] V[p0 + pair[k]]
        for the codensities A[aidx[P]] B[bidx[P]], P in [p0, p1): C-contiguous float64 grids
        (n, ngrid), int32 device index arrays (rows, ptr, pair, src a CSR by target row).'''
        nx, ny, nz = self.mesh
        nkz = nz // 2 + 1
        f = p1 - p0
        (l1, t1, _), (l2, t2, _), (l3, t3, _), (l4, t4, _), (l5, t5, _) = _SHAPES
        X, Y, r = self.X, self.Y, self.roots
        launches = ((f * nx * -(-ny // l1), t1, (A, B, aidx, bidx, np.int64(p0), X, r[nz])),
                    (f * nkz * -(-nx // l2), t2, (X, Y, r[ny])),
                    (f * nkz * -(-ny // l3), t3, (Y, X, self.wT, r[nx])),
                    (f * nx * -(-nkz // l4), t4, (X, Y, r[ny])),
                    (len(rows) * nx * -(-ny // l5), t5, (Y, S, scale, rows, ptr, pair, src, W,
                                                          r[nz])))
        for kernel, nbytes, (blocks, threads, args) in zip(self.kernels, self.shared, launches):
            kernel((blocks,), (threads,), args, shared_mem=nbytes)
