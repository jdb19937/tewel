#include <stdint.h>
// Tiled implicit-GEMM convolution kernels (GPU only). Included from colonel-cuda.cu.
//
// Layouts (unchanged from the naive kernels):
//   activations  in[iz + ic * (ix + iw * (iy + ih * b))]       i.e. [b][y][x][c]
//   weights      wmv[oc + k + K * oz], k = iz + ic * (dx + fw * dy), K = ic * fw * fh
//   biases       wmv[oz]
//   gradients    dwv[same index as wmv]
// Convolution wraps around at the image border (periodic padding), as before.
// A pixel index p = ix + iw * (iy + ih * b) runs over the whole batch: P = bn * ih * iw.

#define CG_BM 64
#define CG_BN 64
#define CG_BK 16
#define CG_THREADS 256

// out[p][oz] = sum_k patch(p, k) * W[oz][k] + bias[oz]
__global__ void _gemm_conv_fwd(
  const float *in, int bn, int iw, int ih, int ic,
  float *out, int oc,
  int fw, int fh, int dxo, int dyo,
  bool relu, int freeze,
  const float *wmv, float wscale
) {
  __shared__ float As[CG_BK][CG_BM + 4];
  __shared__ float Bs[CG_BK][CG_BN + 4];

  const int P = bn * ih * iw;
  const int K = ic * fw * fh;
  const int p0 = blockIdx.x * CG_BM;
  const int n0 = blockIdx.y * CG_BN;
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;

  float acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      acc[i][j] = 0;

  for (int k0 = 0; k0 < K; k0 += CG_BK) {
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int m = e / CG_BK, kk = e % CG_BK;
      int p = p0 + m, k = k0 + kk;
      float v = 0;
      if (p < P && k < K) {
        int iz = k % ic; int t = k / ic; int dx = t % fw; int dy = t / fw;
        int ox = p % iw; int t2 = p / iw; int oy = t2 % ih; int b = t2 / ih;
        int ix = (ox - dxo + dx + iw) % iw;
        int iy = (oy - dyo + dy + ih) % ih;
        v = in[iz + ic * (ix + iw * (iy + ih * b))];
        if (relu && !(v > 0) && iz >= freeze)
          v = 0;
      }
      As[kk][m] = v;
    }
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int n = e / CG_BK, kk = e % CG_BK;
      int oz = n0 + n, k = k0 + kk;
      float v = 0;
      if (oz < oc && k < K)
        v = wmv[oc + k + K * oz] * wscale;
      Bs[kk][n] = v;
    }
    __syncthreads();

    for (int kk = 0; kk < CG_BK; ++kk) {
      float a[4], bb[4];
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
      for (int j = 0; j < 4; ++j) bb[j] = Bs[kk][tx * 4 + j];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          acc[i][j] += a[i] * bb[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    int p = p0 + ty * 4 + i;
    if (p >= P) continue;
    for (int j = 0; j < 4; ++j) {
      int oz = n0 + tx * 4 + j;
      if (oz >= oc) continue;
      float v = acc[i][j] + wmv[oz];
      if (oz < freeze)
        v = in[oz + ic * p];
      out[oz + oc * p] = v;
    }
  }
}

// fin[p][iz] = sum_{k2} fout(p shifted, oz) * W[oz][dy][dx][iz],  k2 = oz + oc * (dx + fw * dy)
// (in place: fin holds the layer input on entry, used for the relu gate)
__global__ void _gemm_conv_bwd_in(
  float *fin, int bn, int iw, int ih, int ic,
  const float *fout, int oc,
  int fw, int fh, int dxo, int dyo,
  bool relu, int freeze,
  const float *wmv, float wscale
) {
  __shared__ float As[CG_BK][CG_BM + 4];
  __shared__ float Bs[CG_BK][CG_BN + 4];

  const int P = bn * ih * iw;
  const int K = ic * fw * fh;
  const int K2 = oc * fw * fh;
  const int p0 = blockIdx.x * CG_BM;
  const int n0 = blockIdx.y * CG_BN;
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;

  float acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      acc[i][j] = 0;

  for (int k0 = 0; k0 < K2; k0 += CG_BK) {
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int m = e / CG_BK, kk = e % CG_BK;
      int p = p0 + m, k2 = k0 + kk;
      float v = 0;
      if (p < P && k2 < K2) {
        int oz = k2 % oc; int t = k2 / oc; int dx = t % fw; int dy = t / fw;
        int ix = p % iw; int t2 = p / iw; int iy = t2 % ih; int b = t2 / ih;
        int ox = (ix + dxo - dx + iw) % iw;
        int oy = (iy + dyo - dy + ih) % ih;
        v = fout[oz + oc * (ox + iw * (oy + ih * b))];
      }
      As[kk][m] = v;
    }
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int kk = e / CG_BN, n = e % CG_BN;
      int iz = n0 + n, k2 = k0 + kk;
      float v = 0;
      if (iz < ic && k2 < K2) {
        int oz = k2 % oc; int t = k2 / oc; int dx = t % fw; int dy = t / fw;
        if (oz >= freeze)
          v = wmv[oc + iz + ic * (dx + fw * (dy + fh * oz))] * wscale;
      }
      Bs[kk][n] = v;
    }
    __syncthreads();

    for (int kk = 0; kk < CG_BK; ++kk) {
      float a[4], bb[4];
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
      for (int j = 0; j < 4; ++j) bb[j] = Bs[kk][tx * 4 + j];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          acc[i][j] += a[i] * bb[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    int p = p0 + ty * 4 + i;
    if (p >= P) continue;
    for (int j = 0; j < 4; ++j) {
      int iz = n0 + tx * 4 + j;
      if (iz >= ic) continue;
      int ini = iz + ic * p;
      float v = acc[i][j];
      if (iz < freeze)
        v += fout[iz + oc * p];
      if (relu && !(fin[ini] > 0) && iz >= freeze)
        v = 0;
      fin[ini] = v;
    }
  }
}

// dW[oz][k] += sum_p fout[p][oz] * patch(p, k), reduction over pixels split across blockIdx.z
__global__ void _gemm_conv_bwd_w(
  const float *in, int bn, int iw, int ih, int ic,
  const float *fout, int oc,
  int fw, int fh, int dxo, int dyo,
  bool relu, int freeze,
  float *dwv, int pchunk, float wscale
) {
  __shared__ float As[CG_BK][CG_BM + 4];   // [p][oz]
  __shared__ float Bs[CG_BK][CG_BN + 4];   // [p][k]

  const int P = bn * ih * iw;
  const int K = ic * fw * fh;
  const int m0 = blockIdx.y * CG_BM;        // oz tile
  const int n0 = blockIdx.x * CG_BN;        // k tile
  const int pa = blockIdx.z * pchunk;
  const int pb = min(P, pa + pchunk);
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;

  float acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      acc[i][j] = 0;

  for (int q0 = pa; q0 < pb; q0 += CG_BK) {
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int kk = e / CG_BM, m = e % CG_BM;
      int p = q0 + kk, oz = m0 + m;
      float v = 0;
      if (p < pb && oz < oc)
        v = fout[oz + oc * p];
      As[kk][m] = v;
    }
    for (int j = 0; j < 4; ++j) {
      int e = tid + CG_THREADS * j;
      int kk = e / CG_BN, n = e % CG_BN;
      int p = q0 + kk, k = n0 + n;
      float v = 0;
      if (p < pb && k < K) {
        int iz = k % ic; int t = k / ic; int dx = t % fw; int dy = t / fw;
        int ox = p % iw; int t2 = p / iw; int oy = t2 % ih; int b = t2 / ih;
        int ix = (ox - dxo + dx + iw) % iw;
        int iy = (oy - dyo + dy + ih) % ih;
        v = in[iz + ic * (ix + iw * (iy + ih * b))];
        if (relu && !(v > 0) && iz >= freeze)
          v = 0;
      }
      Bs[kk][n] = v;
    }
    __syncthreads();

    for (int kk = 0; kk < CG_BK; ++kk) {
      float a[4], bb[4];
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
      for (int j = 0; j < 4; ++j) bb[j] = Bs[kk][tx * 4 + j];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          acc[i][j] += a[i] * bb[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    int oz = m0 + ty * 4 + i;
    if (oz >= oc || oz < freeze) continue;
    for (int j = 0; j < 4; ++j) {
      int k = n0 + tx * 4 + j;
      if (k >= K) continue;
      atomicAdd(&dwv[oc + k + K * oz], acc[i][j] * wscale);
    }
  }
}

// dB[oz] += sum_p fout[p][oz]: each block reduces a pixel range with coalesced reads over channels
__global__ void _gemm_conv_bwd_b(
  const float *fout, long P, int oc, int freeze, float *dwv, long pchunk
) {
  extern __shared__ float sacc[];
  const int tid = threadIdx.x, nt = blockDim.x;
  const long pa = blockIdx.x * pchunk, pb = min(P, pa + pchunk);
  for (int oz = tid; oz < oc; oz += nt) sacc[oz] = 0;
  __syncthreads();
  if (oc <= nt && nt % oc == 0) {
    int oz = tid % oc, lane = tid / oc, nl = nt / oc;
    float s = 0;
    for (long p = pa + lane; p < pb; p += nl) s += fout[oz + oc * p];
    atomicAdd(&sacc[oz], s);
  } else {
    for (long e = pa * oc + tid; e < pb * oc; e += nt) atomicAdd(&sacc[e % oc], fout[e]);
  }
  __syncthreads();
  for (int oz = tid; oz < oc; oz += nt)
    if (oz >= freeze) atomicAdd(&dwv[oz], sacc[oz]);
}
static void launch_conv_bwd_b(const float *fout, long P, int oc, int freeze, float *dwv) {
  long pchunk = (P + 511) / 512; if (pchunk < 64) pchunk = 64;
  pchunk = (pchunk + 31) / 32 * 32;
  unsigned blocks = (unsigned)((P + pchunk - 1) / pchunk);
  _gemm_conv_bwd_b<<<blocks, 256, oc * sizeof(float)>>>(fout, P, oc, freeze, dwv, pchunk);
}

static inline void _conv_geom(int d, int dim, int *fw, int *fh, int *dxo, int *dyo) {
  int f = 2 * d + 1;
  *fw = f;
  *fh = (dim == 1) ? 1 : f;
  *dxo = d;
  *dyo = (dim == 1) ? 0 : d;
}

void gpu_synth_conv(
  const float *in, int bn, int iw, int ih, float *out,
  int d, int ic, int oc, bool relu, int freeze, int dim,
  const float *wmv, float wscale
) {
  int fw, fh, dxo, dyo;
  _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw;
  dim3 grid((P + CG_BM - 1) / CG_BM, (oc + CG_BN - 1) / CG_BN);
  _gemm_conv_fwd<<<grid, CG_THREADS>>>(in, bn, iw, ih, ic, out, oc, fw, fh, dxo, dyo, relu, freeze, wmv, wscale);
}

void gpu_learn_conv_w(
  const float *in, int bn, int iw, int ih, const float *fout,
  int d, int ic, int oc, bool relu, int freeze, int dim,
  float *dwv, float wscale
) {
  int fw, fh, dxo, dyo;
  _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw;
  int K = ic * fw * fh;
  int tiles = ((K + CG_BN - 1) / CG_BN) * ((oc + CG_BM - 1) / CG_BM);
  int splits = (int)((P + 1023) / 1024);
  if (splits * tiles > 4096) splits = (4096 + tiles - 1) / tiles;
  if (splits < 1) splits = 1;
  int pchunk = (int)((P + splits - 1) / splits);
  pchunk = (pchunk + CG_BK - 1) / CG_BK * CG_BK;
  splits = (int)((P + pchunk - 1) / pchunk);
  dim3 grid((K + CG_BN - 1) / CG_BN, (oc + CG_BM - 1) / CG_BM, splits);
  _gemm_conv_bwd_w<<<grid, CG_THREADS>>>(in, bn, iw, ih, ic, fout, oc, fw, fh, dxo, dyo, relu, freeze, dwv, pchunk, wscale);
  launch_conv_bwd_b(fout, P, oc, freeze, dwv);
}

void gpu_learn_conv_in(
  float *fin, int bn, int iw, int ih, const float *fout,
  int d, int ic, int oc, bool relu, int freeze, int dim,
  const float *wmv, float wscale
) {
  int fw, fh, dxo, dyo;
  _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw;
  dim3 grid((P + CG_BM - 1) / CG_BM, (ic + CG_BN - 1) / CG_BN);
  _gemm_conv_bwd_in<<<grid, CG_THREADS>>>(fin, bn, iw, ih, ic, fout, oc, fw, fh, dxo, dyo, relu, freeze, wmv, wscale);
}


// pixel norm backward: one warp per pixel, lanes stride over channels
__global__ void _warp_learn_pnrm(float *fin, int pn, int ic, const float *fout) {
  int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (gw >= pn)
    return;
  float *x = fin + (size_t)gw * ic;
  const float *g = fout + (size_t)gw * ic;
  float s = 0, dot = 0;
  for (int z = lane; z < ic; z += 32) {
    float xv = x[z];
    s += xv * xv;
    dot += g[z] * xv;
  }
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
    dot += __shfl_xor_sync(0xffffffff, dot, o);
  }
  s = sqrtf(s / (float)ic + 1e-8f);
  dot /= ((float)ic * s * s);
  for (int z = lane; z < ic; z += 32)
    x[z] = (g[z] - x[z] * dot) / s;
}

void gpu_learn_pnrm(float *fin, int pn, int ic, const float *fout) {
  long threads = (long)pn * 32;
  _warp_learn_pnrm<<<(threads + 255) / 256, 256>>>(fin, pn, ic, fout);
}

// ======================= tensor-core (WMMA, fp16 in / fp32 accumulate) =======================
// (mma.h is included from colonel-cuda.cu, outside the namespace)
using namespace nvcuda;

// Tensor-core implicit-GEMM conv kernels (fp16 tiles, fp32 accumulate), templated on the block
// tile BM x BN (K step 32) and the warp grid WM x WN; each warp owns a (BM/WM) x (BN/WN) sub-tile
// of 16x16 fragments, warps beyond WM*WN only help with the loads.  As: M x K row-major [BM][32];
// Bs: K x N stored [n][k] [BN][32]; both half, row stride CW_LD to dodge bank conflicts.
// Small tiles keep the narrow layers (3..64 channels) and the small-spatial layers busy; split-K
// (part != NULL) spreads a short grid over more blocks, partial sums are atomically added to part
// and a tiny epilogue kernel finishes.
#define CW_BK 32
#define CW_THREADS 256
#define CW_LD 40

template <int BM, int BN> struct CwSmem {
  __align__(32) half As[BM * CW_LD];
  __align__(32) half Bs[BN * CW_LD];
  __align__(32) float Cs[8][16 * 16];   // per-warp staging for the epilogue
};

template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void cw_mma(CwSmem<BM, BN> &sm,
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&acc)[BM / WM / 16][BN / WN / 16], int wm, int wn, bool active) {
  constexpr int FM = BM / WM / 16, FN = BN / WN / 16;
  __syncthreads();
  if (active) {
    for (int kk = 0; kk < CW_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[FN];
      for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(a[i], sm.As + (wm * (BM / WM) + i * 16) * CW_LD + kk, CW_LD);
      for (int j = 0; j < FN; ++j) wmma::load_matrix_sync(b[j], sm.Bs + (wn * (BN / WN) + j * 16) * CW_LD + kk, CW_LD);
      for (int i = 0; i < FM; ++i) for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void st_half4(half *p, float4 v) {
  half2 *q = (half2 *)p;
  q[0] = __floats2half2_rn(v.x, v.y);
  q[1] = __floats2half2_rn(v.z, v.w);
}

__device__ __forceinline__ float4 relu4(float4 v, int iz, int freeze) {
  if (!(v.x > 0) && iz + 0 >= freeze) v.x = 0;
  if (!(v.y > 0) && iz + 1 >= freeze) v.y = 0;
  if (!(v.z > 0) && iz + 2 >= freeze) v.z = 0;
  if (!(v.w > 0) && iz + 3 >= freeze) v.w = 0;
  return v;
}

// patch(p, k) for an activation tensor [b][y][x][c] with wrap-around
__device__ __forceinline__ float gather1(const float *in, int iw, int ih, int ic, int fw, int dxo, int dyo, int p, int k, bool relu, int freeze) {
  int iz = k % ic; int t = k / ic; int dx = t % fw; int dy = t / fw;
  int ox = p % iw; int t2 = p / iw; int oy = t2 % ih; int b = t2 / ih;
  int ix = (ox - dxo + dx + iw) % iw, iy = (oy - dyo + dy + ih) % ih;
  float v = in[iz + ic * (ix + iw * (iy + ih * b))];
  if (relu && !(v > 0) && iz >= freeze) v = 0;
  return v;
}

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(CW_THREADS) _wmma_conv_fwd(
  const float *in, int bn, int iw, int ih, int ic, float *out, int oc,
  int fw, int fh, int dxo, int dyo, bool relu, int freeze, const float *wmv, float wscale,
  int kchunk, float *part
) {
  constexpr int FM = BM / WM / 16, FN = BN / WN / 16;
  __shared__ CwSmem<BM, BN> sm;
  const int P = bn * ih * iw, K = ic * fw * fh;
  const int p0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int ka = part ? blockIdx.z * kchunk : 0;
  const int kb = part ? min(K, ka + kchunk) : K;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const bool active = warp < WM * WN;
  const bool fast = (ic % 16 == 0) && ((((uintptr_t)in) & 15) == 0);
  // loads: thread t < 2*BM fills row t % BM, k slab (t / BM) * 16 of As; likewise Bs with BN
  const int am = tid % BM, ak = (tid / BM) * 16; const bool a_ld = tid < 2 * BM;
  const int bm = tid % BN, bk = (tid / BN) * 16; const bool b_ld = tid < 2 * BN;
  int a_ox = 0, a_oy = 0, a_b = 0; const bool a_ok = a_ld && (p0 + am < P);
  if (a_ok) { int p = p0 + am; a_ox = p % iw; int t2 = p / iw; a_oy = t2 % ih; a_b = t2 / ih; }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i) for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int k0 = ka; k0 < kb; k0 += CW_BK) {
    if (a_ld) {
      int k = k0 + ak;
      int iz = k % ic; int t = k / ic; int dx = t % fw; int dy = t / fw;
      if (a_ok && k + 15 < kb && iz + 16 <= ic) {
        // the 16-slab lies inside one pixel: contiguous channels
        int ix = (a_ox - dxo + dx + iw) % iw, iy = (a_oy - dyo + dy + ih) % ih;
        const float *src = in + iz + ic * (ix + iw * (iy + ih * a_b));
        if (fast) {
          const float4 *src4 = (const float4 *)src;
          for (int j = 0; j < 4; ++j) {
            float4 v = src4[j];
            if (relu) v = relu4(v, iz + 4 * j, freeze);
            st_half4(sm.As + am * CW_LD + ak + 4 * j, v);
          }
        } else {
          for (int j = 0; j < 16; ++j) {
            float v = src[j];
            if (relu && !(v > 0) && iz + j >= freeze) v = 0;
            sm.As[am * CW_LD + ak + j] = __float2half(v);
          }
        }
      } else {
        for (int j = 0; j < 16; ++j) {
          int kk = k + j; float v = 0;
          if (a_ok && kk < kb) v = gather1(in, iw, ih, ic, fw, dxo, dyo, p0 + am, kk, relu, freeze);
          sm.As[am * CW_LD + ak + j] = __float2half(v);
        }
      }
    }
    if (b_ld) {
      int oz = n0 + bm, k = k0 + bk;
      const float *w = wmv + oc + (long)K * oz + k;
      if (oz < oc && k + 15 < kb && ((((uintptr_t)w) & 15) == 0)) {
        const float4 *src = (const float4 *)w;
        for (int j = 0; j < 4; ++j) { float4 v = src[j]; v.x *= wscale; v.y *= wscale; v.z *= wscale; v.w *= wscale; st_half4(sm.Bs + bm * CW_LD + bk + 4 * j, v); }
      } else {
        for (int j = 0; j < 16; ++j) { float v = 0; if (oz < oc && k + j < kb) v = w[j] * wscale; sm.Bs[bm * CW_LD + bk + j] = __float2half(v); }
      }
    }
    cw_mma<BM, BN, WM, WN>(sm, acc, wm, wn, active);
  }
  if (!active) return;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(sm.Cs[warp], acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int mb = p0 + wm * (BM / WM) + i * 16, nb = n0 + wn * (BN / WN) + j * 16;
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4, c = e & 15;
        int p = mb + r, oz = nb + c;
        if (p < P && oz < oc) {
          if (part) {
            atomicAdd(&part[oz + oc * (long)p], sm.Cs[warp][e]);
          } else {
            float v = sm.Cs[warp][e] + wmv[oz];
            if (oz < freeze) v = in[oz + ic * (long)p];
            out[oz + oc * (long)p] = v;
          }
        }
      }
      __syncwarp();
    }
}

// split-K epilogue: out = part + bias (frozen channels copy the input)
__global__ void _conv_fwd_fin(const float *part, long n, int oc, int ic, const float *in, const float *wmv, int freeze, float *out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int oz = i % oc; long p = i / oc;
  out[i] = (oz < freeze) ? in[oz + ic * p] : part[i] + wmv[oz];
}

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(CW_THREADS) _wmma_conv_bwd_in(
  float *fin, int bn, int iw, int ih, int ic, const float *fout, int oc,
  int fw, int fh, int dxo, int dyo, bool relu, int freeze, const float *wt,
  int kchunk, float *part
) {
  // wt: transposed, scaled weights [iz][k2], k2 = oz + oc * (dx + fw * dy); zero for oz < freeze
  constexpr int FM = BM / WM / 16, FN = BN / WN / 16;
  __shared__ CwSmem<BM, BN> sm;
  const int P = bn * ih * iw, K2 = oc * fw * fh;
  const int p0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int ka = part ? blockIdx.z * kchunk : 0;
  const int kb = part ? min(K2, ka + kchunk) : K2;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const bool active = warp < WM * WN;
  const bool fast = (oc % 16 == 0) && ((((uintptr_t)fout) & 15) == 0);
  const int am = tid % BM, ak = (tid / BM) * 16; const bool a_ld = tid < 2 * BM;
  const int bm = tid % BN, bk = (tid / BN) * 16; const bool b_ld = tid < 2 * BN;
  int a_ix = 0, a_iy = 0, a_b = 0; const bool a_ok = a_ld && (p0 + am < P);
  if (a_ok) { int p = p0 + am; a_ix = p % iw; int t2 = p / iw; a_iy = t2 % ih; a_b = t2 / ih; }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i) for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int k0 = ka; k0 < kb; k0 += CW_BK) {
    if (a_ld) {
      int k2 = k0 + ak;
      int oz = k2 % oc; int t = k2 / oc; int dx = t % fw; int dy = t / fw;
      if (a_ok && k2 + 15 < kb && oz + 16 <= oc) {
        int ox = (a_ix + dxo - dx + iw) % iw, oy = (a_iy + dyo - dy + ih) % ih;
        const float *src = fout + oz + oc * (ox + iw * (oy + ih * a_b));
        if (fast) { const float4 *src4 = (const float4 *)src; for (int j = 0; j < 4; ++j) st_half4(sm.As + am * CW_LD + ak + 4 * j, src4[j]); }
        else { for (int j = 0; j < 16; ++j) sm.As[am * CW_LD + ak + j] = __float2half(src[j]); }
      } else {
        for (int j = 0; j < 16; ++j) {
          int kk = k2 + j; float v = 0;
          if (a_ok && kk < kb) {
            int oz = kk % oc; int t = kk / oc; int dx = t % fw; int dy = t / fw;
            int ox = (a_ix + dxo - dx + iw) % iw, oy = (a_iy + dyo - dy + ih) % ih;
            v = fout[oz + oc * (ox + iw * (oy + ih * a_b))];
          }
          sm.As[am * CW_LD + ak + j] = __float2half(v);
        }
      }
    }
    if (b_ld) {
      int iz = n0 + bm, k2 = k0 + bk;
      const float *w = wt + (long)K2 * iz + k2;
      if (iz < ic && k2 + 15 < kb && ((((uintptr_t)w) & 15) == 0)) {
        const float4 *src = (const float4 *)w;
        for (int j = 0; j < 4; ++j) st_half4(sm.Bs + bm * CW_LD + bk + 4 * j, src[j]);
      } else {
        for (int j = 0; j < 16; ++j) { float v = 0; if (iz < ic && k2 + j < kb) v = w[j]; sm.Bs[bm * CW_LD + bk + j] = __float2half(v); }
      }
    }
    cw_mma<BM, BN, WM, WN>(sm, acc, wm, wn, active);
  }
  if (!active) return;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(sm.Cs[warp], acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int mb = p0 + wm * (BM / WM) + i * 16, nb = n0 + wn * (BN / WN) + j * 16;
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4, c = e & 15;
        int p = mb + r, iz = nb + c;
        if (p < P && iz < ic) {
          long ini = iz + ic * (long)p;
          if (part) {
            atomicAdd(&part[ini], sm.Cs[warp][e]);
          } else {
            float v = sm.Cs[warp][e];
            if (iz < freeze) v += fout[iz + oc * (long)p];
            if (relu && !(fin[ini] > 0) && iz >= freeze) v = 0;
            fin[ini] = v;
          }
        }
      }
      __syncwarp();
    }
}

// split-K epilogue for the input gradient: relu mask from the (still intact) input, frozen passthrough
__global__ void _conv_bwd_in_fin(const float *part, long n, int ic, int oc, const float *fout, bool relu, int freeze, float *fin) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int iz = i % ic; long p = i / ic;
  float v = part[i];
  if (iz < freeze) v += fout[iz + oc * p];
  if (relu && !(fin[i] > 0) && iz >= freeze) v = 0;
  fin[i] = v;
}

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(CW_THREADS) _wmma_conv_bwd_w(
  const float *in, int bn, int iw, int ih, int ic, const float *fout, int oc,
  int fw, int fh, int dxo, int dyo, bool relu, int freeze, float *dwv, int pchunk, float wscale
) {
  // dW[oz][k] += sum_p fout[p][oz] * patch(p, k): M = oc, N = K, reduction over pixels split across blockIdx.z
  constexpr int FM = BM / WM / 16, FN = BN / WN / 16;
  __shared__ CwSmem<BM, BN> sm;
  const int P = bn * ih * iw, K = ic * fw * fh;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int pa = blockIdx.z * pchunk, pb = min(P, pa + pchunk);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp % WN;
  const bool active = warp < WM * WN;
  const bool fast = (ic % 16 == 0) && ((((uintptr_t)in) & 15) == 0);
  const bool fout_al = (oc % 16 == 0) && ((((uintptr_t)fout) & 15) == 0);
  // thread owns pixel column ak = tid & 31 and the 16-row slab am = (tid >> 5) * 16 of As (rows < BM) and Bs (rows < BN)
  const int ak = tid & 31, am = (tid >> 5) * 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i) for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int q0 = pa; q0 < pb; q0 += CW_BK) {
    int p = q0 + ak; bool pok = (p < pb);
    int ox = 0, oy = 0, b = 0;
    if (pok) { ox = p % iw; int t2 = p / iw; oy = t2 % ih; b = t2 / ih; }
    if (am < BM) {
      int oz0 = m0 + am;
      if (pok && fout_al && oz0 + 15 < oc) {
        const float4 *src = (const float4 *)(fout + oz0 + oc * (long)p);
        for (int j = 0; j < 4; ++j) { float4 v = src[j]; sm.As[(am + 4*j) * CW_LD + ak] = __float2half(v.x); sm.As[(am + 4*j + 1) * CW_LD + ak] = __float2half(v.y); sm.As[(am + 4*j + 2) * CW_LD + ak] = __float2half(v.z); sm.As[(am + 4*j + 3) * CW_LD + ak] = __float2half(v.w); }
      } else {
        for (int j = 0; j < 16; ++j) { float v = 0; if (pok && oz0 + j < oc) v = fout[oz0 + j + oc * (long)p]; sm.As[(am + j) * CW_LD + ak] = __float2half(v); }
      }
    }
    if (am < BN) {
      int k = n0 + am;
      int iz = k % ic; int t = k / ic; int dx = t % fw; int dy = t / fw;
      if (pok && k + 15 < K && iz + 16 <= ic) {
        int ix = (ox - dxo + dx + iw) % iw, iy = (oy - dyo + dy + ih) % ih;
        const float *src = in + iz + ic * (ix + iw * (iy + ih * b));
        if (fast) {
          const float4 *src4 = (const float4 *)src;
          for (int j = 0; j < 4; ++j) {
            float4 v = src4[j];
            if (relu) v = relu4(v, iz + 4 * j, freeze);
            sm.Bs[(am + 4*j) * CW_LD + ak] = __float2half(v.x); sm.Bs[(am + 4*j + 1) * CW_LD + ak] = __float2half(v.y); sm.Bs[(am + 4*j + 2) * CW_LD + ak] = __float2half(v.z); sm.Bs[(am + 4*j + 3) * CW_LD + ak] = __float2half(v.w);
          }
        } else {
          for (int j = 0; j < 16; ++j) { float v = src[j]; if (relu && !(v > 0) && iz + j >= freeze) v = 0; sm.Bs[(am + j) * CW_LD + ak] = __float2half(v); }
        }
      } else {
        for (int j = 0; j < 16; ++j) { float v = 0; if (pok && k + j < K) v = gather1(in, iw, ih, ic, fw, dxo, dyo, p, k + j, relu, freeze); sm.Bs[(am + j) * CW_LD + ak] = __float2half(v); }
      }
    }
    cw_mma<BM, BN, WM, WN>(sm, acc, wm, wn, active);
  }
  if (!active) return;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(sm.Cs[warp], acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int mb = m0 + wm * (BM / WM) + i * 16, nb = n0 + wn * (BN / WN) + j * 16;
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4, c = e & 15;
        int oz = mb + r, k = nb + c;
        if (oz < oc && oz >= freeze && k < K)
          atomicAdd(&dwv[oc + k + (long)K * oz], sm.Cs[warp][e] * wscale);
      }
      __syncwarp();
    }
}

// wt[iz][k2] = wscale * W[oz][dy][dx][iz] (0 for frozen output channels), k2 = oz + oc * (dx + fw * dy)
__global__ void _conv_wt(const float *wmv, int ic, int oc, int fw, int fh, int freeze, float wscale, float *wt) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long K2 = (long)oc * fw * fh;
  if (i >= (long)ic * K2) return;
  int iz = i / K2; long k2 = i % K2;
  int oz = k2 % oc; int t = k2 / oc; int dx = t % fw; int dy = t / fw;
  wt[i] = (oz < freeze) ? 0.0f : wmv[oc + iz + ic * (dx + fw * (dy + fh * oz))] * wscale;
}

int kprec = 32;   // 32: fp32 tiles; 16: tensor cores (fp16 tiles, fp32 accumulate)
void setkprec(int p) { kprec = p; }

// tile configurations: BM, BN, WM, WN (sub-tiles must be multiples of 16; 32x32 uses 4 of the 8 warps)
#define CW_CONFIGS(X) \
  X(128, 128, 4, 2) X(128, 64, 4, 2) X(128, 32, 8, 1) \
  X(64, 128, 2, 4) X(64, 64, 4, 2) X(64, 32, 4, 2) \
  X(32, 128, 2, 4) X(32, 64, 2, 4) X(32, 32, 2, 2)

static inline int cw_pick(long n) { return n <= 32 ? 32 : n <= 64 ? 64 : 128; }

// rows x cols output grid: aim for >= 256 blocks (a few per SM) by shrinking the row tile to 64,
// then splitting K; only shrink to 32 rows when K is too short to split
static void cw_plan(long rows, long cols, long K, int *bmp, int *bnp, int *splitsp, int *kchunkp) {
  int bnn = cw_pick(cols), bm = cw_pick(rows);
  long blocks = ((rows + bm - 1) / bm) * ((cols + bnn - 1) / bnn);
  if (blocks < 256 && bm > 64) { bm = 64; blocks = ((rows + bm - 1) / bm) * ((cols + bnn - 1) / bnn); }
  if (blocks < 256 && bm > 32 && K < 512) { bm = 32; blocks = ((rows + bm - 1) / bm) * ((cols + bnn - 1) / bnn); }
  int splits = 1, kchunk = (int)K;
  if (blocks < 256 && K >= 512) {
    splits = (int)((256 + blocks - 1) / blocks);
    if (splits > K / 256) splits = (int)(K / 256);
    if (splits < 1) splits = 1;
    kchunk = (int)((K + splits - 1) / splits);
    kchunk = (kchunk + CW_BK - 1) / CW_BK * CW_BK;
    splits = (int)((K + kchunk - 1) / kchunk);
  }
  *bmp = bm; *bnp = bnn; *splitsp = splits; *kchunkp = kchunk;
}

void gpu_synth_conv16(const float *in, int bn, int iw, int ih, float *out, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale) {
  int fw, fh, dxo, dyo; _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw, K = (long)ic * fw * fh;
  int bm, bnn, splits, kchunk;
  cw_plan(P, oc, K, &bm, &bnn, &splits, &kchunk);
  float *part = NULL;
  if (splits > 1) { kmake(&part, P * oc); kfill(part, P * oc, 0); }
  dim3 grid((P + bm - 1) / bm, (oc + bnn - 1) / bnn, splits);
#define X(BM, BN, WM, WN) if (bm == BM && bnn == BN) _wmma_conv_fwd<BM, BN, WM, WN><<<grid, CW_THREADS>>>(in, bn, iw, ih, ic, out, oc, fw, fh, dxo, dyo, relu, freeze, wmv, wscale, kchunk, part);
  CW_CONFIGS(X)
#undef X
  if (part) {
    long n = P * oc;
    _conv_fwd_fin<<<(unsigned)((n + 255) / 256), 256>>>(part, n, oc, ic, in, wmv, freeze, out);
    kfree(part);
  }
}

void gpu_learn_conv_in16(float *fin, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale) {
  int fw, fh, dxo, dyo; _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw;
  long K2 = (long)oc * fw * fh;
  float *wt; kmake(&wt, (long)ic * K2);
  _conv_wt<<<(unsigned)(((long)ic * K2 + 255) / 256), 256>>>(wmv, ic, oc, fw, fh, freeze, wscale, wt);
  int bm, bnn, splits, kchunk;
  cw_plan(P, ic, K2, &bm, &bnn, &splits, &kchunk);
  float *part = NULL;
  if (splits > 1) { kmake(&part, P * ic); kfill(part, P * ic, 0); }
  dim3 grid((P + bm - 1) / bm, (ic + bnn - 1) / bnn, splits);
#define X(BM, BN, WM, WN) if (bm == BM && bnn == BN) _wmma_conv_bwd_in<BM, BN, WM, WN><<<grid, CW_THREADS>>>(fin, bn, iw, ih, ic, fout, oc, fw, fh, dxo, dyo, relu, freeze, wt, kchunk, part);
  CW_CONFIGS(X)
#undef X
  if (part) {
    long n = P * ic;
    _conv_bwd_in_fin<<<(unsigned)((n + 255) / 256), 256>>>(part, n, ic, oc, fout, relu, freeze, fin);
    kfree(part);
  }
  kfree(wt);
}

void gpu_learn_conv_w16(const float *in, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, float *dwv, float wscale) {
  int fw, fh, dxo, dyo; _conv_geom(d, dim, &fw, &fh, &dxo, &dyo);
  long P = (long)bn * ih * iw;
  int K = ic * fw * fh;
  int bm = cw_pick(oc), bnn = cw_pick(K);
  long tiles = ((K + bnn - 1) / bnn) * ((oc + bm - 1) / bm);
  // aim for ~256 blocks, each reducing at least 256 pixels
  long splits = (256 + tiles - 1) / tiles;
  if (splits > (P + 255) / 256) splits = (P + 255) / 256;
  if (splits < 1) splits = 1;
  long pchunk = (P + splits - 1) / splits;
  pchunk = (pchunk + CW_BK - 1) / CW_BK * CW_BK;
  splits = (P + pchunk - 1) / pchunk;
  dim3 grid((K + bnn - 1) / bnn, (oc + bm - 1) / bm, (unsigned)splits);
#define X(BM, BN, WM, WN) if (bm == BM && bnn == BN) _wmma_conv_bwd_w<BM, BN, WM, WN><<<grid, CW_THREADS>>>(in, bn, iw, ih, ic, fout, oc, fw, fh, dxo, dyo, relu, freeze, dwv, (int)pchunk, wscale);
  CW_CONFIGS(X)
#undef X
  launch_conv_bwd_b(fout, P, oc, freeze, dwv);
}

// style affine (AdaIN): one warp per (sample, scale-or-shift channel), lanes stride over the style dims
__global__ void _styl_affine_warp(const float *wmv, const float *kstyle, long n, int ic, int sdim, float wscale, float *ksc) {
  long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (gw >= n) return;
  int j = gw % (2 * ic); long b = gw / (2 * ic);
  const float *s = kstyle + b * sdim, *w = wmv + 2 * ic + (long)j * sdim;
  float v = 0;
  for (int k = lane; k < sdim; k += 32) v += w[k] * s[k];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if (lane == 0) ksc[gw] = wmv[j] + wscale * v;
}
void gpu_styl_affine(const float *wmv, const float *kstyle, int bn, int ic, int sdim, float wscale, float *ksc) {
  long n = (long)bn * 2 * ic;
  _styl_affine_warp<<<(unsigned)((n * 32 + 255) / 256), 256>>>(wmv, kstyle, n, ic, sdim, wscale, ksc);
}

// pixel norm forward: one warp per pixel, lanes stride over channels
__global__ void _warp_synth_pnrm(const float *in, long pn, int ic, float *out) {
  long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (w >= pn) return;
  const float *x = in + w * ic;
  float s = 0;
  for (int c = lane; c < ic; c += 32) s += x[c] * x[c];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  s = 1.0f / sqrtf(s / (float)ic + 1e-8f);
  float *y = out + w * ic;
  for (int c = lane; c < ic; c += 32) y[c] = x[c] * s;
}
void gpu_synth_pnrm(const float *in, long pn, int ic, float *out) {
  _warp_synth_pnrm<<<(unsigned)((pn * 32 + 255) / 256), 256>>>(in, pn, ic, out);
}

// style backward reductions: kd[b][c] = sum_p g*x (d scale), kd[b][ic+c] = sum_p g (d shift);
// grid = (pixel chunks, samples), threads stride over channels with coalesced reads, block partials atomically added
__global__ void _styl_dsc_blk(const float *x, const float *g, long pn, int ic, long pchunk, float *kd) {
  extern __shared__ float sacc[];   // 2 * ic
  const int tid = threadIdx.x, nt = blockDim.x;
  const long b = blockIdx.y, pa = blockIdx.x * pchunk, pb = min(pn, pa + pchunk);
  for (int j = tid; j < 2 * ic; j += nt) sacc[j] = 0;
  __syncthreads();
  const long base = b * pn * ic;
  if (ic <= nt && nt % ic == 0) {
    int c = tid % ic, lane = tid / ic, nl = nt / ic;
    float sx = 0, sg = 0;
    for (long p = pa + lane; p < pb; p += nl) { long q = base + p * ic + c; float gv = g[q]; sx += gv * x[q]; sg += gv; }
    atomicAdd(&sacc[c], sx); atomicAdd(&sacc[ic + c], sg);
  } else {
    for (long e = pa * ic + tid; e < pb * ic; e += nt) { int c = e % ic; long q = base + e; float gv = g[q]; atomicAdd(&sacc[c], gv * x[q]); atomicAdd(&sacc[ic + c], gv); }
  }
  __syncthreads();
  for (int j = tid; j < 2 * ic; j += nt) atomicAdd(&kd[b * 2 * ic + j], sacc[j]);
}
void gpu_styl_dsc(const float *x, const float *g, long pn, int ic, int bn, float *kd) {
  kfill(kd, (unsigned)((long)bn * 2 * ic), 0);
  long pchunk = (pn + 15) / 16; if (pchunk < 64) pchunk = 64;
  dim3 grid((unsigned)((pn + pchunk - 1) / pchunk), bn);
  _styl_dsc_blk<<<grid, 256, 2 * ic * sizeof(float)>>>(x, g, pn, ic, pchunk, kd);
}
