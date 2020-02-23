#define __MAKEMORE_COLONEL_CC__ 1

#include <stdio.h>
#include <assert.h>

#include "colonel.hh"

namespace makemore {

void enkv(const void *a, unsigned int n, void *da) {
  cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
}

void dekv(const void *da, unsigned int n, void *a) {
  cudaMemcpy(a, da, n, cudaMemcpyDeviceToHost);
}

void kmakev(void **dp, unsigned int n) {
  void *d = NULL;
  assert(n > 0);
  int ret = cudaMalloc((void **)&d, n);
  assert(d != NULL);
  assert(ret == 0);
  *dp = d;
}

void kfreev(void *x) {
  cudaFree(x);
}

void kzerov(void *x, unsigned int n) {
  cudaMemset((void *)x, 0, n);
}

void kfill(double *x, unsigned int n, double v) {
  double *y = new double[n];
  for (unsigned int i = 0; i < n; ++i)
    y[i] = v;
  enk(y, n, x);
  delete[] y;
}

void kcopyv(const void *x, unsigned int n, void *y) {
  cudaMemcpy(y, x, n, cudaMemcpyDeviceToDevice);
}

__global__ void _kaddvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

void kaddvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 256;
  int gs = ((n + bs - 1) / bs);
  _kaddvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void _ksubvec(const double *a, const double *b, unsigned int n, double *c) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] - b[i];
}

void ksubvec(const double *a, const double *b, unsigned int n, double *c) {
  int bs = 256;
  int gs = ((n + bs - 1) / bs);
  _ksubvec<<<gs, bs>>>(a, b, n, c);
}

__global__ void _ksumsq(
  const double *a, unsigned int n, double *sumsqp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);
  
  double s = 0;
  for (unsigned int i = i0; i < i1; ++i)
    s += a[i] * a[i];
  sumsqp[si] = s;
}

double ksumsq(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *sumsqp = NULL;
  unsigned int sumsqn = ((n + 127) / 128);
  kmake(&sumsqp, sumsqn);

  int bs = 128;
  int gs = (sumsqn + bs - 1) / bs;
  _ksumsq<<<gs, bs>>>(a, n, sumsqp);

  double *sumsqv = new double[sumsqn];
  dek(sumsqp, sumsqn, sumsqv);
  kfree(sumsqp);

  double s = 0;
  for (int i = 0; i < sumsqn; ++i)
    s += sumsqv[i];

  delete[] sumsqv;

  return s;
}

__global__ void _kmaxabs(
  const double *a, unsigned int n, double *maxp
) {
  unsigned int si = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i0 = si * 128;
  if (i0 >= n)
    return;
  unsigned int i1 = (i0 + 128 >= n) ? n : (i0 + 128);

  unsigned int i = i0;
  double s = fabs(a[i]);
  ++i;
  
  for (; i < i1; ++i) {
    double aa = fabs(a[i]);
    if (aa > s)
      s = aa;
  }

  maxp[si] = s;
}

double kmaxabs(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *maxp = NULL;
  unsigned int maxn = ((n + 127) / 128);
  kmake(&maxp, maxn);

  int bs = 128;
  int gs = (maxn + bs - 1) / bs;
  _kmaxabs<<<gs, bs>>>(a, n, maxp);

  double *maxv = new double[maxn];
  dek(maxp, maxn, maxv);

  double s = maxv[0];
  for (int i = 1; i < maxn; ++i)
    if (maxv[i] > s)
      s = maxv[i];

  kfree(maxp);
  delete[] maxv;

  return s;
}


__global__ void _kspliceadd(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * xk)
    return;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] += x[in * xm + xa + ixk];
}

void kspliceadd(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  int bs = 256;
  int gs = ((n * xk + bs - 1) / bs);
  _kspliceadd<<<gs, bs>>>(x, n, xm, xa, xk, y, ym, ya);
}



__global__ void _ksplice(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * xk)
    return;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] = x[in * xm + xa + ixk];
}

void ksplice(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  int bs = 256;
  int gs = ((n * xk + bs - 1) / bs);
  _ksplice<<<gs, bs>>>(x, n, xm, xa, xk, y, ym, ya);
}



__global__ void k_synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;


  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int ix0 = ox - d;
  int ix1 = ox + d;
  int iy0 = oy - d;
  int iy1 = oy + d;

  int f = d * 2 + 1;
  // int wmvn = 3 * (oc + ic * f * f * oc);

  double outv = 0;

  for (int iyt = iy0; iyt <= iy1; ++iyt) {
    int iy = (iyt + ih) % ih;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;
      for (int iz = 0; iz < ic; ++iz) {
        int ini = iz + ic * (ix + iw * iy);

        int dx = ixt - ix0;
        int dy = iyt - iy0;
        int wi = oc + (iz + ic * (dx + f * (dy + f * oz)));

        double v = in[ini];
        double w = wmv[wi * 3];

        outv += v * w;
      }
    }
  }

  outv += wmv[oz * 3];

  out[outi] = outv;
}


int size_conv(
  int d, int ic, int oc
) {
  int f = d * 2 + 1;
  return (3 * (oc + ic * f * f * oc));
}



void synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_conv<<<gs, bs>>>(
    in, iw, ih, out, d, ic, oc, wmv
  );
}

__global__ void k_learn_conv1(
  const double *in, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *wmv,
  double nu, double b1, double b2
) {
  if (!(nu > 0))
    return;

  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);

  int wi = blockIdx.x * blockDim.x + threadIdx.x;
  if (wi >= wn)
    return;

  int ow = iw;
  int oh = ih;

  double dw = 0;

  if (wi < oc) {
    int oz = wi;

    for (int oy = 0; oy < oh; ++oy) {
      for (int ox = 0; ox < ow; ++ox) {
        int outi = oz + oc * (ox + ow * oy);
        dw += fout[outi];
      }
    }
  } else {
    int tmp = wi - oc;
    int iz = tmp % ic; tmp /= ic;
    int dx = tmp % f; tmp /= f;
    int dy = tmp % f; tmp /= f;
    int oz = tmp % oc;

    for (int oy = 0; oy < oh; ++oy) {
      for (int ox = 0; ox < ow; ++ox) {
        int ix = (ox - d + dx + iw) % iw;
        int iy = (oy - d + dy + ih) % ih;

        int ini = iz + ic * (ix + iw * iy);
        int outi = oz + oc * (ox + ow * oy);

        dw += fout[outi] * in[ini];
      }
    }
  }

  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];

  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;

  wmv[wi * 3 + 1] = m;
  wmv[wi * 3 + 2] = v;
}


__global__ void k_learn_conv2(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  const double *wmv
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int ow = iw;
  int oh = ih;

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int ox0 = ix - d;
  int ox1 = ix + d;
  int oy0 = iy - d;
  int oy1 = iy + d;

  int f = d * 2 + 1;
  // int wmvn = 3 * (oc + ic * f * f * oc);

  double finv = 0;

  for (int oyt = oy0; oyt <= oy1; ++oyt) {
    int oy = (oyt + oh) % oh;
    for (int oxt = ox0; oxt <= ox1; ++oxt) { 
      int ox = (oxt + ow) % ow;
      for (int oz = 0; oz < oc; ++oz) {
        int outi = oz + oc * (ox + ow * oy);

        int dx = ix - oxt + d;
        int dy = iy - oyt + d;
        int wi = oc + (iz + ic * (dx + f * (dy + f * oz)));

        double v = fout[outi];
        double w = wmv[wi * 3];
        finv += v * w;
      }
    }
  }

  fin[ini] = finv;
}


__global__ void k_learn_conv3(
  int d, int ic, int oc,

  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  if (!(nu > 0))
    return;

  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);

  int wi = blockIdx.x * blockDim.x + threadIdx.x;
  if (wi >= wn)
    return;

  double w = wmv[wi * 3 + 0];
  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];

  if (rounds < 1024.0) {
    m = m / (1.0 - pow(1.0 - b1, 1.0 + rounds));
    v = v / (1.0 - pow(1.0 - b2, 1.0 + rounds));
  }

  w += nu * m / (sqrt(v) + eps);

  wmv[wi * 3 + 0] = w;
}

void learn_conv(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);
  int inn = iw * ih * ic;

  int bs = 256;
  int gsw = (wn + bs - 1) / bs;
  int gsi = (inn + bs - 1) / bs;

  k_learn_conv1<<<gsw, bs>>>(fin, iw, ih, fout, d, ic, oc, wmv, nu, b1, b2);
  k_learn_conv2<<<gsi, bs>>>(fin, iw, ih, fout, d, ic, oc, wmv);
  k_learn_conv3<<<gsw, bs>>>(d, ic, oc, wmv, nu, b1, b2, eps, rounds);
}


__global__ void k_synth_pad(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;
  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int ix = ox;
  int iy = oy;

  if (oz < ic) {
    int iz = oz;
    int ini = iz + ic * (ix + iw * iy);
    out[outi] = in[ini];
  } else {
    int kbufi = (oz - ic) + (oc - ic) * (ox + ow * oy);
    out[outi] = kbuf[kbufi];
  }

}

void synth_pad(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_pad<<<gs, bs>>>(
    in, iw, ih, out, ic, oc, kbuf
  );
}

__global__ void k_learn_pad(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int ow = iw;

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int oz = iz;
  int ox = ix;
  int oy = iy;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi];
}

void learn_pad(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  int bs = 256;
  int gs = (inn + bs - 1) / bs;
  k_learn_pad<<<gs, bs>>>(
    fin, iw, ih, fout, ic, oc
  );
}

__global__ void k_synth_relu(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  int ini = outi;
  if (in[ini] > 0)
    out[outi] = in[ini];
  else
    out[outi] = 0;
}

void synth_relu(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_relu<<<gs, bs>>>(
    in, iw, ih, out, ic
  );
}

__global__ void k_learn_relu(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int outi = ini;

  if (fin[ini] > 0) {
    fin[ini] = fout[outi];
  } else {
    fin[ini] = 0;
  }
}

void learn_relu(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  int bs = 256;
  int gs = (inn + bs - 1) / bs;
  k_learn_relu<<<gs, bs>>>(
    fin, iw, ih, fout, ic
  );
}


__global__ void k_synth_abs(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  int ini = outi;
  if (in[ini] > 0)
    out[outi] = in[ini];
  else
    out[outi] = -in[ini];
}

void synth_abs(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_abs<<<gs, bs>>>(
    in, iw, ih, out, ic
  );
}

__global__ void k_learn_abs(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int outi = ini;

  if (fin[ini] > 0) {
    fin[ini] = fout[outi];
  } else {
    fin[ini] = -fout[outi];
  }
}

void learn_abs(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  int bs = 256;
  int gs = (inn + bs - 1) / bs;
  k_learn_abs<<<gs, bs>>>(
    fin, iw, ih, fout, ic
  );
}


__global__ void k_synth_upscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  int ow = (iw << s);
  int oh = (ih << s);

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  // assert((oc << (s + s)) == ic);
  int ix = (ox >> s);
  int iy = (oy >> s);

  int f = (1 << s);
  int dx = ox % f;
  int dy = oy % f;

  int iz = oz + oc * (dx + f * dy);
  int ini = iz + ic * (ix + iw * iy);

  out[outi] = in[ini];
}

void synth_upscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  assert((oc << (s + s)) == ic);

  int ow = (iw << s);
  int oh = (ih << s);
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_upscale<<<gs, bs>>>(
    in, iw, ih, out, s, ic, oc
  );
}

__global__ void k_learn_upscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int ow = (iw << s);
  // int oh = (ih << s);

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int f = (1 << s);
  tmp = iz;
  int oz = tmp % oc; tmp /= oc;
  int dx = tmp % f; tmp /= f;
  int dy = tmp;
  int ox = (ix << s) + dx;
  int oy = (iy << s) + dy;

  int outi = oz + oc * (ox + ow * oy);

  fin[ini] = fout[outi];
}

void learn_upscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;

  int bs = 256;
  int gs = (inn + bs - 1) / bs;

  k_learn_upscale<<<gs, bs>>>(fin, iw, ih, fout, s, ic, oc);
}



__global__ void k_synth_downscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  int ow = (iw >> s);
  int oh = (ih >> s);

  int outn = ow * oh * oc;
  int outi = blockIdx.x * blockDim.x + threadIdx.x;
  if (outi >= outn)
    return;

  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int f = (1 << s);
  tmp = oz;
  int iz = tmp % ic; tmp /= ic;
  int dx = tmp % f; tmp /= f;
  int dy = tmp;

  int ix = (ox << s) + dx;
  int iy = (oy << s) + dy;

  int ini = iz + ic * (ix + iw * iy);

  out[outi] = in[ini];
}

void synth_downscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  int bs = 256;
  int gs = (outn + bs - 1) / bs;
  k_synth_downscale<<<gs, bs>>>(
    in, iw, ih, out, s, ic, oc
  );
}


__global__ void k_learn_downscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;
  int ini = blockIdx.x * blockDim.x + threadIdx.x;
  if (ini >= inn)
    return;

  int ow = (iw >> s);
  // int oh = (ih >> s);

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int f = (1 << s);
  int ox = (ix >> s);
  int oy = (iy >> s);

  int dx = ix % f;
  int dy = iy % f;
  int oz = iz + ic * (dx + f * dy);

  int outi = oz + oc * (ox + ow * oy);

  fin[ini] = fout[outi];
}

void learn_downscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;

  int bs = 256;
  int gs = (inn + bs - 1) / bs;

  k_learn_downscale<<<gs, bs>>>(fin, iw, ih, fout, s, ic, oc);
}





}
