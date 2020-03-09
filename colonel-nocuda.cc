#define __MAKEMORE_COLONEL_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <math.h>

#include <cmath>
#include <limits>

#include "colonel.hh"

namespace makemore {

int kdev = 0;
int kbs = 256;

void setkdev(int i) {
  assert(i == 0);
  kdev = 0;
}

void setkbs(int i) {
  assert(i > 0);
  kbs = i;
}

int kndevs() {
  return 1;
}




// MIT License
// 
// Copyright (c) 2017-2019 Lakshay Garg <lakshayg@outlook.in>

// Implementation of the inverse error function based on the rational
// approximation of percentage points of normal distribution available from
// https://www.jstor.org/stable/2347330.

// The code has been tested on an x86_64 machine with Intel Core i7, the
// tests provided in this repository might not pass for different hardware
// configuration due to difference in floating point operations.

// golang's math library uses the same implementation for erfinv


static long double erfinv(long double x) {

  if (x < -1 || x > 1) {
    return std::numeric_limits<long double>::quiet_NaN();
  } else if (x == 1.0) {
    return std::numeric_limits<long double>::infinity();
  } else if (x == -1.0) {
    return -std::numeric_limits<long double>::infinity();
  }

  const long double LN2 = 6.931471805599453094172321214581e-1L;

  const long double A0 = 1.1975323115670912564578e0L;
  const long double A1 = 4.7072688112383978012285e1L;
  const long double A2 = 6.9706266534389598238465e2L;
  const long double A3 = 4.8548868893843886794648e3L;
  const long double A4 = 1.6235862515167575384252e4L;
  const long double A5 = 2.3782041382114385731252e4L;
  const long double A6 = 1.1819493347062294404278e4L;
  const long double A7 = 8.8709406962545514830200e2L;

  const long double B0 = 1.0000000000000000000e0L;
  const long double B1 = 4.2313330701600911252e1L;
  const long double B2 = 6.8718700749205790830e2L;
  const long double B3 = 5.3941960214247511077e3L;
  const long double B4 = 2.1213794301586595867e4L;
  const long double B5 = 3.9307895800092710610e4L;
  const long double B6 = 2.8729085735721942674e4L;
  const long double B7 = 5.2264952788528545610e3L;

  const long double C0 = 1.42343711074968357734e0L;
  const long double C1 = 4.63033784615654529590e0L;
  const long double C2 = 5.76949722146069140550e0L;
  const long double C3 = 3.64784832476320460504e0L;
  const long double C4 = 1.27045825245236838258e0L;
  const long double C5 = 2.41780725177450611770e-1L;
  const long double C6 = 2.27238449892691845833e-2L;
  const long double C7 = 7.74545014278341407640e-4L;

  const long double D0 = 1.4142135623730950488016887e0L;
  const long double D1 = 2.9036514445419946173133295e0L;
  const long double D2 = 2.3707661626024532365971225e0L;
  const long double D3 = 9.7547832001787427186894837e-1L;
  const long double D4 = 2.0945065210512749128288442e-1L;
  const long double D5 = 2.1494160384252876777097297e-2L;
  const long double D6 = 7.7441459065157709165577218e-4L;
  const long double D7 = 1.4859850019840355905497876e-9L;

  const long double E0 = 6.65790464350110377720e0L;
  const long double E1 = 5.46378491116411436990e0L;
  const long double E2 = 1.78482653991729133580e0L;
  const long double E3 = 2.96560571828504891230e-1L;
  const long double E4 = 2.65321895265761230930e-2L;
  const long double E5 = 1.24266094738807843860e-3L;
  const long double E6 = 2.71155556874348757815e-5L;
  const long double E7 = 2.01033439929228813265e-7L;

  const long double F0 = 1.414213562373095048801689e0L;
  const long double F1 = 8.482908416595164588112026e-1L;
  const long double F2 = 1.936480946950659106176712e-1L;
  const long double F3 = 2.103693768272068968719679e-2L;
  const long double F4 = 1.112800997078859844711555e-3L;
  const long double F5 = 2.611088405080593625138020e-5L;
  const long double F6 = 2.010321207683943062279931e-7L;
  const long double F7 = 2.891024605872965461538222e-15L;

  long double abs_x = abs(x);

  if (abs_x <= 0.85L) {
    long double r =  0.180625L - 0.25L * x * x;
    long double num = (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0);
    long double den = (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + B0);
    return x * num / den; 
  }

  long double r = sqrt(LN2 - log(1.0L - abs_x));

  long double num, den;
  if (r <= 5.0L) {
    r = r - 1.6L;
    num = (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0);
    den = (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + D0);
  } else {
    r = r - 5.0L;
    num = (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0);
    den = (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + F0);
  }

  if (x < 0L) {
    return -num / den;
  } else {
    return num / den;
  }
}

static long double erfinv(long double x, int nr_iter) {
  long double k = 0.8862269254527580136490837416706L; // 0.5 * sqrt(pi)
  long double y = erfinv(x);
  while (nr_iter-- > 0) {
    y -= k * (std::erf(y) - x) / std::exp(-y * y);
  }
  return y;
}






#undef DEFN_KERNEL
#define DEFN_KERNEL(f, args...) \
  void _cpu_ ## f (long i, long __n, args)
#undef PREF_KERNEL
#define PREF_KERNEL \
  if (i >= __n) \
    return;

#include "colonel.inc"


#define CALL_KERNEL(f, _n, args...) do { \
  long __n = (_n); \
  for (long __i = __n - 1; __i >= 0; --__i) { \
    _cpu_ ## f (__i, __n, args); \
  } \
} while (0);



void enkv(const void *a, unsigned int n, void *da) {
  ::memcpy(da, a, n);
}

void dekv(const void *da, unsigned int n, void *a) {
  ::memcpy(a, da, n);
}

void kmakev(void **dp, unsigned int n) {
  *dp = (void *)(new char[n]);
  assert(*dp);
}

void kfreev(void *x) {
  delete[] ((char *)x);
}

void kzerov(void *x, unsigned int n) {
  ::memset(x, 0, n);
}

void kfill(double *x, unsigned int n, double v) {
  for (unsigned int i = 0; i < n; ++i)
    x[i] = v;
}

void kcopyv(const void *x, unsigned int n, void *y) {
  ::memcpy(y, x, n);
}

void kaddvec(const double *a, const double *b, unsigned int n, double *c) {
  CALL_KERNEL(kaddvec, n, a, b, n, c);
}

void ksubvec(const double *a, const double *b, unsigned int n, double *c) {
  CALL_KERNEL(ksubvec, n, a, b, n, c);
}

double ksumsq(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *sumsqp = NULL;
  unsigned int sumsqn = ((n + 127) / 128);
  kmake(&sumsqp, sumsqn);

  CALL_KERNEL(ksumsq, sumsqn, a, n, sumsqp);

  double *sumsqv = new double[sumsqn];
  dek(sumsqp, sumsqn, sumsqv);
  kfree(sumsqp);

  double s = 0;
  for (int i = 0; i < sumsqn; ++i)
    s += sumsqv[i];

  delete[] sumsqv;

  return s;
}

double kmaxabs(
  const double *a, unsigned int n
) {
  if (n == 0)
    return 0;

  double *maxp = NULL;
  unsigned int maxn = ((n + 127) / 128);
  kmake(&maxp, maxn);

  CALL_KERNEL(kmaxabs, maxn, a, n, maxp);

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


void kspliceadd(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  CALL_KERNEL(kspliceadd, n * xk, x, n, xm, xa, xk, y, ym, ya);
}



void ksplice(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  CALL_KERNEL(ksplice, n * xk, x, n, xm, xa, xk, y, ym, ya);
}

int size_norm(
  int ic, int ow, int oh, int oc
) {
  return (2 * 3 * (ow * oh * oc));
}

void synth_norm(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_norm, outn,
    in, iw, ih, out, ic, oc, wmv
  );
}


void learn_norm(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  int outn = iw * ih * oc;
  int wn = outn * 2;
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_norm1, wn, fin, iw, ih, fout, ic, oc, wmv, nu, b1, b2);
  CALL_KERNEL(learn_norm2, inn, fin, iw, ih, fout, ic, oc, wmv);
  CALL_KERNEL(learn_norm3, wn, iw, ih, ic, oc, wmv, nu, b1, b2, eps, rounds);
}


int size_bias(
  int ic, int ow, int oh, int oc
) {
  return (3 * (ow * oh * oc));
}

void synth_bias(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_bias, outn,
    in, iw, ih, out, ic, oc, wmv
  );
}


void learn_bias(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  int outn = iw * ih * oc;
  int wn = outn;
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_bias1, wn, fin, iw, ih, fout, ic, oc, wmv, nu, b1, b2);
  CALL_KERNEL(learn_bias2, inn, fin, iw, ih, fout, ic, oc, wmv);
  CALL_KERNEL(learn_bias3, wn, iw, ih, ic, oc, wmv, nu, b1, b2, eps, rounds);
}

int size_local(
  int d, int ic, int ow, int oh, int oc
) {
  int f = d * 2 + 1;
  return (3 * (ow * oh * oc * (1 + ic * f * f)));
}

void synth_local(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_local, outn,
    in, iw, ih, out, d, ic, oc, wmv
  );
}


void learn_local(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  int f = d * 2 + 1;
  int outn = iw * ih * oc;
  int wn = (outn + ic * f * f * outn);
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_local1, wn, fin, iw, ih, fout, d, ic, oc, wmv, nu, b1, b2);
  CALL_KERNEL(learn_local2, inn, fin, iw, ih, fout, d, ic, oc, wmv);
  CALL_KERNEL(learn_local3, wn, iw, ih, d, ic, oc, wmv, nu, b1, b2, eps, rounds);
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

  CALL_KERNEL(synth_conv, outn,
    in, iw, ih, out, d, ic, oc, wmv
  );
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

  CALL_KERNEL(learn_conv1, wn, fin, iw, ih, fout, d, ic, oc, wmv, nu, b1, b2);
  CALL_KERNEL(learn_conv2, inn, fin, iw, ih, fout, d, ic, oc, wmv);
  CALL_KERNEL(learn_conv3, wn, d, ic, oc, wmv, nu, b1, b2, eps, rounds);
}

void synth_iden(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_iden, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_iden(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_iden, inn,
    fin, iw, ih, fout, ic, oc
  );
}



void synth_mean(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_mean, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_mean(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_mean, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_pad(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_pad, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_pad(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_pad, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_sigm(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_sigm, outn,
    in, iw, ih, out, ic
  );
}

void learn_sigm(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_sigm, inn,
    fin, iw, ih, fout, ic
  );
}





void synth_nerf(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_nerf, outn,
    in, iw, ih, out, ic
  );
}

void learn_nerf(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_nerf, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_inrf(
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_inrf, outn,
    in, iw, ih, out, ic
  );
}

void learn_inrf(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_inrf, inn,
    fin, iw, ih, fout, ic
  );
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

  CALL_KERNEL(synth_relu, outn,
    in, iw, ih, out, ic
  );
}

void learn_relu(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_relu, inn,
    fin, iw, ih, fout, ic
  );
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

  CALL_KERNEL(synth_abs, outn,
    in, iw, ih, out, ic
  );
}

void learn_abs(
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_abs, inn,
    fin, iw, ih, fout, ic
  );
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

  CALL_KERNEL(synth_upscale, outn,
    in, iw, ih, out, s, ic, oc
  );
}


void learn_upscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_upscale, inn, fin, iw, ih, fout, s, ic, oc);
}




void synth_downscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_downscale, outn,
    in, iw, ih, out, s, ic, oc
  );
}


void learn_downscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_downscale, inn, fin, iw, ih, fout, s, ic, oc);
}


}
