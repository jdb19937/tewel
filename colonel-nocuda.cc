#define __MAKEMORE_COLONEL_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <math.h>

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


int size_conv(
  int d, int ic, int oc, bool stripped
) {
  int f = d * 2 + 1;
  return ((stripped ? 1 : 3) * (oc + ic * f * f * oc));
}



void synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv, bool stripped
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_conv, outn,
    in, iw, ih, out, d, ic, oc, wmv, stripped
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
