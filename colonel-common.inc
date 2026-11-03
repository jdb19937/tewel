#include <stdint.h>
// float4 paths for the big elementwise passes (GPU, 16-byte aligned, n % 4 == 0)
static inline bool kvec4ok(const void *a, const void *b, const void *c, unsigned int n) {
  return kdev && (n % 4 == 0) && ((((uintptr_t)a) | ((uintptr_t)b) | ((uintptr_t)c)) & 15) == 0;
}

void kaxpy(const float *x, float a, unsigned int n, float *y) {
  if (kvec4ok(x, y, y, n)) { gpu_axpy4(x, a, n / 4, y); return; }
  CALL_KERNEL(kaxpy, n, x, a, n, y);
}

void kaddvec(const float *a, const float *b, unsigned int n, float *c) {
  if (kvec4ok(a, b, c, n)) { gpu_addvec4(a, b, n / 4, c); return; }
  CALL_KERNEL(kaddvec, n, a, b, n, c);
}

void ksubvec(const float *a, const float *b, unsigned int n, float *c) {
  if (kvec4ok(a, b, c, n)) { gpu_subvec4(a, b, n / 4, c); return; }
  CALL_KERNEL(ksubvec, n, a, b, n, c);
}

void kmul(const float *a, const float b, unsigned int n, float *c) {
  if (kvec4ok(a, c, c, n)) { gpu_mul4(a, b, n / 4, c); return; }
  CALL_KERNEL(kmul, n, a, b, n, c);
}

// u8 image rows -> float samples on the device: upload the bytes, convert/crop/flip in a kernel
void ku8crop(const unsigned char *rows, size_t nbytes, int sw, int sc, int x0, int pw, int ph, int pc, bool flip, float *kdat) {
  unsigned char *ksrc; kmakev((void **)&ksrc, (unsigned int)nbytes);
  enkv(rows, (unsigned int)nbytes, ksrc);
  CALL_KERNEL(u8crop, (long)pw * ph * pc, ksrc, sw, sc, x0, pw, ph, pc, flip ? 1 : 0, kdat);
  kfreev(ksrc);
}

float ksum(
  const float *a, unsigned int n
) {
  if (n == 0)
    return 0;

  float *sump = NULL;
  unsigned int sumn = ((n + 127) / 128);
  kmake(&sump, sumn);

  CALL_KERNEL(ksum, sumn, a, n, sump);

  float *sumv = new float[sumn];
  dek(sump, sumn, sumv);
  kfree(sump);

  float s = 0;
  for (int i = 0; i < sumn; ++i)
    s += sumv[i];

  delete[] sumv;

  return s;
}

float ksumsq(
  const float *a, unsigned int n
) {
  if (n == 0)
    return 0;

  float *sumsqp = NULL;
  unsigned int sumsqn = ((n + 127) / 128);
  kmake(&sumsqp, sumsqn);

  CALL_KERNEL(ksumsq, sumsqn, a, n, sumsqp);

  float *sumsqv = new float[sumsqn];
  dek(sumsqp, sumsqn, sumsqv);
  kfree(sumsqp);

  float s = 0;
  for (int i = 0; i < sumsqn; ++i)
    s += sumsqv[i];

  delete[] sumsqv;

  return s;
}

float kmaxabs(
  const float *a, unsigned int n
) {
  if (n == 0)
    return 0;

  float *maxp = NULL;
  unsigned int maxn = ((n + 127) / 128);
  kmake(&maxp, maxn);

  CALL_KERNEL(kmaxabs, maxn, a, n, maxp);

  float *maxv = new float[maxn];
  dek(maxp, maxn, maxv);

  float s = maxv[0];
  for (int i = 1; i < maxn; ++i)
    if (maxv[i] > s)
      s = maxv[i];

  kfree(maxp);
  delete[] maxv;

  return s;
}


void kspliceadd(
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  CALL_KERNEL(kspliceadd, n * xk, x, n, xm, xa, xk, y, ym, ya);
}



void ksplice(
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  CALL_KERNEL(ksplice, n * xk, x, n, xm, xa, xk, y, ym, ya);
}

void ksplice2(
  const float *x, const float *z, float l, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  CALL_KERNEL(ksplice2, n * xk, x, z, l, n, xm, xa, xk, y, ym, ya);
}

int size_norm(
  int ic, int ow, int oh, int oc
) {
  return (2 * 3 * (ow * oh * oc));
}

void synth_norm(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_norm, outn,
    in, iw, ih, out, ic, oc, wmv
  );
}


void learn_norm(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
) {
  // Adam bias correction, computed once per layer instead of per weight
  float mc = 1.0, vc = 1.0;
  if (rounds < 1024.0) {
    double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
    if (dm > 1e-12) mc = 1.0 / dm;
    if (dv > 1e-12) vc = 1.0 / dv;
  }
  int outn = iw * ih * oc;
  int wn = outn * 2;
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_norm1, wn, fin, iw, ih, fout, ic, oc, dwv, nu, b1, b2);
  CALL_KERNEL(learn_norm2, inn, fin, iw, ih, fout, ic, oc, wmv);
  CALL_KERNEL(learn_norm3, wn, iw, ih, ic, oc, wmv, dwv, nu, b1, b2, eps, clip, rounds, mc, vc);
}

int size_bias(
  int ic, int ow, int oh, int oc
) {
  return (3 * (ow * oh * oc));
}

void synth_bias(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_bias, outn,
    in, iw, ih, out, ic, oc, wmv
  );
}


void learn_bias(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
) {
  // Adam bias correction, computed once per layer instead of per weight
  float mc = 1.0, vc = 1.0;
  if (rounds < 1024.0) {
    double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
    if (dm > 1e-12) mc = 1.0 / dm;
    if (dv > 1e-12) vc = 1.0 / dv;
  }
  int outn = iw * ih * oc;
  int wn = outn;
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_bias1, wn, fin, iw, ih, fout, ic, oc, dwv, nu, b1, b2);
  CALL_KERNEL(learn_bias2, inn, fin, iw, ih, fout, ic, oc, wmv);
  CALL_KERNEL(learn_bias3, wn, iw, ih, ic, oc, wmv, dwv, nu, b1, b2, eps, clip, rounds, mc, vc);
}

int size_local(
  int d, int ic, int ow, int oh, int oc
) {
  int f = d * 2 + 1;
  return (3 * (ow * oh * oc * (1 + ic * f * f)));
}

void synth_local(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  const float *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_local, outn,
    in, iw, ih, out, d, ic, oc, wmv
  );
}


void learn_local(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
) {
  // Adam bias correction, computed once per layer instead of per weight
  float mc = 1.0, vc = 1.0;
  if (rounds < 1024.0) {
    double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
    if (dm > 1e-12) mc = 1.0 / dm;
    if (dv > 1e-12) vc = 1.0 / dv;
  }
  int f = d * 2 + 1;
  int outn = iw * ih * oc;
  int wn = (outn + ic * f * f * outn);
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_local1, wn, fin, iw, ih, fout, d, ic, oc, dwv, nu, b1, b2);
  CALL_KERNEL(learn_local2, inn, fin, iw, ih, fout, d, ic, oc, wmv);
  CALL_KERNEL(learn_local3, wn, iw, ih, d, ic, oc, wmv, dwv, nu, b1, b2, eps, clip, rounds, mc, vc);
}


int size_conv(
  int d, int ic, int oc, int dim
) {
  int f = d * 2 + 1;
  assert(f > 0);
  assert(dim == 1 || dim == 2);
  if (dim == 1) {
    return (3 * (oc + ic * f * oc));
  } else {
    return (3 * (oc + ic * f * f * oc));
  }
}

void synth_crop(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic,
  int ix0, int iy0
) {
  assert(d > 0);

  int ow = d;
  int oh = d;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_crop, outn,
    in, iw, ih, out, d, ic, ix0, iy0
  );
}

void synth_padd(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic
) {
  int ow = iw + 2 * d;
  int oh = ih + 2 * d;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_padd, outn,
    in, iw, ih, out, d, ic
  );
}

void synth_trim(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic
) {
  int ow = iw - 2 * d;
  int oh = ih - 2 * d;
  int oc = ic;
  int outn = ow * oh * oc;

  assert(ow > 0 && oh > 0 && oc > 0);

  CALL_KERNEL(synth_trim, outn,
    in, iw, ih, out, d, ic
  );
}


void synth_conv(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  const float *wmv,
  int bn, float wscale
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
  int inn = iw * ih * ic;

  if (kdev) {
    if (kprec == 16) gpu_synth_conv16(in, bn, iw, ih, out, d, ic, oc, relu, freeze, dim, wmv, wscale);
    else gpu_synth_conv(in, bn, iw, ih, out, d, ic, oc, relu, freeze, dim, wmv, wscale);
    return;
  }

  for (int b = 0; b < bn; ++b) {
    CALL_KERNEL(synth_conv, outn,
      in + (size_t)b * inn, iw, ih, out + (size_t)b * outn, d, ic, oc, relu, freeze, dim, wmv, wscale
    );
  }
}

void learn_crop(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic
) {
  assert(0);
}

void learn_padd(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_padd, inn, fin, iw, ih, fout, d, ic);
}

void learn_trim(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_trim, inn, fin, iw, ih, fout, d, ic);
}


void learn_conv(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,
  bool relu, int freeze, int dim,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds,
  int bn, float wscale
) {
  // Adam bias correction, computed once per layer instead of per weight
  float mc = 1.0, vc = 1.0;
  if (rounds < 1024.0) {
    double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
    if (dm > 1e-12) mc = 1.0 / dm;
    if (dv > 1e-12) vc = 1.0 / dv;
  }
  int f = d * 2 + 1;
  int wn = (dim == 1) ? (oc + ic * f * oc) : (oc + ic * f * f * oc);
  int inn = iw * ih * ic;
  int outn = iw * ih * oc;

  if (kdev) {
    if (dwv) {
      if (kprec == 16) gpu_learn_conv_w16(fin, bn, iw, ih, fout, d, ic, oc, relu, freeze, dim, dwv, wscale);
      else gpu_learn_conv_w(fin, bn, iw, ih, fout, d, ic, oc, relu, freeze, dim, dwv, wscale);
    }
    if (kprec == 16) gpu_learn_conv_in16(fin, bn, iw, ih, fout, d, ic, oc, relu, freeze, dim, wmv, wscale);
    else gpu_learn_conv_in(fin, bn, iw, ih, fout, d, ic, oc, relu, freeze, dim, wmv, wscale);
  } else {
    for (int b = 0; b < bn; ++b) {
      const float *inb = fin + (size_t)b * inn;
      const float *foutb = fout + (size_t)b * outn;
      CALL_KERNEL(learn_conv1, (long)wn * (long)ih, inb, iw, ih, foutb, d, ic, oc, relu, freeze, dim, dwv, nu, b1, b2, wscale);
    }
    for (int b = 0; b < bn; ++b) {
      CALL_KERNEL(learn_conv2, inn, fin + (size_t)b * inn, iw, ih, fout + (size_t)b * outn, d, ic, oc, relu, freeze, dim, wmv, wscale);
    }
  }

  CALL_KERNEL(learn_conv3, wn, d, ic, oc, dim, wmv, dwv, nu, b1, b2, eps, clip, rounds, mc, vc);
}

void synth_addgeo(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  assert(oc == ic + 4);

  CALL_KERNEL(synth_addgeo, outn,
    in, iw, ih, out, ic, oc
  );
}


void learn_addgeo(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_addgeo, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_pad(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_pad, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_pad(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_pad, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_zero(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_zero, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_zero(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_zero, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_tlab(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  assert(ic == 3);
  assert(oc == 3);

  CALL_KERNEL(synth_tlab, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}

void synth_flab(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
 
  assert(ic == 3);
  assert(oc == 3);

  CALL_KERNEL(synth_flab, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void synth_popp(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_popp, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_popp(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_popp, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_iden(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_iden, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_iden(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_iden, inn,
    fin, iw, ih, fout, ic, oc
  );
}


void synth_smax(
  float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_smax, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_smax(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_smax, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_mean(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_mean, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_mean(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_mean, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_blur(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_blur, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_blur(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {

  assert(0);

//  int inn = iw * ih * ic;
//  CALL_KERNEL(learn_blur, inn,
//    fin, iw, ih, fout, ic, oc
//  );
}

void synth_median(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_median, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_median(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_median, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_diff(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
 
  assert(oc * 2 == ic);

  CALL_KERNEL(synth_diff, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_diff(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  assert(oc * 2 == ic);

  CALL_KERNEL(learn_diff, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_sum(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_sum, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_sum(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_sum, inn,
    fin, iw, ih, fout, ic, oc
  );
}

void synth_cent(
  const float *in, int iw, int ih,
  float *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_cent, outn,
    in, iw, ih, out, ic
  );
}

void learn_cent(
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_cent, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_sigm(
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_sigm, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_nerf(
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_nerf, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_inrf(
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_inrf, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_clamp(
  const float *in, int iw, int ih,
  float *out,
  int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_clamp, outn,
    in, iw, ih, out, ic
  );
}

void learn_clamp(
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_clamp, inn,
    fin, iw, ih, fout, ic
  );
}

void synth_relu(
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_relu, inn,
    fin, iw, ih, fout, ic
  );
}


void synth_abs(
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_abs, inn,
    fin, iw, ih, fout, ic
  );
}


void synth_upscale(
  const float *in, int iw, int ih,
  float *out,
  int s, int ic, int oc, int dim
) {
  assert(dim == 1 || dim == 2);
  if (dim == 1) {
    assert((oc << s) == ic);
  } else {
    assert((oc << (s + s)) == ic);
  }

  int ow = (iw << s);
  int oh = (ih << s);
  int outn = ow * oh * oc;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(synth_upscale, outn,
    in, iw, ih, out, s, ic, oc, dim
  );
}


void learn_upscale(
  float *fin, int iw, int ih,
  const float *fout,
  int s, int ic, int oc, int dim
) {
  int inn = iw * ih * ic;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(learn_upscale, inn, fin, iw, ih, fout, s, ic, oc, dim);
}




void synth_downscale(
  const float *in, int iw, int ih,
  float *out,
  int s, int ic, int oc, int dim
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(synth_downscale, outn,
    in, iw, ih, out, s, ic, oc, dim
  );
}


void learn_downscale(
  float *fin, int iw, int ih,
  const float *fout,
  int s, int ic, int oc, int dim
) {
  int inn = iw * ih * ic;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(learn_downscale, inn, fin, iw, ih, fout, s, ic, oc, dim);
}



void kcharb(float *e, unsigned int n, float delta) {
  CALL_KERNEL(kcharb, n, e, n, delta);
}

void synth_lrlu(const float *in, int iw, int ih, float *out, int ic, float slope) {
  int outn = iw * ih * ic;
  if (kvec4ok(in, out, out, outn)) { gpu_lrlu4_fwd(in, outn / 4, out, slope); return; }
  CALL_KERNEL(synth_lrlu, outn, in, iw, ih, out, ic, slope);
}

void learn_lrlu(float *fin, int iw, int ih, const float *fout, int ic, float slope) {
  int inn = iw * ih * ic;
  if (kvec4ok(fin, fout, fout, inn)) { gpu_lrlu4_bwd(fin, inn / 4, fout, slope); return; }
  CALL_KERNEL(learn_lrlu, inn, fin, iw, ih, fout, ic, slope);
}

void synth_pnrm(const float *in, int iw, int ih, float *out, int ic) {
  int outn = iw * ih * ic;
  if (kdev) { gpu_synth_pnrm(in, iw * ih, ic, out); return; }
  CALL_KERNEL(synth_pnrm, outn, in, iw, ih, out, ic);
}

void learn_pnrm(float *fin, int iw, int ih, const float *fout, int ic) {
  int pn = iw * ih;
  if (kdev) {
    gpu_learn_pnrm(fin, pn, ic, fout);
    return;
  }
  CALL_KERNEL(learn_pnrm, pn, fin, iw, ih, fout, ic);
}


void synth_upbl(const float *in, int iw, int ih, float *out, int ic, int bn) {
  long outn = (long)bn * iw * 2 * ih * 2 * ic;
  CALL_KERNEL(synth_upbl, outn, in, iw, ih, out, ic, bn);
}
void learn_upbl(float *fin, int iw, int ih, const float *fout, int ic, int bn) {
  long inn = (long)bn * iw * ih * ic;
  CALL_KERNEL(learn_upbl, inn, fin, iw, ih, fout, ic, bn);
}
void synth_dnav(const float *in, int iw, int ih, float *out, int ic) {
  long outn = (long)(iw / 2) * (ih / 2) * ic;
  CALL_KERNEL(synth_dnav, outn, in, iw, ih, out, ic);
}
void learn_dnav(float *fin, int iw, int ih, const float *fout, int ic) {
  long inn = (long)iw * ih * ic;
  CALL_KERNEL(learn_dnav, inn, fin, iw, ih, fout, ic);
}
void kema(float *a, const float *b, unsigned int n, float beta) {
  CALL_KERNEL(kema, n, a, b, n, beta);
}
void klogis(float *x, unsigned int n, int sign) {
  CALL_KERNEL(klogis, n, x, n, sign);
}


void synth_mbsd(const float *in, int iw, int ih, float *out, int ic, int bn) {
  long pn = (long)iw * ih;
  long pc = pn * ic;
  float *ksig; kmake(&ksig, pc);
  CALL_KERNEL(mbsd_stat, pc, in, bn, pc, ksig);
  float s = ksum(ksig, pc) / (float)pc;
  kfree(ksig);
  CALL_KERNEL(mbsd_fill, (long)bn * pn * (ic + 1), in, bn, pn, ic, out, s);
}

void learn_mbsd(float *fin, int iw, int ih, const float *fout, int ic, int bn) {
  long pn = (long)iw * ih;
  long n = (long)bn * pn;
  float *tmp; kmake(&tmp, n);
  CALL_KERNEL(mbsd_extra, n, fout, pn, bn, ic, tmp);
  float gs = ksum(tmp, n);
  kfree(tmp);
  // the learn kernel reads other samples' inputs while writing its own gradient:
  // work from a copy of the inputs, write the gradient into fin
  float *x; kmake(&x, n * ic);
  CALL_KERNEL(kcopyn, n * ic, fin, n * ic, x);
  CALL_KERNEL(mbsd_learn, n * ic, x, bn, pn, ic, fout, gs);
  CALL_KERNEL(kcopyn, n * ic, x, n * ic, fin);
  kfree(x);
}


int size_styl(int ic, int sdim) {
  return 3 * (2 * ic + 2 * ic * sdim);
}

void synth_styl(const float *in, int iw, int ih, float *out, int ic, int bn, const float *wmv, const float *kstyle, int sdim, float wscale) {
  long pn = (long)iw * ih;
  float *ksc; kmake(&ksc, (long)bn * 2 * ic);
  if (kdev) gpu_styl_affine(wmv, kstyle, bn, ic, sdim, wscale, ksc); else CALL_KERNEL(styl_affine, (long)bn * 2 * ic, wmv, kstyle, bn, ic, sdim, wscale, ksc);
  CALL_KERNEL(styl_apply, (long)bn * pn * ic, in, pn, ic, ksc, out);
  kfree(ksc);
}

void learn_styl(float *fin, int iw, int ih, const float *fout, int ic, int bn, float *wmv, float *dwv,
  const float *kstyle, float *kstylegrad, int sdim, float wscale,
  float nu, float b1, float b2, float eps, float clip, float rounds) {
  long pn = (long)iw * ih;
  int wn = 2 * ic + 2 * ic * sdim;
  float *ksc; kmake(&ksc, (long)bn * 2 * ic);
  float *kd; kmake(&kd, (long)bn * 2 * ic);
  if (kdev) gpu_styl_affine(wmv, kstyle, bn, ic, sdim, wscale, ksc); else CALL_KERNEL(styl_affine, (long)bn * 2 * ic, wmv, kstyle, bn, ic, sdim, wscale, ksc);
  if (kdev) gpu_styl_dsc(fin, fout, pn, ic, bn, kd); else CALL_KERNEL(styl_dsc, (long)bn * 2 * ic, fin, fout, pn, ic, kd);
  if (dwv)
    CALL_KERNEL(styl_dw, (long)wn, kd, kstyle, bn, ic, sdim, wscale, dwv);
  if (kstylegrad)
    CALL_KERNEL(styl_ds, (long)bn * sdim, kd, wmv, bn, ic, sdim, wscale, kstylegrad);
  CALL_KERNEL(styl_din, (long)bn * pn * ic, fin, fout, pn, ic, ksc);
  kfree(kd);
  kfree(ksc);
  float mc = 1.0, vc = 1.0;
  if (rounds < 1024.0) {
    double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
    if (dm > 1e-12) mc = 1.0 / dm;
    if (dv > 1e-12) vc = 1.0 / dv;
  }
  CALL_KERNEL(learn_adam, (long)wn, wmv, dwv, wn, 2 * ic, nu, b1, b2, eps, clip, mc, vc);
}


int size_mcon(int ic, int oc, int sdim) {
  return 3 * (oc + oc * ic * 9 + ic + ic * sdim);
}

// forward; `stash` (bn*pn*oc) keeps the output for the backward pass
void synth_mcon(const float *in, int iw, int ih, float *out, int ic, int oc, int bn,
  const float *wmv, const float *kstyle, int sdim, float wscale, float wscaleA, float *stash) {
  long pn = (long)iw * ih;
  long offab = oc + 9L * oc * ic, offA = offab + ic;
  float *ks, *w2, *kd, *xs;
  kmake(&ks, (long)bn * ic); kmake(&w2, (long)oc * ic); kmake(&kd, (long)bn * oc); kmake(&xs, (long)bn * pn * ic);
  CALL_KERNEL(mcon_affine, (long)bn * ic, wmv, offab, offA, kstyle, bn, ic, sdim, wscaleA, ks);
  CALL_KERNEL(mcon_w2, (long)oc * ic, wmv, ic, oc, wscale, w2);
  CALL_KERNEL(mcon_demod, (long)bn * oc, w2, ks, bn, ic, oc, kd);
  CALL_KERNEL(mcon_scale_in, (long)bn * pn * ic, in, ks, pn, ic, xs);
  // the raw conv output y goes to the stash (the backward needs it: dout/dd = -y/d^2), out = y / d
  synth_conv(xs, iw, ih, stash, 1, ic, oc, false, 0, 2, wmv, bn, wscale);
  CALL_KERNEL(mcon_divc, (long)bn * pn * oc, stash, kd, pn, oc, out);
  kfree(xs); kfree(kd); kfree(w2); kfree(ks);
}

void learn_mcon(float *fin, int iw, int ih, float *fout, int ic, int oc, int bn,
  float *wmv, float *dwv, const float *kstyle, float *kstylegrad, int sdim, float wscale, float wscaleA, const float *stash,
  float nu, float b1, float b2, float eps, float clip, float rounds) {
  long pn = (long)iw * ih;
  long offab = oc + 9L * oc * ic, offA = offab + ic;
  int wn = oc + oc * ic * 9 + ic + ic * sdim;
  float *ks, *w2, *kd, *xs, *kgd, *kgs;
  kmake(&ks, (long)bn * ic); kmake(&w2, (long)oc * ic); kmake(&kd, (long)bn * oc);
  kmake(&xs, (long)bn * pn * ic); kmake(&kgd, (long)bn * oc); kmake(&kgs, (long)bn * ic);
  CALL_KERNEL(mcon_affine, (long)bn * ic, wmv, offab, offA, kstyle, bn, ic, sdim, wscaleA, ks);
  CALL_KERNEL(mcon_w2, (long)oc * ic, wmv, ic, oc, wscale, w2);
  CALL_KERNEL(mcon_demod, (long)bn * oc, w2, ks, bn, ic, oc, kd);
  if (kdev) { gpu_dot_bc(fout, stash, bn, pn, oc, kgd); CALL_KERNEL(mcon_gd_fin, (long)bn * oc, kgd, kd); }
  else CALL_KERNEL(mcon_gd, (long)bn * oc, fout, stash, kd, pn, oc, kgd);
  CALL_KERNEL(mcon_div_out, (long)bn * pn * oc, fout, kd, pn, oc);          // fout := g_y
  CALL_KERNEL(mcon_scale_in, (long)bn * pn * ic, fin, ks, pn, ic, xs);         // xs := x'
  // conv weight gradient into dwv (no update: nu = 0) and input gradient into xs
  learn_conv(xs, iw, ih, fout, 1, ic, oc, false, 0, 2, wmv, dwv, 0.0f, b1, b2, eps, clip, rounds, bn, wscale);
  if (kdev) { gpu_dot_bc(xs, fin, bn, pn, ic, kgs); CALL_KERNEL(mcon_gs_fin, (long)bn * ic, kgs, ks, kgd, kd, w2, ic, oc); }
  else CALL_KERNEL(mcon_gs, (long)bn * ic, xs, fin, ks, kgd, kd, w2, pn, ic, oc, kgs);
  CALL_KERNEL(mcon_gx, (long)bn * pn * ic, xs, ks, pn, ic, fin);               // fin := g_x
  if (dwv) {
    CALL_KERNEL(mcon_dw_demod, 9L * oc * ic, wmv, ks, kgd, kd, bn, ic, oc, wscale, dwv);
    CALL_KERNEL(mcon_daffine, (long)ic + (long)ic * sdim, kgs, kstyle, bn, ic, sdim, wscaleA, offab, offA, dwv);
  }
  if (kstylegrad)
    CALL_KERNEL(mcon_dstyle, (long)bn * sdim, kgs, wmv, offA, bn, ic, sdim, wscaleA, kstylegrad);
  kfree(kgs); kfree(kgd); kfree(xs); kfree(kd); kfree(w2); kfree(ks);
  if (dwv) {
    float mc = 1.0, vc = 1.0;
    if (rounds < 1024.0) {
      double dm = 1.0 - pow(1.0 - b1, 1.0 + rounds), dv = 1.0 - pow(1.0 - b2, 1.0 + rounds);
      if (dm > 1e-12) mc = 1.0 / dm;
      if (dv > 1e-12) vc = 1.0 / dv;
    }
    CALL_KERNEL(learn_adam, (long)wn, wmv, dwv, wn, oc, nu, b1, b2, eps, clip, mc, vc);
  }
}
