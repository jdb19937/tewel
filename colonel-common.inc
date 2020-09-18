void kaddvec(const double *a, const double *b, unsigned int n, double *c) {
  CALL_KERNEL(kaddvec, n, a, b, n, c);
}

void ksubvec(const double *a, const double *b, unsigned int n, double *c) {
  CALL_KERNEL(ksubvec, n, a, b, n, c);
}

void kmul(const double *a, const double b, unsigned int n, double *c) {
  CALL_KERNEL(kmul, n, a, b, n, c);
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

void ksplice2(
  const double *x, const double *z, double l, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  CALL_KERNEL(ksplice2, n * xk, x, z, l, n, xm, xa, xk, y, ym, ya);
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
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  int f = d * 2 + 1;
  int outn = iw * ih * oc;
  int wn = (outn + ic * f * f * outn);
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_local1, wn, fin, iw, ih, fout, d, ic, oc, wmv, nu, b1, b2);
  CALL_KERNEL(learn_local2, inn, fin, iw, ih, fout, d, ic, oc, wmv);
  CALL_KERNEL(learn_local3, wn, iw, ih, d, ic, oc, wmv, nu, b1, b2, eps, clip, rounds);
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

void synth_trim(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic
) {
  int ow = iw;
  int oh = ih;
  int oc = ic;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_trim, outn,
    in, iw, ih, out, d, ic
  );
}


void synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  const double *wmv
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_conv, outn,
    in, iw, ih, out, d, ic, oc, relu, freeze, dim, wmv
  );
}

void learn_trim(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_trim, inn, fin, iw, ih, fout, d, ic);
}


void learn_conv(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,
  bool relu, int freeze, int dim,

  double *wmv,
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_conv1, wn, fin, iw, ih, fout, d, ic, oc, relu, freeze, dim, wmv, nu, b1, b2);
  CALL_KERNEL(learn_conv2, inn, fin, iw, ih, fout, d, ic, oc, relu, freeze, dim, wmv);
  CALL_KERNEL(learn_conv3, wn, d, ic, oc, dim, wmv, nu, b1, b2, eps, clip, rounds);
}

void synth_addgeo(
  const double *in, int iw, int ih,
  double *out,
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
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_addgeo, inn,
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


void synth_smax(
  double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_smax, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_smax(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_smax, inn,
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

void synth_sum(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;

  CALL_KERNEL(synth_sum, outn,
    in, iw, ih, out, ic, oc, kbuf
  );
}


void learn_sum(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_sum, inn,
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

void synth_clamp(
  const double *in, int iw, int ih,
  double *out,
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
  double *fin, int iw, int ih,
  const double *fout,

  int ic
) {
  int inn = iw * ih * ic;

  CALL_KERNEL(learn_clamp, inn,
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
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
) {
  int inn = iw * ih * ic;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(learn_upscale, inn, fin, iw, ih, fout, s, ic, oc, dim);
}




void synth_downscale(
  const double *in, int iw, int ih,
  double *out,
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
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
) {
  int inn = iw * ih * ic;

  assert(dim == 1 || dim == 2);

  CALL_KERNEL(learn_downscale, inn, fin, iw, ih, fout, s, ic, oc, dim);
}

