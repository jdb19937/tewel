DEFN_KERNEL(kaddvec, const double *a, const double *b, unsigned int n, double *c) {
  PREF_KERNEL;
  c[i] = a[i] + b[i];
}

DEFN_KERNEL(ksubvec, const double *a, const double *b, unsigned int n, double *c) {
  PREF_KERNEL;
  c[i] = a[i] - b[i];
}

DEFN_KERNEL(ksumsq, const double *a, unsigned int n, double *sumsqp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);
  
  double s = 0;
  for (unsigned int j = j0; j < j1; ++j)
    s += a[j] * a[j];

  sumsqp[i] = s;
}

DEFN_KERNEL(kmaxabs, const double *a, unsigned int n, double *maxp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);

  unsigned int j = j0;
  double s = fabs(a[j]);
  ++j;
  
  for (; j < j1; ++j) {
    double aa = fabs(a[j]);
    if (aa > s)
      s = aa;
  }

  maxp[i] = s;
}

DEFN_KERNEL(kspliceadd,
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] += x[in * xm + xa + ixk];
}

DEFN_KERNEL(ksplice,
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] = x[in * xm + xa + ixk];
}

DEFN_KERNEL(synth_conv,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv,
  bool stripped
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
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
        double w = stripped ? wmv[wi] : wmv[wi * 3];

        outv += v * w;
      }
    }
  }

  outv += stripped ? wmv[oz] : wmv[oz * 3];

  out[outi] = outv;
}


DEFN_KERNEL(learn_conv1,
  const double *in, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *wmv,
  double nu, double b1, double b2
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);

  int wi = i;
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


DEFN_KERNEL(learn_conv2,
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  const double *wmv
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
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


DEFN_KERNEL(learn_conv3,
  int d, int ic, int oc,
  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int f = d * 2 + 1;
  int wn = (oc + ic * f * f * oc);

  int wi = i;
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

DEFN_KERNEL(synth_pad,
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
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

DEFN_KERNEL(learn_pad,
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
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

DEFN_KERNEL(synth_relu,
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int ini = outi;
  if (in[ini] > 0)
    out[outi] = in[ini];
  else
    out[outi] = 0;
}

DEFN_KERNEL(learn_relu,
  double *fin, int iw, int ih,
  const double *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;

  if (fin[ini] > 0) {
    fin[ini] = fout[outi];
  } else {
    fin[ini] = 0;
  }
}

DEFN_KERNEL(synth_abs,
  const double *in, int iw, int ih,
  double *out,
  int ic
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int ini = outi;
  if (in[ini] > 0)
    out[outi] = in[ini];
  else
    out[outi] = -in[ini];
}

DEFN_KERNEL(learn_abs,
  double *fin, int iw, int ih,
  const double *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;

  if (fin[ini] > 0) {
    fin[ini] = fout[outi];
  } else {
    fin[ini] = -fout[outi];
  }
}


DEFN_KERNEL(synth_upscale,
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  PREF_KERNEL;

  int ow = (iw << s);
  int oh = (ih << s);

  int outn = ow * oh * oc;
  int outi = i;
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


DEFN_KERNEL(learn_upscale,
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
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


DEFN_KERNEL(synth_downscale,
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
) {
  PREF_KERNEL;

  int ow = (iw >> s);
  int oh = (ih >> s);

  int outn = ow * oh * oc;
  int outi = i;
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


DEFN_KERNEL(learn_downscale,
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
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

