
DEFN_KERNEL(kaddvec, const double *a, const double *b, unsigned int n, double *c) {
  PREF_KERNEL;
  c[i] = a[i] + b[i];
}

DEFN_KERNEL(ksubvec, const double *a, const double *b, unsigned int n, double *c) {
  PREF_KERNEL;
  c[i] = a[i] - b[i];
}

DEFN_KERNEL(kmul, const double *a, const double b, unsigned int n, double *c) {
  PREF_KERNEL;
  c[i] = a[i] * b;
}

DEFN_KERNEL(krectify, const double *a, double *b, unsigned int n) {
  PREF_KERNEL;
  if (a[i] <= 0.0)
    b[i] = 0.0;
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

DEFN_KERNEL(ksum, const double *a, unsigned int n, double *sumsqp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);
  
  double s = 0;
  for (unsigned int j = j0; j < j1; ++j)
    s += a[j];

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

DEFN_KERNEL(ksplice2,
  const double *x, const double *z, double l, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] =
    (1.0 - l) * x[in * xm + xa + ixk] +
    l * z[in * xm + xa + ixk];
}

DEFN_KERNEL(synth_norm,
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int ini = outi;

  double outv = 0;
  outv += in[ini] * (1.0 + wmv[(outi + outn) * 3]);
  outv += wmv[outi * 3];

  out[outi] = outv;
}


DEFN_KERNEL(learn_norm1,
  const double *in, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *dwv,
  double nu, double b1, double b2
) {
  PREF_KERNEL;

  if (!dwv)
    return;

  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
  int wn = outn * 2;

  int wi = i;
  if (wi >= wn)
    return;

  double dw = 0;

  if (wi < outn) {
    int outi = wi;
    dw += fout[outi];
  } else {
#if 0
    int outi = wi - outn;
    dw += fout[outi] * (1.0 + wmv[wi * 3]);
#endif
    dw = 0;
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_norm2,
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,

  const double *wmv
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  fin[ini] = fout[ini];
}


DEFN_KERNEL(learn_norm3,
  int iw, int ih,
  int ic, int oc,
  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int outn = iw * ih * oc;
  int wn = outn * 2;

  int wi = i;
  if (wi >= wn)
    return;

  double dw = dwv[wi];

  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wi * 3 + 1] = m;
  wmv[wi * 3 + 2] = v;

  double w = wmv[wi * 3 + 0];

  if (rounds < 1024.0) {
    m = m / (1.0 - pow(1.0 - b1, 1.0 + rounds));
    v = v / (1.0 - pow(1.0 - b2, 1.0 + rounds));
  }

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi * 3 + 0] = w;
  dwv[wi] = 0;
}



DEFN_KERNEL(synth_bias,
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;


  int ini = outi;

  double outv = 0;
  outv += in[ini];
  outv += wmv[outi * 3];

  out[outi] = outv;
}


DEFN_KERNEL(learn_bias1,
  const double *in, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *dwv,
  double nu, double b1, double b2
) {
  PREF_KERNEL;

  if (!dwv)
    return;

  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
  int wn = outn;

  int wi = i;
  if (wi >= wn)
    return;

  double dw = 0;

  if (wi < outn) {
    int outi = wi;
    dw += fout[outi];
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_bias2,
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,
  const double *wmv
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  fin[ini] = fout[ini];
}


DEFN_KERNEL(learn_bias3,
  int iw, int ih,
  int ic, int oc,
  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int outn = iw * ih * oc;
  int wn = outn;

  int wi = i;
  if (wi >= wn)
    return;

  double dw = dwv[wi];

  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wi * 3 + 1] = m;
  wmv[wi * 3 + 2] = v;

  double w = wmv[wi * 3 + 0];

  if (rounds < 1024.0) {
    m = m / (1.0 - pow(1.0 - b1, 1.0 + rounds));
    v = v / (1.0 - pow(1.0 - b2, 1.0 + rounds));
  }

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi * 3 + 0] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_local,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;


  int tmp = outi;
  tmp /= oc;
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
        int wi = outn + (iz + ic * (dx + f * (dy + f * outi)));

        double v = in[ini];
        double w = wmv[wi * 3];

        outv += v * w;
      }
    }
  }

  outv += wmv[outi * 3];

  out[outi] = outv;
}


DEFN_KERNEL(learn_local1,
  const double *in, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *dwv,
  double nu, double b1, double b2
) {
  PREF_KERNEL;

  if (!dwv)
    return;

  int ow = iw;
  int oh = ih;
  int outn = ow * oh * oc;
  int f = d * 2 + 1;
  int wn = (outn + ic * f * f * outn);

  int wi = i;
  if (wi >= wn)
    return;

  double dw = 0;

  if (wi < outn) {
    int outi = wi;
    dw += fout[outi];
  } else {
    int tmp = wi - outn;
    int iz = tmp % ic; tmp /= ic;
    int dx = tmp % f; tmp /= f;
    int dy = tmp % f; tmp /= f;
    int outi = tmp % outn;

    tmp = outi;
    tmp /= oc;
    int ox = tmp % ow; tmp /= ow;
    int oy = tmp % oh;

    int ix = (ox - d + dx + iw) % iw;
    int iy = (oy - d + dy + ih) % ih;

    int ini = iz + ic * (ix + iw * iy);

    dw += fout[outi] * in[ini];
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_local2,
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
  int outn = ow * oh * oc;

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
        int wi = outn + (iz + ic * (dx + f * (dy + f * outi)));

        double v = fout[outi];
        double w = wmv[wi * 3];
        finv += v * w;
      }
    }
  }

  fin[ini] = finv;
}


DEFN_KERNEL(learn_local3,
  int iw, int ih,
  int d, int ic, int oc,
  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int outn = iw * ih * oc;
  int f = d * 2 + 1;
  int wn = (outn + ic * f * f * outn);

  int wi = i;
  if (wi >= wn)
    return;

  double dw = dwv[wi];

  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wi * 3 + 1] = m;
  wmv[wi * 3 + 2] = v;

  double w = wmv[wi * 3 + 0];

  if (rounds < 1024.0) {
    m = m / (1.0 - pow(1.0 - b1, 1.0 + rounds));
    v = v / (1.0 - pow(1.0 - b2, 1.0 + rounds));
  }

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi * 3 + 0] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_padd,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic
) {
  PREF_KERNEL;

  int ow = iw + 2 * d;
  int oh = ih + 2 * d;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int iz = oz;
  int ix = ox - d;
  int iy = oy - d;

  if (ix < 0 || ix >= iw || iy < 0 || iy >= ih) {
    out[outi] = 0;
    return;
  }

  int ini = iz + ic * (ix + iw * iy);

  out[outi] = in[ini];
}

DEFN_KERNEL(synth_crop,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic,
  int ix0, int iy0
) {
  PREF_KERNEL;

  int ow = d;
  int oh = d;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int iz = oz;
  int ix = ix0 + ox;
  int iy = iy0 + oy;

  int ini = iz + ic * (ix + iw * iy);

  out[outi] = in[ini];
}

DEFN_KERNEL(synth_trim,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic
) {
  PREF_KERNEL;

  int ow = iw - 2 * d;
  int oh = ih - 2 * d;
  int oc = ic;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int tmp = outi;
  int oz = tmp % oc; tmp /= oc;
  int ox = tmp % ow; tmp /= ow;
  int oy = tmp;

  int iz = oz;
  int ix = ox + d;
  int iy = oy + d;

  int ini = iz + ic * (ix + iw * iy);

  out[outi] = in[ini];
}


DEFN_KERNEL(synth_conv,
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  bool relu, int freeze,
  int dim,
  const double *wmv
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

  if (oz < freeze) {
    int ix = ox;
    int iy = oy;
    int iz = oz;
    int ini = iz + ic * (ix + iw * iy);
    double v = in[ini];

    out[outi] = v;
    return;
  }

  int ix0 = ox - d;
  int ix1 = ox + d;
  double outv = 0;
  int f = d * 2 + 1;

  if (dim == 1) {
    // int wmvn = 3 * (oc + ic * f * f * oc);

    int iy = oy;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;
      for (int iz = 0; iz < ic; ++iz) {
        int ini = iz + ic * (ix + iw * iy);
        double v = in[ini];

        if (!relu || v > 0 || iz < freeze) {
          int dx = ixt - ix0;
          int wi = oc + (iz + ic * (dx + f * oz));

          double w = wmv[wi * 3];

          outv += v * w;
        }
      }
    }
  } else {
    int iy0 = oy - d;
    int iy1 = oy + d;
    // int wmvn = 3 * (oc + ic * f * f * oc);

    for (int iyt = iy0; iyt <= iy1; ++iyt) {
      int iy = (iyt + ih) % ih;
      for (int ixt = ix0; ixt <= ix1; ++ixt) { 
        int ix = (ixt + iw) % iw;
        for (int iz = 0; iz < ic; ++iz) {
          int ini = iz + ic * (ix + iw * iy);
          double v = in[ini];

          if (!relu || v > 0 || iz < freeze) {
            int dx = ixt - ix0;
            int dy = iyt - iy0;
            int wi = oc + (iz + ic * (dx + f * (dy + f * oz)));

            double w = wmv[wi * 3];
  
            outv += v * w;
          }
        }
      }
    }
  }

  outv += wmv[oz * 3];

  out[outi] = outv;
}

DEFN_KERNEL(learn_padd,
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int ow = iw + 2 * d;
  // int oh = ih + 2 * d;
  int oc = ic;

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int oz = iz;
  int ox = ix + d;
  int oy = iy + d;
  int outi = oz + oc * (ox + ow * oy);

  fin[ini] = fout[outi];
}

DEFN_KERNEL(learn_trim,
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int ow = iw - 2 * d;
  // int oh = ih - 2 * d;
  int oc = ic;

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  if (ix < d || ix >= iw - d) {
    fin[ini] = 0;
    return;
  }
  if (iy < d || iy >= ih - d) {
    fin[ini] = 0;
    return;
  }

  int oz = iz;
  int ox = ix - d;
  int oy = iy - d;
  int outi = oz + oc * (ox + ow * oy);

  fin[ini] = fout[outi];
}


DEFN_KERNEL(learn_conv1,
  const double *in, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,
  bool relu,

  int freeze,
  int dim,
  double *dwv,
  double nu, double b1, double b2
) {
  PREF_KERNEL;

  if (!dwv)
    return;

  int f = d * 2 + 1;
  int wn;
  if (dim == 1) {
    wn = (oc + ic * f * oc);
  } else {
    wn = (oc + ic * f * f * oc);
  }

  int wi = i;
  if (wi >= wn)
    return;

  int ow = iw;
  int oh = ih;

  double dw = 0;

  if (wi < oc) {
    int oz = wi;

    if (oz < freeze) {
      dw = 0;
    } else {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          int outi = oz + oc * (ox + ow * oy);
          dw += fout[outi];
        }
      }
    }
  } else {
    if (dim == 1) {
      int tmp = wi - oc;
      int iz = tmp % ic; tmp /= ic;
      int dx = tmp % f; tmp /= f;
      int oz = tmp % oc;

      if (oz < freeze) {
        dw = 0;
      } else {
        for (int oy = 0; oy < oh; ++oy) {
          for (int ox = 0; ox < ow; ++ox) {
            int ix = (ox - d + dx + iw) % iw;
            int iy = oy;
  
            int ini = iz + ic * (ix + iw * iy);
            double v = in[ini];
            if (!relu || v > 0 || iz < freeze) {
              int outi = oz + oc * (ox + ow * oy);
              dw += fout[outi] * v;
            }
          }
        }
      }
    } else {
      int tmp = wi - oc;
      int iz = tmp % ic; tmp /= ic;
      int dx = tmp % f; tmp /= f;
      int dy = tmp % f; tmp /= f;
      int oz = tmp % oc;

      if (oz < freeze) {
        dw = 0;
      } else {
        for (int oy = 0; oy < oh; ++oy) {
          for (int ox = 0; ox < ow; ++ox) {
            int ix = (ox - d + dx + iw) % iw;
            int iy = (oy - d + dy + ih) % ih;
    
            int ini = iz + ic * (ix + iw * iy);
            double v = in[ini];
            if (!relu || v > 0 || iz < freeze) {
              int outi = oz + oc * (ox + ow * oy);
              dw += fout[outi] * v;
            }
          }
        }
      }
    }
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_conv2,
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,
  bool relu, int freeze,
  int dim,

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

  if (relu && fin[ini] <= 0 && iz >= freeze) {
    fin[ini] = 0;
    return;
  }

  int ox0 = ix - d;
  int ox1 = ix + d;
  int f = d * 2 + 1;
  double finv = 0;

  if (dim == 1) {
    int oy = iy;
    for (int oxt = ox0; oxt <= ox1; ++oxt) { 
      int ox = (oxt + ow) % ow;
      for (int oz = 0; oz < oc; ++oz) {
        int outi = oz + oc * (ox + ow * oy);

        double v = fout[outi];
        double w;

        if (oz < freeze) {
          if (ix == ox && iy == oy && iz == oz) {
            w = 1.0;
          } else {
            w = 0.0;
          }
        } else {
          int dx = ix - oxt + d;
          int wi = oc + (iz + ic * (dx + f * oz));
          w = wmv[wi * 3];
        }

        finv += v * w;
      }
    }
  } else {
    int oy0 = iy - d;
    int oy1 = iy + d;
    // int wmvn = 3 * (oc + ic * f * f * oc);

    for (int oyt = oy0; oyt <= oy1; ++oyt) {
      int oy = (oyt + oh) % oh;
      for (int oxt = ox0; oxt <= ox1; ++oxt) { 
        int ox = (oxt + ow) % ow;
        for (int oz = 0; oz < oc; ++oz) {
          int outi = oz + oc * (ox + ow * oy);

          double v = fout[outi];
          double w;

          if (oz < freeze) {
            if (ix == ox && iy == oy && iz == oz) {
              w = 1.0;
            } else {
              w = 0.0;
            }
          } else {
            int dx = ix - oxt + d;
            int dy = iy - oyt + d;
            int wi = oc + (iz + ic * (dx + f * (dy + f * oz)));
  
            w = wmv[wi * 3];
          }

          finv += v * w;
        }
      }
    }
  }

  fin[ini] = finv;
}


DEFN_KERNEL(learn_conv3,
  int d, int ic, int oc, int dim,
  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int f = d * 2 + 1;
  int wn;
  if (dim == 1) {
    wn = (oc + ic * f * oc);
  } else {
    wn = (oc + ic * f * f * oc);
  }

  int wi = i;
  if (wi >= wn)
    return;

  double dw = dwv[wi];

  double m = wmv[wi * 3 + 1];
  double v = wmv[wi * 3 + 2];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wi * 3 + 1] = m;
  wmv[wi * 3 + 2] = v;

  double w = wmv[wi * 3 + 0];

  if (rounds < 1024.0) {
    m = m / (1.0 - pow(1.0 - b1, 1.0 + rounds));
    v = v / (1.0 - pow(1.0 - b2, 1.0 + rounds));
  }

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi * 3 + 0] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_addgeo,
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc
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
  } else if (oz == ic) {
    out[outi] = (double)ox / (double)ow;
  } else if (oz == ic + 1) {
    out[outi] = (double)oy / (double)oh;
  } else if (oz == ic + 2) {
    out[outi] = 1.0 - 2.0 * fabs(0.5 - (double)ox / (double)ow);
  } else if (oz == ic + 3) {
    out[outi] = 1.0 - 2.0 * fabs(0.5 - (double)oy / (double)oh);
  } else if (oz == ic + 4) {
    out[outi] = (log((double)ow) / log(2.0)) / 16.0;
  } else {
    out[outi] = 0;
  }
}

DEFN_KERNEL(learn_addgeo,
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

DEFN_KERNEL(synth_zero,
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

  out[outi] = 0;
}

DEFN_KERNEL(learn_zero,
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  fin[ini] = 0;
}

DEFN_KERNEL(synth_iden,
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
  int iz = oz % ic;

  int ini = iz + ic * (ix + iw * iy);
  out[outi] = in[ini];
}

DEFN_KERNEL(learn_iden,
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

  int ox = ix;
  int oy = iy;

  double v = 0;
  for (int oz = iz; oz < oc; oz += ic) {
    int outi = oz + oc * (ox + ow * oy);
    v += fout[outi];
  }

  fin[ini] = v;
}

DEFN_KERNEL(synth_smax,
  double *in, int iw, int ih,
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

  double v = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    if (iz == oz || in[ini] > v)
      v = in[ini];
  }

  unsigned int nmax = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    if (in[ini] >= v)
      ++nmax;
  }

  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    in[ini] = (in[ini] >= v) ? 1.0 / (double)nmax : 0.0;
  }

  out[outi] = v;
}

DEFN_KERNEL(learn_smax,
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

  int oz = iz % oc;
  int ox = ix;
  int oy = iy;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi] * fin[ini];
}

DEFN_KERNEL(synth_mean,
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

  double v = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    v += in[ini];
  }
  v *= (double)oc / (double)ic;
  out[outi] = v;
}

DEFN_KERNEL(learn_mean,
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

  int oz = iz % oc;
  int ox = ix;
  int oy = iy;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi] * (double)oc / (double)ic;
}

DEFN_KERNEL(synth_median,
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

  int ix0 = ox - 1;
  int iy0 = oy - 1;
  int ix1 = ox + 1;
  int iy1 = oy + 1;
  int iz = oz;

  int ar[9];
  int k = 0;

  for (int iyt = iy0; iyt <= iy1; ++iyt) {
    int iy = (iyt + ih) % ih;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;

      int ini = iz + ic * (ix + iw * iy);
      ar[k++] = ini;
    }
  }

  int sw;
  do {
    sw = 0;
    for (k = 1; k < 9; ++k) {
      if (in[ar[k - 1]] > in[ar[k]]) {
        int q = ar[k - 1];
        ar[k - 1] = ar[k];
        ar[k] = q;
        ++sw;
      }
    }
  } while (sw > 0);

  out[outi] = in[ar[4]];
}

DEFN_KERNEL(learn_median,
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int ow = iw;
  int oh = ih;

  int ox0 = ix - 1;
  int oy0 = iy - 1;
  int ox1 = ix + 1;
  int oy1 = iy + 1;
  int oz = iz;

  double dv = 0;

  for (int oyt = oy0; oyt <= oy1; ++oyt) {
    int oy = (oyt + oh) % oh;
    for (int oxt = ox0; oxt <= ox1; ++oxt) { 
      int ox = (oxt + ow) % iw;


      int ix0 = ox - 1;
      int iy0 = oy - 1;
      int ix1 = ox + 1;
      int iy1 = oy + 1;

      int ar[9];
      int k = 0;

      for (int iyt = iy0; iyt <= iy1; ++iyt) {
        int jy = (iyt + ih) % ih;
        for (int ixt = ix0; ixt <= ix1; ++ixt) { 
          int jx = (ixt + iw) % iw;

          int ini = iz + ic * (jx + iw * jy);
          ar[k++] = ini;
        }
      }

      int sw;
      do {
        sw = 0;
        for (k = 1; k < 9; ++k) {
          if (fin[ar[k - 1]] > fin[ar[k]]) {
            int q = ar[k - 1];
            ar[k - 1] = ar[k];
            ar[k] = q;
            ++sw;
          }
        }
      } while (sw > 0);

      if (ini == ar[4]) {
        int outi = oz + oc * (ox + ow * oy);
        dv += fout[outi];
      }
    }
  }

  syncthreads();

  fin[ini] = dv;
}

DEFN_KERNEL(synth_blur,
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

  int ix0 = ox - 1;
  int iy0 = oy - 1;
  int ix1 = ox + 1;
  int iy1 = oy + 1;
  int iz = oz;

  double v = 0;

  for (int iyt = iy0; iyt <= iy1; ++iyt) {
    int iy = (iyt + ih) % ih;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;

      double w = 0.25;
      if (ix != ox) 
        w *= 0.5;
      if (iy != oy) 
        w *= 0.5;

      int ini = iz + ic * (ix + iw * iy);
      v += in[ini] * w;
    }
  }

  out[outi] = v;
}

DEFN_KERNEL(synth_diff,
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

  int iz = oz;
  int ini = iz + ic * (ix + iw * iy);
  double v = in[ini];

  iz += oc;
  ini = iz + ic * (ix + iw * iy);
  v -= in[ini];

  out[outi] = v;
}

DEFN_KERNEL(learn_diff,
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

  int oz = iz % oc;
  int ox = ix;
  int oy = iy;

  int q = (iz >= oc);
  double sgn = (q % 2) ? -1.0 : 1.0;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi] * sgn;
}

DEFN_KERNEL(synth_sum,
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

  double v = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    v += in[ini];
  }
  out[outi] = v;
}

DEFN_KERNEL(learn_sum,
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

  int oz = iz % oc;
  int ox = ix;
  int oy = iy;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi];
}

DEFN_KERNEL(synth_sigm,
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
  out[outi] = 1.0 / (1.0 + exp(-in[ini]));
}

DEFN_KERNEL(learn_sigm,
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
  double outv = 1.0 / (1.0 + exp(-fin[ini]));
  fin[ini] = fout[outi] * outv * (1.0 - outv);
}

DEFN_KERNEL(synth_inrf,
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
  double v = in[ini];

  if (v < 1.0/256.0)
    v = 1.0/256.0;
  if (v >= 255.0/256.0)
    v = 255.0/256.0;

  out[outi] = sqrt(2.0) * erfinv(-1.0 + 2.0 * v);
}

DEFN_KERNEL(learn_inrf,
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

  const double sq2pi = 2.506628275;
  fin[ini] = fout[outi] * sq2pi * exp(fout[outi] * fout[outi] * 2.0);
}

DEFN_KERNEL(synth_nerf,
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

  const double isq2 = 0.7071067812;

  int ini = outi;

  out[outi] = (erf(in[ini] * isq2) + 1.0) / 2.0;
}

DEFN_KERNEL(learn_nerf,
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

  const double isq2pi = 0.3989422804;
  fin[ini] = fout[outi] * isq2pi * exp(-fin[ini] * fin[ini] / 2.0);
}


DEFN_KERNEL(synth_clamp,
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
  if (in[ini] > 0) {
    if (in[ini] < 1)
      out[outi] = in[ini];
    else
      out[outi] = 1;
  } else
    out[outi] = 0;
}

DEFN_KERNEL(learn_clamp,
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

  if (fin[ini] > 0 && fin[ini] <= 1) {
    fin[ini] = fout[outi];
  } else {
    fin[ini] = 0;
  }
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
  int s, int ic, int oc,
  int dim
) {
  PREF_KERNEL;

  int ow = (iw << s);

  if (dim == 1) {
    int oh = ih;

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
    int iy = oy;

    int f = (1 << s);
    int dx = ox % f;

    int iz = oz + oc * dx;
    int ini = iz + ic * (ix + iw * iy);

    out[outi] = in[ini];
  } else {
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
}


DEFN_KERNEL(learn_upscale,
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int ow = (iw << s);
  
  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int f = (1 << s);
  tmp = iz;
  int oz = tmp % oc; tmp /= oc;
  int dx = tmp % f; tmp /= f;
  int ox = (ix << s) + dx;

  if (dim == 1) {
    int oy = iy;
    int outi = oz + oc * (ox + ow * oy);
    fin[ini] = fout[outi];
  } else {
    int dy = tmp;
    int oy = (iy << s) + dy;
    int outi = oz + oc * (ox + ow * oy);
    fin[ini] = fout[outi];
  }
}


DEFN_KERNEL(synth_downscale,
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc, int dim
) {
  PREF_KERNEL;

  if (dim == 1) {
    int ow = (iw >> s);
    int oh = ih;
  
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
    int dx = tmp % f;
  
    int ix = (ox << s) + dx;
    int iy = oy;

    int ini = iz + ic * (ix + iw * iy);

    out[outi] = in[ini];
  } else {
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
}


DEFN_KERNEL(learn_downscale,
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int ow = (iw >> s);
  
  int tmp = ini;
  int iz = tmp % ic; tmp /= ic;
  int ix = tmp % iw; tmp /= iw;
  int iy = tmp;

  int f = (1 << s);
  int ox = (ix >> s);

  if (dim == 1) {
    int oy = iy;
  
    int dx = ix % f;
    int oz = iz + ic * dx;
  
    int outi = oz + oc * (ox + ow * oy);
  
    fin[ini] = fout[outi];
  } else {
    int oy = (iy >> s);
  
    int dx = ix % f;
    int dy = iy % f;
    int oz = iz + ic * (dx + f * dy);
  
    int outi = oz + oc * (ox + ow * oy);
  
    fin[ini] = fout[outi];
  }
}



static inline __device__ double PRE(f)(double n) {
   return (n > 0.04045 ? pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

static inline __device__ void PRE(rgbtoxyz)(unsigned char r, unsigned char g, unsigned char b, double *xp, double *yp, double *zp) {
  double dr = PRE(f)((double)r / 255.0);
  double dg = PRE(f)((double)g / 255.0);
  double db = PRE(f)((double)b / 255.0);

  *xp = dr * 0.4124 + dg * 0.3576 + db * 0.1805;
  *yp = dr * 0.2126 + dg * 0.7152 + db * 0.0722;
  *zp = dr * 0.0193 + dg * 0.1192 + db * 0.9505;
}

static inline __device__ void PRE(xyztorgb)(double x, double y, double z, unsigned char *rp, unsigned char *gp, unsigned char *bp) {
  double dx = x / 100.0;
  double dy = y / 100.0;
  double dz = z / 100.0;

  double pr = dx * 3.2404542 + dy * -1.5371385 + dz * -0.4985314;
  double pg = dx * -0.9692660 + dy * 1.8760108 + dz * 0.0415560;
  double pb = dx * 0.0556434 + dy * -0.2040259 + dz * 1.0572252;

  double qr = (((pr > 0.0031308) ? (1.055*pow(pr, 1 / 2.4) - 0.055) : (12.92*pr)) * 255.0);
  double qg = (((pg > 0.0031308) ? (1.055*pow(pg, 1 / 2.4) - 0.055) : (12.92*pg)) * 255.0);
  double qb = (((pb > 0.0031308) ? (1.055*pow(pb, 1 / 2.4) - 0.055) : (12.92*pb)) * 255.0);

  if (qr < 0) qr = 0; if (qr > 255.0) qr = 255.0;
  if (qg < 0) qg = 0; if (qg > 255.0) qg = 255.0;
  if (qb < 0) qb = 0; if (qb > 255.0) qb = 255.0;

  *rp = qr;
  *gp = qg;
  *bp = qb;
}

		

static inline __device__ double PRE(g)(double n) {
  const double EPS = 0.008856;
  const double KAPPA = 903.3;
  return n > EPS ? pow(n, 1.0/3.0) : (KAPPA * n + 16.0) / 116.0;
}

static inline __device__ void PRE(xyztolab)(double x, double y, double z, double *lp, double *ap, double *bp) {
  double dx = PRE(g)(x / 95.047);
  double dy = PRE(g)(y / 100.0);
  double dz = PRE(g)(z / 108.883);

  *lp = 116.0 * dy - 16.0;
  if (*lp < 0)
    *lp = 0;
  *ap = 500.0 * (dx - dy);
  *bp = 200.0 * (dy - dz);

  *lp /= 100.0;
  *ap /= 100.0;
  *bp /= 100.0;
}

static inline __device__ void PRE(labtoxyz)(double l, double a, double b, double *xp, double *yp, double *zp) {
  l *= 100.0;
  a *= 100.0;
  b *= 100.0;

  double py = (l + 16.0) / 116.0;
  double px = a / 500.0 + py;
  double pz = py - b / 200.0;

  double x3 = px*px*px;
  double y3 = py*py*py;
  double z3 = pz*pz*pz;

  *xp = ((x3 > 0.008856) ? x3 : ((px - 16.0 / 116.0) / 7.787)) * 95.047;
  *yp = ((y3 > 0.008856) ? y3 : ((py - 16.0 / 116.0) / 7.787)) * 100.0;
  *zp = ((z3 > 0.008856) ? z3 : ((pz - 16.0 / 116.0) / 7.787)) * 108.883;
}

static inline __device__ unsigned char PRE(dtob)(double d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((unsigned char)(d + 0.5));
}

static inline __device__ void PRE(rgbtolab)(double r, double g, double b, double *lp, double *ap, double *bp) {
  double x, y, z;
  PRE(rgbtoxyz)(PRE(dtob)(r), PRE(dtob)(g), PRE(dtob)(b), &x, &y, &z);
  PRE(xyztolab)(x, y, z, lp, ap, bp);
}

static inline __device__ void PRE(labtorgb)(double l, double a, double b, double *rp, double *gp, double *bp) {
  double x, y, z;
  PRE(labtoxyz)(l, a, b, &x, &y, &z);
  unsigned char rip, gip, bip;
  PRE(xyztorgb)(x, y, z, &rip, &gip, &bip);
  *rp = (double)rip / 255.0;
  *gp = (double)gip / 255.0;
  *bp = (double)bip / 255.0;
}





DEFN_KERNEL(synth_tlab,
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

  int rgbi = 0 + ic * (ix + iw * iy);
  double lab[3] = {0,0,0};
  PRE(rgbtolab)(in[rgbi], in[rgbi+1], in[rgbi+2], lab, lab+1, lab+2);
  out[outi] = lab[oz];
}

DEFN_KERNEL(synth_flab,
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

  int labi = 0 + ic * (ix + iw * iy);
  double rgb[3] = {0,0,0};
  PRE(labtorgb)(in[labi], in[labi+1], in[labi+2], rgb, rgb+1, rgb+2);
  out[outi] = rgb[oz];
}
