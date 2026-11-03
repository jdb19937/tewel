
DEFN_KERNEL(kaddvec, const float *a, const float *b, unsigned int n, float *c) {
  PREF_KERNEL;
  c[i] = a[i] + b[i];
}

DEFN_KERNEL(ksubvec, const float *a, const float *b, unsigned int n, float *c) {
  PREF_KERNEL;
  c[i] = a[i] - b[i];
}

DEFN_KERNEL(kaxpy, const float *x, const float a, unsigned int n, float *y) {
  PREF_KERNEL;
  y[i] += a * x[i];
}

DEFN_KERNEL(kmul, const float *a, const float b, unsigned int n, float *c) {
  PREF_KERNEL;
  c[i] = a[i] * b;
}

DEFN_KERNEL(krectify, const float *a, float *b, unsigned int n) {
  PREF_KERNEL;
  if (a[i] <= 0.0)
    b[i] = 0.0;
}

DEFN_KERNEL(ksumsq, const float *a, unsigned int n, float *sumsqp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);
  
  float s = 0;
  for (unsigned int j = j0; j < j1; ++j)
    s += a[j] * a[j];

  sumsqp[i] = s;
}

DEFN_KERNEL(ksum, const float *a, unsigned int n, float *sumsqp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);
  
  float s = 0;
  for (unsigned int j = j0; j < j1; ++j)
    s += a[j];

  sumsqp[i] = s;
}

DEFN_KERNEL(kmaxabs, const float *a, unsigned int n, float *maxp) {
  PREF_KERNEL;

  unsigned int j0 = i * 128;
  if (j0 >= n)
    return;
  unsigned int j1 = (j0 + 128 >= n) ? n : (j0 + 128);

  unsigned int j = j0;
  float s = fabs(a[j]);
  ++j;
  
  for (; j < j1; ++j) {
    float aa = fabs(a[j]);
    if (aa > s)
      s = aa;
  }

  maxp[i] = s;
}

DEFN_KERNEL(kspliceadd,
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] += x[in * xm + xa + ixk];
}

DEFN_KERNEL(ksplice,
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] = x[in * xm + xa + ixk];
}

DEFN_KERNEL(ksplice2,
  const float *x, const float *z, float l, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
) {
  PREF_KERNEL;

  int ixk = i / n;
  int in = i % n;

  y[in * ym + ya + ixk] =
    (1.0 - l) * x[in * xm + xa + ixk] +
    l * z[in * xm + xa + ixk];
}

DEFN_KERNEL(synth_norm,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;

  int ini = outi;

  float outv = 0;
  outv += in[ini] * (1.0 + wmv[outi + outn]);
  outv += wmv[outi];

  out[outi] = outv;
}


DEFN_KERNEL(learn_norm1,
  const float *in, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *dwv,
  float nu, float b1, float b2
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

  float dw = 0;

  if (wi < outn) {
    int outi = wi;
    dw += fout[outi];
  } else {
#if 0
    int outi = wi - outn;
    dw += fout[outi] * (1.0 + wmv[wi]);
#endif
    dw = 0;
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_norm2,
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,

  const float *wmv
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
  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds, float mc, float vc
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int outn = iw * ih * oc;
  int wn = outn * 2;

  int wi = i;
  if (wi >= wn)
    return;

  float dw = dwv[wi];

  float m = wmv[wn + wi];
  float v = wmv[2 * wn + wi];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wn + wi] = m;
  wmv[2 * wn + wi] = v;

  float w = wmv[wi];

  m *= mc;
  v *= vc;

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi] = w;
  dwv[wi] = 0;
}



DEFN_KERNEL(synth_bias,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
) {
  PREF_KERNEL;

  int ow = iw;
  int oh = ih;

  int outn = ow * oh * oc;
  int outi = i;
  if (outi >= outn)
    return;


  int ini = outi;

  float outv = 0;
  outv += in[ini];
  outv += wmv[outi];

  out[outi] = outv;
}


DEFN_KERNEL(learn_bias1,
  const float *in, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *dwv,
  float nu, float b1, float b2
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

  float dw = 0;

  if (wi < outn) {
    int outi = wi;
    dw += fout[outi];
  }

  dwv[wi] += dw;
}


DEFN_KERNEL(learn_bias2,
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,
  const float *wmv
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
  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds, float mc, float vc
) {
  PREF_KERNEL;

  if (!(nu > 0))
    return;

  int outn = iw * ih * oc;
  int wn = outn;

  int wi = i;
  if (wi >= wn)
    return;

  float dw = dwv[wi];

  float m = wmv[wn + wi];
  float v = wmv[2 * wn + wi];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wn + wi] = m;
  wmv[2 * wn + wi] = v;

  float w = wmv[wi];

  m *= mc;
  v *= vc;

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_local,
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  const float *wmv
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

  float outv = 0;

  for (int iyt = iy0; iyt <= iy1; ++iyt) {
    int iy = (iyt + ih) % ih;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;
      for (int iz = 0; iz < ic; ++iz) {
        int ini = iz + ic * (ix + iw * iy);

        int dx = ixt - ix0;
        int dy = iyt - iy0;
        int wi = outn + (iz + ic * (dx + f * (dy + f * outi)));

        float v = in[ini];
        float w = wmv[wi];

        outv += v * w;
      }
    }
  }

  outv += wmv[outi];

  out[outi] = outv;
}


DEFN_KERNEL(learn_local1,
  const float *in, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,

  float *dwv,
  float nu, float b1, float b2
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

  float dw = 0;

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
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,

  const float *wmv
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

  float finv = 0;

  for (int oyt = oy0; oyt <= oy1; ++oyt) {
    int oy = (oyt + oh) % oh;
    for (int oxt = ox0; oxt <= ox1; ++oxt) { 
      int ox = (oxt + ow) % ow;
      for (int oz = 0; oz < oc; ++oz) {
        int outi = oz + oc * (ox + ow * oy);

        int dx = ix - oxt + d;
        int dy = iy - oyt + d;
        int wi = outn + (iz + ic * (dx + f * (dy + f * outi)));

        float v = fout[outi];
        float w = wmv[wi];
        finv += v * w;
      }
    }
  }

  fin[ini] = finv;
}


DEFN_KERNEL(learn_local3,
  int iw, int ih,
  int d, int ic, int oc,
  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds, float mc, float vc
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

  float dw = dwv[wi];

  float m = wmv[wn + wi];
  float v = wmv[2 * wn + wi];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wn + wi] = m;
  wmv[2 * wn + wi] = v;

  float w = wmv[wi];

  m *= mc;
  v *= vc;

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_padd,
  const float *in, int iw, int ih,
  float *out,
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
  const float *in, int iw, int ih,
  float *out,
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
  const float *in, int iw, int ih,
  float *out,
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
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  bool relu, int freeze,
  int dim,
  const float *wmv, float wscale
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
    float v = in[ini];

    out[outi] = v;
    return;
  }

  int ix0 = ox - d;
  int ix1 = ox + d;
  float outv = 0;
  int f = d * 2 + 1;

  if (dim == 1) {
    // int wmvn = 3 * (oc + ic * f * f * oc);

    int iy = oy;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;
      for (int iz = 0; iz < ic; ++iz) {
        int ini = iz + ic * (ix + iw * iy);
        float v = in[ini];

        if (!relu || v > 0 || iz < freeze) {
          int dx = ixt - ix0;
          int wi = oc + (iz + ic * (dx + f * oz));

          float w = wmv[wi] * wscale;

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
          float v = in[ini];

          if (!relu || v > 0 || iz < freeze) {
            int dx = ixt - ix0;
            int dy = iyt - iy0;
            int wi = oc + (iz + ic * (dx + f * (dy + f * oz)));

            float w = wmv[wi] * wscale;
  
            outv += v * w;
          }
        }
      }
    }
  }

  outv += wmv[oz];

  out[outi] = outv;
}

DEFN_KERNEL(learn_padd,
  float *fin, int iw, int ih,
  const float *fout,

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
  float *fin, int iw, int ih,
  const float *fout,

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
  const float *in, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,
  bool relu,

  int freeze,
  int dim,
  float *dwv,
  float nu, float b1, float b2, float wscale
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

  int wi = i % wn;
  int oy = i / wn;
  if (oy >= ih)
    return;

  int ow = iw;

  float dw = 0;

  if (wi < oc) {
    int oz = wi;

    if (oz < freeze) {
      dw = 0;
    } else {
      {
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
        {
          for (int ox = 0; ox < ow; ++ox) {
            int ix = (ox - d + dx + iw) % iw;
            int iy = oy;
  
            int ini = iz + ic * (ix + iw * iy);
            float v = in[ini];
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
        {
          for (int ox = 0; ox < ow; ++ox) {
            int ix = (ox - d + dx + iw) % iw;
            int iy = (oy - d + dy + ih) % ih;
    
            int ini = iz + ic * (ix + iw * iy);
            float v = in[ini];
            if (!relu || v > 0 || iz < freeze) {
              int outi = oz + oc * (ox + ow * oy);
              dw += fout[outi] * v;
            }
          }
        }
      }
    }
  }

  KATOMIC_ADD(&dwv[wi], (wi < oc) ? dw : dw * wscale);
}


DEFN_KERNEL(learn_conv2,
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,
  bool relu, int freeze,
  int dim,

  const float *wmv, float wscale
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
  float finv = 0;

  if (dim == 1) {
    int oy = iy;
    for (int oxt = ox0; oxt <= ox1; ++oxt) { 
      int ox = (oxt + ow) % ow;
      for (int oz = 0; oz < oc; ++oz) {
        int outi = oz + oc * (ox + ow * oy);

        float v = fout[outi];
        float w;

        if (oz < freeze) {
          if (ix == ox && iy == oy && iz == oz) {
            w = 1.0;
          } else {
            w = 0.0;
          }
        } else {
          int dx = ix - oxt + d;
          int wi = oc + (iz + ic * (dx + f * oz));
          w = wmv[wi] * wscale;
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

          float v = fout[outi];
          float w;

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
  
            w = wmv[wi] * wscale;
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
  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds, float mc, float vc
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

  float dw = dwv[wi];

  float m = wmv[wn + wi];
  float v = wmv[2 * wn + wi];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wn + wi] = m;
  wmv[2 * wn + wi] = v;

  float w = wmv[wi];

  m *= mc;
  v *= vc;

  w += nu * m / (sqrt(v) + eps);

  if (clip > 0 && wi >= oc) {
    if (w > clip)
      w = clip;
    else if (w < -clip)
      w = -clip;
  }

  wmv[wi] = w;
  dwv[wi] = 0;
}

DEFN_KERNEL(synth_addgeo,
  const float *in, int iw, int ih,
  float *out,
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
    out[outi] = (float)ox / (float)ow;
  } else if (oz == ic + 1) {
    out[outi] = (float)oy / (float)oh;
  } else if (oz == ic + 2) {
    out[outi] = 1.0 - 2.0 * fabs(0.5 - (float)ox / (float)ow);
  } else if (oz == ic + 3) {
    out[outi] = 1.0 - 2.0 * fabs(0.5 - (float)oy / (float)oh);
  } else if (oz == ic + 4) {
    out[outi] = (log((float)ow) / log(2.0)) / 16.0;
  } else {
    out[outi] = 0;
  }
}

DEFN_KERNEL(learn_addgeo,
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  fin[ini] = 0;
}

DEFN_KERNEL(synth_popp,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  int iz = ((ic - oc) + oz) % ic;

  int ini = iz + ic * (ix + iw * iy);
  out[outi] = in[ini];
}

DEFN_KERNEL(learn_popp,
  float *fin, int iw, int ih,
  const float *fout,
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

  if (iz < (ic - oc)) {
    fin[ini] = 0;
    return;
  }

  int ox = ix;
  int oy = iy;
  int oz = (iz - (ic - oc));

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi];
}

DEFN_KERNEL(synth_iden,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float *fin, int iw, int ih,
  const float *fout,
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

  float v = 0;
  for (int oz = iz; oz < oc; oz += ic) {
    int outi = oz + oc * (ox + ow * oy);
    v += fout[outi];
  }

  fin[ini] = v;
}

DEFN_KERNEL(synth_smax,
  float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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

  float v = 0;
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
    in[ini] = (in[ini] >= v) ? 1.0 / (float)nmax : 0.0;
  }

  out[outi] = v;
}

DEFN_KERNEL(learn_smax,
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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

  float v = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    v += in[ini];
  }
  v *= (float)oc / (float)ic;
  out[outi] = v;
}

DEFN_KERNEL(learn_mean,
  float *fin, int iw, int ih,
  const float *fout,
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
  fin[ini] = fout[outi] * (float)oc / (float)ic;
}

DEFN_KERNEL(synth_median,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float *fin, int iw, int ih,
  const float *fout,
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

  float dv = 0;

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
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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

  float v = 0;

  for (int iyt = iy0; iyt <= iy1; ++iyt) {
    int iy = (iyt + ih) % ih;
    for (int ixt = ix0; ixt <= ix1; ++ixt) { 
      int ix = (ixt + iw) % iw;

      float w = 0.25;
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
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float v = in[ini];

  iz += oc;
  ini = iz + ic * (ix + iw * iy);
  v -= in[ini];

  out[outi] = v;
}

DEFN_KERNEL(learn_diff,
  float *fin, int iw, int ih,
  const float *fout,
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
  float sgn = (q % 2) ? -1.0 : 1.0;

  int outi = oz + oc * (ox + ow * oy);
  fin[ini] = fout[outi] * sgn;
}

DEFN_KERNEL(synth_sum,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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

  float v = 0;
  for (int iz = oz; iz < ic; iz += oc) {
    int ini = iz + ic * (ix + iw * iy);
    v += in[ini];
  }
  out[outi] = v;
}

DEFN_KERNEL(learn_sum,
  float *fin, int iw, int ih,
  const float *fout,
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

DEFN_KERNEL(synth_cent,
  const float *in, int iw, int ih,
  float *out,
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

DEFN_KERNEL(learn_cent,
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;
  fin[ini] = fout[outi];
}

DEFN_KERNEL(synth_sigm,
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;
  float outv = 1.0 / (1.0 + exp(-fin[ini]));
  fin[ini] = fout[outi] * outv * (1.0 - outv);
}

DEFN_KERNEL(synth_inrf,
  const float *in, int iw, int ih,
  float *out,
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
  float v = in[ini];

  if (v < 1.0/256.0)
    v = 1.0/256.0;
  if (v >= 255.0/256.0)
    v = 255.0/256.0;

  out[outi] = sqrt(2.0) * erfinv(-1.0 + 2.0 * v);
}

DEFN_KERNEL(learn_inrf,
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;

  const float sq2pi = 2.506628275;
  fin[ini] = fout[outi] * sq2pi * exp(fout[outi] * fout[outi] * 2.0);
}

DEFN_KERNEL(synth_nerf,
  const float *in, int iw, int ih,
  float *out,
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

  const float isq2 = 0.7071067812;

  int ini = outi;

  out[outi] = (erf(in[ini] * isq2) + 1.0) / 2.0;
}

DEFN_KERNEL(learn_nerf,
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;

  int inn = iw * ih * ic;
  int ini = i;
  if (ini >= inn)
    return;

  int outi = ini;

  const float isq2pi = 0.3989422804;
  fin[ini] = fout[outi] * isq2pi * exp(-fin[ini] * fin[ini] / 2.0);
}


DEFN_KERNEL(synth_clamp,
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
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
  const float *in, int iw, int ih,
  float *out,
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
  float *fin, int iw, int ih,
  const float *fout,
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



static inline __device__ float PRE(f)(float n) {
   return (n > 0.04045 ? pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

static inline __device__ void PRE(rgbtoxyz)(unsigned char r, unsigned char g, unsigned char b, float *xp, float *yp, float *zp) {
  float dr = PRE(f)((float)r / 255.0);
  float dg = PRE(f)((float)g / 255.0);
  float db = PRE(f)((float)b / 255.0);

  *xp = dr * 0.4124 + dg * 0.3576 + db * 0.1805;
  *yp = dr * 0.2126 + dg * 0.7152 + db * 0.0722;
  *zp = dr * 0.0193 + dg * 0.1192 + db * 0.9505;
}

static inline __device__ void PRE(xyztorgb)(float x, float y, float z, unsigned char *rp, unsigned char *gp, unsigned char *bp) {
  float dx = x / 100.0;
  float dy = y / 100.0;
  float dz = z / 100.0;

  float pr = dx * 3.2404542 + dy * -1.5371385 + dz * -0.4985314;
  float pg = dx * -0.9692660 + dy * 1.8760108 + dz * 0.0415560;
  float pb = dx * 0.0556434 + dy * -0.2040259 + dz * 1.0572252;

  float qr = (((pr > 0.0031308) ? (1.055*pow(pr, 1 / 2.4) - 0.055) : (12.92*pr)) * 255.0);
  float qg = (((pg > 0.0031308) ? (1.055*pow(pg, 1 / 2.4) - 0.055) : (12.92*pg)) * 255.0);
  float qb = (((pb > 0.0031308) ? (1.055*pow(pb, 1 / 2.4) - 0.055) : (12.92*pb)) * 255.0);

  if (qr < 0) qr = 0; if (qr > 255.0) qr = 255.0;
  if (qg < 0) qg = 0; if (qg > 255.0) qg = 255.0;
  if (qb < 0) qb = 0; if (qb > 255.0) qb = 255.0;

  *rp = qr;
  *gp = qg;
  *bp = qb;
}

		

static inline __device__ float PRE(g)(float n) {
  const float EPS = 0.008856;
  const float KAPPA = 903.3;
  return n > EPS ? pow(n, 1.0/3.0) : (KAPPA * n + 16.0) / 116.0;
}

static inline __device__ void PRE(xyztolab)(float x, float y, float z, float *lp, float *ap, float *bp) {
  float dx = PRE(g)(x / 95.047);
  float dy = PRE(g)(y / 100.0);
  float dz = PRE(g)(z / 108.883);

  *lp = 116.0 * dy - 16.0;
  if (*lp < 0)
    *lp = 0;
  *ap = 500.0 * (dx - dy);
  *bp = 200.0 * (dy - dz);

  *lp /= 100.0;
  *ap /= 100.0;
  *bp /= 100.0;
}

static inline __device__ void PRE(labtoxyz)(float l, float a, float b, float *xp, float *yp, float *zp) {
  l *= 100.0;
  a *= 100.0;
  b *= 100.0;

  float py = (l + 16.0) / 116.0;
  float px = a / 500.0 + py;
  float pz = py - b / 200.0;

  float x3 = px*px*px;
  float y3 = py*py*py;
  float z3 = pz*pz*pz;

  *xp = ((x3 > 0.008856) ? x3 : ((px - 16.0 / 116.0) / 7.787)) * 95.047;
  *yp = ((y3 > 0.008856) ? y3 : ((py - 16.0 / 116.0) / 7.787)) * 100.0;
  *zp = ((z3 > 0.008856) ? z3 : ((pz - 16.0 / 116.0) / 7.787)) * 108.883;
}

static inline __device__ unsigned char PRE(dtob)(float d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((unsigned char)(d + 0.5));
}

static inline __device__ void PRE(rgbtolab)(float r, float g, float b, float *lp, float *ap, float *bp) {
  float x, y, z;
  PRE(rgbtoxyz)(PRE(dtob)(r), PRE(dtob)(g), PRE(dtob)(b), &x, &y, &z);
  PRE(xyztolab)(x, y, z, lp, ap, bp);
}

static inline __device__ void PRE(labtorgb)(float l, float a, float b, float *rp, float *gp, float *bp) {
  float x, y, z;
  PRE(labtoxyz)(l, a, b, &x, &y, &z);
  unsigned char rip, gip, bip;
  PRE(xyztorgb)(x, y, z, &rip, &gip, &bip);
  *rp = (float)rip / 255.0;
  *gp = (float)gip / 255.0;
  *bp = (float)bip / 255.0;
}





DEFN_KERNEL(synth_tlab,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float lab[3] = {0,0,0};
  PRE(rgbtolab)(in[rgbi], in[rgbi+1], in[rgbi+2], lab, lab+1, lab+2);
  out[outi] = lab[oz];
}

DEFN_KERNEL(synth_flab,
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
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
  float rgb[3] = {0,0,0};
  PRE(labtorgb)(in[labi], in[labi+1], in[labi+2], rgb, rgb+1, rgb+2);
  out[outi] = rgb[oz];
}


DEFN_KERNEL(kcharb, float *e, unsigned int n, float delta) {
  PREF_KERNEL;
  float x = e[i];
  e[i] = x / sqrt(x * x + delta * delta);
}

DEFN_KERNEL(synth_lrlu,
  const float *in, int iw, int ih,
  float *out,
  int ic, float slope
) {
  PREF_KERNEL;
  float v = in[i];
  out[i] = (v > 0) ? v : v * slope;
}

DEFN_KERNEL(learn_lrlu,
  float *fin, int iw, int ih,
  const float *fout,
  int ic, float slope
) {
  PREF_KERNEL;
  fin[i] = (fin[i] > 0) ? fout[i] : fout[i] * slope;
}

// pixel norm: y = x / sqrt(mean_c(x^2) + eps), one thread per output element
DEFN_KERNEL(synth_pnrm,
  const float *in, int iw, int ih,
  float *out,
  int ic
) {
  PREF_KERNEL;
  long p = i / ic;
  const float *x = in + p * ic;
  float s = 0;
  for (int z = 0; z < ic; ++z)
    s += x[z] * x[z];
  s = sqrt(s / (float)ic + 1e-8);
  out[i] = in[i] / s;
}

// pixel norm backward, one thread per pixel (owns all channels, no race)
DEFN_KERNEL(learn_pnrm,
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;
  float *x = fin + i * ic;
  const float *g = fout + i * ic;
  float s = 0;
  for (int z = 0; z < ic; ++z)
    s += x[z] * x[z];
  s = sqrt(s / (float)ic + 1e-8);
  float dot = 0;
  for (int z = 0; z < ic; ++z)
    dot += g[z] * x[z];
  dot /= ((float)ic * s * s);
  for (int z = 0; z < ic; ++z)
    x[z] = (g[z] - x[z] * dot) / s;
}


// 2x bilinear upsample (half-pixel centers, clamped borders), ic == oc
DEFN_KERNEL(synth_upbl,
  const float *in, int iw, int ih,
  float *out,
  int ic, int bn
) {
  PREF_KERNEL;   // i over bn * oh * ow * ic
  int ow = iw * 2, oh = ih * 2;
  long outi = i;
  int oz = outi % ic; long t = outi / ic;
  int ox = t % ow; long t2 = t / ow; int oy = t2 % oh; long b = t2 / oh;
  in += b * (long)iw * ih * ic;
  float sx = (ox + 0.5f) * 0.5f - 0.5f;
  float sy = (oy + 0.5f) * 0.5f - 0.5f;
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  float wx = sx - x0, wy = sy - y0;
  int x0c = x0 < 0 ? 0 : x0, x1c = x0 + 1 >= iw ? iw - 1 : x0 + 1;
  int y0c = y0 < 0 ? 0 : y0, y1c = y0 + 1 >= ih ? ih - 1 : y0 + 1;
  float v =
    (1 - wy) * ((1 - wx) * in[oz + ic * (x0c + iw * y0c)] + wx * in[oz + ic * (x1c + iw * y0c)]) +
    wy       * ((1 - wx) * in[oz + ic * (x0c + iw * y1c)] + wx * in[oz + ic * (x1c + iw * y1c)]);
  out[outi] = v;
}

DEFN_KERNEL(learn_upbl,
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int bn
) {
  PREF_KERNEL;   // i over bn * ih * iw * ic
  int ow = iw * 2, oh = ih * 2;
  long ini = i;
  int iz = ini % ic; long t = ini / ic;
  int ix = t % iw; long t2 = t / iw; int iy = t2 % ih; long b = t2 / ih;
  fout += b * (long)ow * oh * ic;
  float g = 0;
  int oy0 = 2 * iy - 2 < 0 ? 0 : 2 * iy - 2, oy1 = 2 * iy + 3 >= oh ? oh - 1 : 2 * iy + 3;
  int ox0 = 2 * ix - 2 < 0 ? 0 : 2 * ix - 2, ox1 = 2 * ix + 3 >= ow ? ow - 1 : 2 * ix + 3;
  for (int oy = oy0; oy <= oy1; ++oy) {
    float sy = (oy + 0.5f) * 0.5f - 0.5f;
    int y0 = (int)floorf(sy); float wy = sy - y0;
    int y0c = y0 < 0 ? 0 : y0, y1c = y0 + 1 >= ih ? ih - 1 : y0 + 1;
    float cy = (y0c == iy ? (1 - wy) : 0) + (y1c == iy ? wy : 0);
    if (cy == 0) continue;
    for (int ox = ox0; ox <= ox1; ++ox) {
      float sx = (ox + 0.5f) * 0.5f - 0.5f;
      int x0 = (int)floorf(sx); float wx = sx - x0;
      int x0c = x0 < 0 ? 0 : x0, x1c = x0 + 1 >= iw ? iw - 1 : x0 + 1;
      float cx = (x0c == ix ? (1 - wx) : 0) + (x1c == ix ? wx : 0);
      if (cx == 0) continue;
      g += cx * cy * fout[iz + ic * (ox + ow * oy)];
    }
  }
  fin[ini] = g;
}

// 2x average pool, ic == oc
DEFN_KERNEL(synth_dnav,
  const float *in, int iw, int ih,
  float *out,
  int ic
) {
  PREF_KERNEL;
  int ow = iw / 2;
  long outi = i;
  int oz = outi % ic; long t = outi / ic;
  int ox = t % ow; int oy = t / ow;
  int ix = ox * 2, iy = oy * 2;
  float v = in[oz + ic * (ix + iw * iy)] + in[oz + ic * (ix + 1 + iw * iy)]
          + in[oz + ic * (ix + iw * (iy + 1))] + in[oz + ic * (ix + 1 + iw * (iy + 1))];
  out[outi] = v * 0.25f;
}

DEFN_KERNEL(learn_dnav,
  float *fin, int iw, int ih,
  const float *fout,
  int ic
) {
  PREF_KERNEL;
  int ow = iw / 2;
  long ini = i;
  int iz = ini % ic; long t = ini / ic;
  int ix = t % iw; int iy = t / iw;
  fin[ini] = 0.25f * fout[iz + ic * (ix / 2 + ow * (iy / 2))];
}

// a = beta * a + (1 - beta) * b
DEFN_KERNEL(kema, float *a, const float *b, unsigned int n, float beta) {
  PREF_KERNEL;
  a[i] = beta * a[i] + (1.0f - beta) * b[i];
}

// logistic (non-saturating) GAN gradient on logits, ascent convention:
//   sign > 0: x = sigmoid(-x)   (push logit up)    sign < 0: x = -sigmoid(x)  (push logit down)
DEFN_KERNEL(klogis, float *x, unsigned int n, int sign) {
  PREF_KERNEL;
  float v = x[i];
  x[i] = (sign > 0) ? 1.0f / (1.0f + expf(v)) : -1.0f / (1.0f + expf(-v));
}


// minibatch standard deviation (StyleGAN): appends one channel holding the mean over
// (pixels, channels) of the per-(pixel, channel) std across the batch. Batched kernels.
DEFN_KERNEL(mbsd_stat, const float *in, int bn, long pc, float *ksig) {
  PREF_KERNEL;   // i over p*ic
  long whc = pc;
  float mu = 0;
  for (int b = 0; b < bn; ++b)
    mu += in[i + whc * b];
  mu /= bn;
  float var = 0;
  for (int b = 0; b < bn; ++b) {
    float d = in[i + whc * b] - mu;
    var += d * d;
  }
  ksig[i] = sqrtf(var / bn + 1e-8f);
}

DEFN_KERNEL(mbsd_fill, const float *in, int bn, long pn, int ic, float *out, float s) {
  PREF_KERNEL;   // i over bn * pn * (ic + 1)
  int oc = ic + 1;
  int z = i % oc; long bp = i / oc;   // bp = p + pn * b
  out[i] = (z < ic) ? in[z + ic * bp] : s;
}

DEFN_KERNEL(mbsd_extra, const float *fout, long pn, int bn, int ic, float *tmp) {
  PREF_KERNEL;   // i over bn * pn: gather the appended channel's gradient
  tmp[i] = fout[ic + (ic + 1) * i];
}

DEFN_KERNEL(mbsd_learn, float *fin, int bn, long pn, int ic, const float *fout, float gs) {
  PREF_KERNEL;   // i over bn * pn * ic
  int z = i % ic; long bp = i / ic; long p = bp % pn; int b = bp / pn;
  long whc = pn * ic;
  long j = z + ic * p;   // (p, c) index within a sample
  float mu = 0;
  for (int k = 0; k < bn; ++k)
    mu += fin[j + whc * k];
  mu /= bn;
  float var = 0;
  for (int k = 0; k < bn; ++k) {
    float d = fin[j + whc * k] - mu;
    var += d * d;
  }
  float sig = sqrtf(var / bn + 1e-8f);
  float x = fin[i];
  float g = fout[z + (ic + 1) * bp] + gs * (x - mu) / ((float)bn * sig * (float)pn * (float)ic);
  fin[i] = g;   // note: other threads still read fin[j + whc*k]; see wrapper (two-pass)
}

DEFN_KERNEL(kcopyn, const float *a, unsigned int n, float *b) {
  PREF_KERNEL;
  b[i] = a[i];
}


// ---- style modulation (AdaIN-style affine from a per-sample style vector) ----
// weights: [scale bias: ic][shift bias: ic][A: ic x sdim][B: ic x sdim]; wi >= 2*ic: j = (wi-2ic)/sdim, k = (wi-2ic)%sdim
// out[b,p,c] = in[b,p,c] * (1 + sc[b,c]) + sh[b,c]
DEFN_KERNEL(styl_affine, const float *wmv, const float *kstyle, int bn, int ic, int sdim, float wscale, float *ksc) {
  PREF_KERNEL;   // i over bn * 2ic
  int j = i % (2 * ic); int b = i / (2 * ic);
  const float *s = kstyle + (long)b * sdim;
  const float *w = wmv + 2 * ic + (long)j * sdim;
  float v = 0;
  for (int k = 0; k < sdim; ++k)
    v += w[k] * s[k];
  ksc[i] = wmv[j] + wscale * v;
}

DEFN_KERNEL(styl_apply, const float *in, long pn, int ic, const float *ksc, float *out) {
  PREF_KERNEL;   // i over bn * pn * ic
  int c = i % ic; int b = i / (pn * ic);
  const float *sc = ksc + (long)b * 2 * ic;
  out[i] = in[i] * (1.0f + sc[c]) + sc[ic + c];
}

// per (b, j): j < ic: sum_p g * x  (d scale);  j >= ic: sum_p g  (d shift)
DEFN_KERNEL(styl_dsc, const float *x, const float *g, long pn, int ic, float *kd) {
  PREF_KERNEL;   // i over bn * 2ic
  int j = i % (2 * ic); int b = i / (2 * ic);
  long base = (long)b * pn * ic;
  float v = 0;
  if (j < ic) {
    for (long p = 0; p < pn; ++p) { long q = base + p * ic + j; v += g[q] * x[q]; }
  } else {
    int c = j - ic;
    for (long p = 0; p < pn; ++p) v += g[base + p * ic + c];
  }
  kd[i] = v;
}

DEFN_KERNEL(styl_din, float *fin, const float *fout, long pn, int ic, const float *ksc) {
  PREF_KERNEL;   // in place: fin = fout * (1 + sc)
  int c = i % ic; int b = i / (pn * ic);
  fin[i] = fout[i] * (1.0f + ksc[(long)b * 2 * ic + c]);
}

DEFN_KERNEL(styl_dw, const float *kd, const float *kstyle, int bn, int ic, int sdim, float wscale, float *dwv) {
  PREF_KERNEL;   // i over 2ic + 2ic*sdim
  float v = 0;
  if (i < 2 * ic) {
    for (int b = 0; b < bn; ++b) v += kd[(long)b * 2 * ic + i];
  } else {
    long t = i - 2 * ic; int j = t / sdim; int k = t % sdim;
    for (int b = 0; b < bn; ++b) v += kd[(long)b * 2 * ic + j] * kstyle[(long)b * sdim + k];
    v *= wscale;
  }
  dwv[i] += v;
}

DEFN_KERNEL(styl_ds, const float *kd, const float *wmv, int bn, int ic, int sdim, float wscale, float *kstylegrad) {
  PREF_KERNEL;   // i over bn * sdim
  int k = i % sdim; int b = i / sdim;
  const float *d = kd + (long)b * 2 * ic;
  const float *w = wmv + 2 * ic;
  float v = 0;
  for (int j = 0; j < 2 * ic; ++j)
    v += d[j] * w[(long)j * sdim + k];
  kstylegrad[i] += wscale * v;
}

// generic Adam step over wn weights (no clipping of the first nbias entries)
DEFN_KERNEL(learn_adam, float *wmv, float *dwv, int wn, int nbias,
  float nu, float b1, float b2, float eps, float clip, float mc, float vc) {
  PREF_KERNEL;
  if (!(nu > 0)) return;
  int wi = i;
  float dw = dwv[wi];
  float m = wmv[wn + wi], v = wmv[2 * wn + wi];
  m = (1.0 - b1) * m + b1 * dw;
  v = (1.0 - b2) * v + b2 * dw * dw;
  wmv[wn + wi] = m; wmv[2 * wn + wi] = v;
  float w = wmv[wi];
  w += nu * (m * mc) / (sqrtf(v * vc) + eps);
  if (clip > 0 && wi >= nbias) { if (w > clip) w = clip; else if (w < -clip) w = -clip; }
  wmv[wi] = w;
  dwv[wi] = 0;
}


// ---- modulated 3x3 convolution with demodulation (StyleGAN2) ----
// weights: [conv bias: oc][conv w: oc*ic*9 (k = iz + ic*(dx + 3*dy))][affine bias: ic][A: ic x sdim]
// s[b,iz] = abias[iz] + wscaleA * A[iz] . style[b]
// x'[b,p,iz] = x * s[b,iz];  y = conv(x', wscale * w) + bias;  out = y / d[b,oz]
// d[b,oz] = sqrt(sum_iz W2[oz,iz] * s[b,iz]^2 + eps),  W2[oz,iz] = sum_{dx,dy} (wscale * w)^2
DEFN_KERNEL(mcon_affine, const float *wmv, long offab, long offA, const float *kstyle, int bn, int ic, int sdim, float wscaleA, float *ks) {
  PREF_KERNEL;   // i over bn * ic
  int iz = i % ic; int b = i / ic;
  const float *a = wmv + offA + (long)iz * sdim;
  const float *st = kstyle + (long)b * sdim;
  float v = 0;
  for (int k = 0; k < sdim; ++k) v += a[k] * st[k];
  ks[i] = wmv[offab + iz] + wscaleA * v;
}

DEFN_KERNEL(mcon_w2, const float *wmv, int ic, int oc, float wscale, float *w2) {
  PREF_KERNEL;   // i over oc * ic
  int iz = i % ic; int oz = i / ic;
  long K = 9L * ic;
  float v = 0;
  for (int t = 0; t < 9; ++t) { float w = wscale * wmv[oc + iz + ic * t + K * oz]; v += w * w; }
  w2[i] = v;
}

DEFN_KERNEL(mcon_demod, const float *w2, const float *ks, int bn, int ic, int oc, float *kd) {
  PREF_KERNEL;   // i over bn * oc
  int oz = i % oc; int b = i / oc;
  float v = 0;
  for (int iz = 0; iz < ic; ++iz) { float s = ks[(long)b * ic + iz]; v += w2[(long)oz * ic + iz] * s * s; }
  kd[i] = sqrtf(v + 1e-8f);
}

DEFN_KERNEL(mcon_scale_in, const float *x, const float *ks, long pn, int ic, float *xs) {
  PREF_KERNEL;   // i over bn * pn * ic
  int c = i % ic; int b = i / (pn * ic);
  xs[i] = x[i] * ks[(long)b * ic + c];
}

// out = y / d, reading the raw conv output y (kept as the stash for the backward pass)
DEFN_KERNEL(mcon_divc, const float *y, const float *kd, long pn, int oc, float *out) {
  PREF_KERNEL;   // i over bn * pn * oc
  int c = i % oc; int b = i / (pn * oc);
  out[i] = y[i] / kd[(long)b * oc + c];
}

// u8 image rows -> float, cropped in x and optionally mirrored: out[z + pc*(x + pw*y)] = (0.5 + v) / 256
DEFN_KERNEL(u8crop, const unsigned char *src, int sw, int sc, int x0, int pw, int ph, int pc, int flip, float *out) {
  PREF_KERNEL;   // i over pw * ph * pc
  int z = i % pc; long t = i / pc; int x = t % pw; int y = t / pw;
  int xs = flip ? (pw - 1 - x) : x;
  out[i] = (0.5f + (float)src[z + sc * (x0 + xs + (long)sw * y)]) / 256.0f;
}

DEFN_KERNEL(mcon_div_out, float *out, const float *kd, long pn, int oc) {
  PREF_KERNEL;   // i over bn * pn * oc, in place
  int c = i % oc; int b = i / (pn * oc);
  out[i] /= kd[(long)b * oc + c];
}

// g_d[b,oz] = -(1/d) sum_p g * out   (out = stashed forward output)
DEFN_KERNEL(mcon_gd, const float *g, const float *outs, const float *kd, long pn, int oc, float *kgd) {
  PREF_KERNEL;   // i over bn * oc
  int oz = i % oc; int b = i / oc;
  long base = (long)b * pn * oc;
  float v = 0;
  for (long p = 0; p < pn; ++p) { long q = base + p * oc + oz; v += g[q] * outs[q]; }
  kgd[i] = -v / (kd[i] * kd[i]);   // outs is the raw conv output y = out * d
}

// g_s[b,iz] = sum_p g_x' * x  +  sum_oz g_d[b,oz] * s[b,iz] * W2[oz,iz] / d[b,oz]
// finishing steps after gpu_dot_bc computed the pixel sums (GPU path)
DEFN_KERNEL(mcon_gd_fin, float *kgd, const float *kd) {
  PREF_KERNEL;   // i over bn * oc; kgd holds sum_p g * y, y = out * d
  kgd[i] = -kgd[i] / (kd[i] * kd[i]);
}
DEFN_KERNEL(mcon_gs_fin, float *kgs, const float *ks, const float *kgd, const float *kd, const float *w2, int ic, int oc) {
  PREF_KERNEL;   // i over bn * ic; kgs holds sum_p gxs * x
  int iz = i % ic; int b = i / ic;
  float s = ks[i], u = 0;
  for (int oz = 0; oz < oc; ++oz) u += kgd[(long)b * oc + oz] * w2[(long)oz * ic + iz] / kd[(long)b * oc + oz];
  kgs[i] = kgs[i] + s * u;
}

DEFN_KERNEL(mcon_gs, const float *gxs, const float *x, const float *ks, const float *kgd, const float *kd, const float *w2, long pn, int ic, int oc, float *kgs) {
  PREF_KERNEL;   // i over bn * ic
  int iz = i % ic; int b = i / ic;
  long base = (long)b * pn * ic;
  float v = 0;
  for (long p = 0; p < pn; ++p) { long q = base + p * ic + iz; v += gxs[q] * x[q]; }
  float s = ks[i], u = 0;
  for (int oz = 0; oz < oc; ++oz) u += kgd[(long)b * oc + oz] * w2[(long)oz * ic + iz] / kd[(long)b * oc + oz];
  kgs[i] = v + s * u;
}

DEFN_KERNEL(mcon_gx, const float *gxs, const float *ks, long pn, int ic, float *fin) {
  PREF_KERNEL;   // i over bn * pn * ic; writes the input gradient over x
  int c = i % ic; int b = i / (pn * ic);
  fin[i] = gxs[i] * ks[(long)b * ic + c];
}

// demodulation's contribution to the conv weight gradient:
// dw[oz][k] += sum_b g_d[b,oz] * wscale^2 * w[oz][k] * s[b,iz]^2 / d[b,oz]
DEFN_KERNEL(mcon_dw_demod, const float *wmv, const float *ks, const float *kgd, const float *kd, int bn, int ic, int oc, float wscale, float *dwv) {
  PREF_KERNEL;   // i over oc * ic * 9 (conv weights, excluding bias)
  long K = 9L * ic;
  int oz = i / K; long k = i % K; int iz = k % ic;
  float w = wmv[oc + i];
  float v = 0;
  for (int b = 0; b < bn; ++b) { float s = ks[(long)b * ic + iz]; v += kgd[(long)b * oc + oz] * s * s / kd[(long)b * oc + oz]; }
  dwv[oc + i] += v * wscale * wscale * w;
}

DEFN_KERNEL(mcon_daffine, const float *kgs, const float *kstyle, int bn, int ic, int sdim, float wscaleA, long offab, long offA, float *dwv) {
  PREF_KERNEL;   // i over ic + ic*sdim
  float v = 0;
  if (i < ic) {
    for (int b = 0; b < bn; ++b) v += kgs[(long)b * ic + i];
    dwv[offab + i] += v;
  } else {
    long t = i - ic; int iz = t / sdim; int k = t % sdim;
    for (int b = 0; b < bn; ++b) v += kgs[(long)b * ic + iz] * kstyle[(long)b * sdim + k];
    dwv[offA + t] += wscaleA * v;
  }
}

DEFN_KERNEL(mcon_dstyle, const float *kgs, const float *wmv, long offA, int bn, int ic, int sdim, float wscaleA, float *kstylegrad) {
  PREF_KERNEL;   // i over bn * sdim
  int k = i % sdim; int b = i / sdim;
  float v = 0;
  for (int iz = 0; iz < ic; ++iz) v += kgs[(long)b * ic + iz] * wmv[offA + (long)iz * sdim + k];
  kstylegrad[i] += wscaleA * v;
}
