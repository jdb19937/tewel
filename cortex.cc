#define __MAKEMORE_CORTEX_CC__ 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <netinet/in.h>
#include <assert.h>

#include <math.h>

#include "colonel.hh"
#include "cortex.hh"
#include "random.hh"
#include "youtil.hh"

namespace makemore {

#define CC4(a,b,c,d) \
  ((uint32_t)(((a) << 24) | ((b) << 16) | ((c) << 8) | (d)))

const uint32_t TYPE_LOC0 = CC4('l','o','c','0');
const uint32_t TYPE_LOC1 = CC4('l','o','c','1');
const uint32_t TYPE_LOC2 = CC4('l','o','c','2');
const uint32_t TYPE_BIAS = CC4('b','i','a','s');
const uint32_t TYPE_NORM = CC4('n','o','r','m');
const uint32_t TYPE_RCN0 = CC4('r','c','n','0');
const uint32_t TYPE_RCN1 = CC4('r','c','n','1');
const uint32_t TYPE_RCN2 = CC4('r','c','n','2');
const uint32_t TYPE_RCN3 = CC4('r','c','n','3');
const uint32_t TYPE_CON0 = CC4('c','o','n','0');
const uint32_t TYPE_CON1 = CC4('c','o','n','1');
const uint32_t TYPE_CON2 = CC4('c','o','n','2');
const uint32_t TYPE_CON3 = CC4('c','o','n','3');
const uint32_t TYPE_CONV = CC4('c','o','n','v');
const uint32_t TYPE_SCAL = CC4('s','c','a','l');
const uint32_t TYPE_UPS1 = CC4('u','p','s','1');
const uint32_t TYPE_UPS2 = CC4('u','p','s','2');
const uint32_t TYPE_UPS3 = CC4('u','p','s','3');
const uint32_t TYPE_UPS4 = CC4('u','p','s','4');
const uint32_t TYPE_UPS5 = CC4('u','p','s','5');
const uint32_t TYPE_DNS1 = CC4('d','n','s','1');
const uint32_t TYPE_DNS2 = CC4('d','n','s','2');
const uint32_t TYPE_DNS3 = CC4('d','n','s','3');
const uint32_t TYPE_DNS4 = CC4('d','n','s','4');
const uint32_t TYPE_DNS5 = CC4('d','n','s','5');
const uint32_t TYPE_TRIM = CC4('t','r','i','m');
const uint32_t TYPE_PADD = CC4('p','a','d','d');
const uint32_t TYPE_CROP = CC4('c','r','o','p');
const uint32_t TYPE_SIGM = CC4('s','i','g','m');
const uint32_t TYPE_RELU = CC4('r','e','l','u');
const uint32_t TYPE_CLMP = CC4('c','l','m','p');
const uint32_t TYPE_ABSV = CC4('a','b','s','v');
const uint32_t TYPE_GRND = CC4('g','r','n','d');
const uint32_t TYPE_LRND = CC4('l','r','n','d');
const uint32_t TYPE_PAD1 = CC4('p','a','d','1');
const uint32_t TYPE_ADDG = CC4('a','d','d','g');
const uint32_t TYPE_IDEN = CC4('i','d','e','n');
const uint32_t TYPE_MEAN = CC4('m','e','a','n');
const uint32_t TYPE_SMAX = CC4('s','m','a','x');
const uint32_t TYPE_SUMM = CC4('s','u','m','m');
const uint32_t TYPE_BLUR = CC4('b','l','u','r');
const uint32_t TYPE_MEDI = CC4('m','e','d','i');
const uint32_t TYPE_NERF = CC4('n','e','r','f');
const uint32_t TYPE_INRF = CC4('i','n','r','f');
const uint32_t TYPE_ZERO = CC4('z','e','r','o');

static int _scale_factor(int ic, int oc) {
  assert(ic > 0);
  assert(oc > 0);

  int s = 0;
  if (ic < oc) {
    while (ic < oc) {
      ic <<= 2;
      --s;
    }
  } else if (ic > oc) {
    while (oc < ic) {
      oc <<= 2;
      ++s;
    }
  }

  if (ic != oc)
    error("channel ratio must be multiple of 4 to scale");

  return s;
}

static void _parse_header(
  const Cortex::Segment &hdr,
  uint32_t *typep, int *lenp,
  int *icp = NULL, int *ocp = NULL,
  int *iwp = NULL, int *ihp = NULL, int *owp = NULL, int *ohp = NULL,
  double *mulp = NULL, int *radp = NULL, int *freezep = NULL, int *relup = NULL, int *dimp = NULL
) {
  assert(!memcmp(hdr.magic, "MakeMore", 8));

  *typep = ntohl(hdr.type);
  *lenp = ntohl(hdr.len);

  if (icp)
    *icp = ntohl(hdr.ic);
  if (ocp)
    *ocp = ntohl(hdr.oc);
  if (iwp)
    *iwp = ntohl(hdr.iw);
  if (ihp)
    *ihp = ntohl(hdr.ih);
  if (owp)
    *owp = ntohl(hdr.ow);
  if (ohp)
    *ohp = ntohl(hdr.oh);
  if (mulp)
    *mulp = hdr.mul + 1.0;
  if (radp)
    *radp = ntohl(hdr.rad);
  if (freezep)
    *freezep = ntohl(hdr.freeze);
  if (relup)
    *relup = ntohl(hdr.relu);
  if (dimp)
    *dimp = ntohl(hdr.dim);
}


static size_t pipe_prep(uint8_t *base, size_t basen, int iw, int ih, int *icp, int *owp, int *ohp, int *ocp, unsigned long *dwnp) {
  int i = 0;
  size_t kbufn = 0;
  unsigned int basei = 0;
  int ow = 0, oh = 0, ic = 0, oc = 0;

  if (basei == basen)
    return 0;
  assert(basei < basen);

  *dwnp = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic;
    int viw, vih, vow, voh;
    int rad, freeze, relu;
    double _mul;
    int dim;
    _parse_header(hdr, &type, &len, &vic, &oc, &viw, &vih, &vow, &voh, &_mul, &rad, &freeze, &relu, &dim);

    if (viw)
      assert(viw == iw);
    if (vih)
      assert(vih == ih);
    if (iw == 0)
      iw = viw;
    if (ih == 0)
      ih = vih;

    if (i == 0) {
      ic = vic;
      kbufn += iw * ih * ic * sizeof(double);
      if (icp)
        *icp = ic;
    } else {
      assert(ic == vic);
    }

    assert(iw > 0);
    assert(ih > 0);
    assert(ic > 0);


    switch (type) {
    case TYPE_NORM:
      {
        int wmvn = size_norm(ic, iw, ih, oc);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        assert(oc == ic);
        break;
      }
    case TYPE_BIAS:
      {
        int wmvn = size_bias(ic, iw, ih, oc);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        assert(oc == ic);
        break;
      }
    case TYPE_LOC0:
      {
        int wmvn = size_local(0, ic, iw, ih, oc);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_LOC1:
      {
        int wmvn = size_local(1, ic, iw, ih, oc);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_LOC2:
      {
        int wmvn = size_local(2, ic, iw, ih, oc);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_PADD:
      {
        assert(len == 0);
        assert(ic == oc);
        ow = iw + rad * 2;
        oh = ih + rad * 2;
        break;
      }
    case TYPE_CROP:
      {
        assert(len == 0);
        assert(ic == oc);
        assert(iw > rad);
        assert(ih > rad);
        ow = rad;
        oh = rad;
        break;
      }
    case TYPE_TRIM:
      {
        assert(len == 0);
        assert(ic == oc);
        assert(iw > rad * 2);
        assert(ih > rad * 2);
        ow = iw - rad * 2;
        oh = ih - rad * 2;
        break;
      }
    case TYPE_CONV:
      {
        int wmvn = size_conv(rad, ic, oc, dim);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_RCN0:
    case TYPE_CON0:
      {
        int wmvn = size_conv(0, ic, oc, dim);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_RCN1:
    case TYPE_CON1:
      {
        int wmvn = size_conv(1, ic, oc, dim);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_RCN2:
    case TYPE_CON2:
      {
        int wmvn = size_conv(2, ic, oc, dim);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_RCN3:
    case TYPE_CON3:
      {
        int wmvn = size_conv(3, ic, oc, dim);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_SCAL:
      {
        int s = _scale_factor(ic, oc);
        if (s > 0) {
          ow = (iw << s);
          oh = (ih << s);
        } else if (s < 0) {
          ow = (iw >> -s);
          oh = (ih >> -s);
          assert((ow << -s) == iw);
          assert((oh << -s) == ih);
        } else {
          ow = iw;
          oh = ih;
        }
        break;
      }
    case TYPE_UPS1:
      {
        int s = 1;
        if (dim == 1) {
          assert(len == 0);
          assert((oc << s) == ic);
          ow = (iw << s);
          oh = ih;
        } else {
          assert(len == 0);
          assert((oc << (s + s)) == ic);
          ow = (iw << s);
          oh = (ih << s);
        }
        break;
      }
    case TYPE_UPS2:
      {
        int s = 2;
        if (dim == 1) {
          assert(len == 0);
          assert((oc << s) == ic);
          ow = (iw << s);
          oh = ih;
        } else {
          assert(len == 0);
          assert((oc << (s + s)) == ic);
          ow = (iw << s);
          oh = (ih << s);
        }
        break;
      }
    case TYPE_UPS3:
      {
        int s = 3;
        if (dim == 1) {
          assert(len == 0);
          assert((oc << s) == ic);
          ow = (iw << s);
          oh = ih;
        } else {
          assert(len == 0);
          assert((oc << (s + s)) == ic);
          ow = (iw << s);
          oh = (ih << s);
        }
        break;
      }
    case TYPE_UPS4:
      {
        int s = 4;
        if (dim == 1) {
          assert(len == 0);
          assert((oc << s) == ic);
          ow = (iw << s);
          oh = ih;
        } else {
          assert(len == 0);
          assert((oc << (s + s)) == ic);
          ow = (iw << s);
          oh = (ih << s);
        }
        break;
      }
    case TYPE_UPS5:
      {
        int s = 5;
        if (dim == 1) {
          assert(len == 0);
          assert((oc << s) == ic);
          ow = (iw << s);
          oh = ih;
        } else {
          assert(len == 0);
          assert((oc << (s + s)) == ic);
          ow = (iw << s);
          oh = (ih << s);
        }
        break;
      }
    case TYPE_DNS1:
      {
        int s = 1;
        assert(len == 0);
        if (dim == 1) {
          assert((ic << s) == oc);
          ow = (iw >> s);
          oh = ih;
          assert((ow << s) == iw);
          assert(oh == ih);
        } else {
          assert((ic << (s + s)) == oc);
          ow = (iw >> s);
          oh = (ih >> s);
          assert((ow << s) == iw);
          assert((oh << s) == ih);
        }
        break;
      }
    case TYPE_DNS2:
      {
        int s = 2;
        assert(len == 0);
        if (dim == 1) {
          assert((ic << s) == oc);
          ow = (iw >> s);
          oh = ih;
          assert((ow << s) == iw);
          assert(oh == ih);
        } else {
          assert((ic << (s + s)) == oc);
          ow = (iw >> s);
          oh = (ih >> s);
          assert((ow << s) == iw);
          assert((oh << s) == ih);
        }
        break;
      }
    case TYPE_DNS3:
      {
        int s = 3;
        assert(len == 0);
        if (dim == 1) {
          assert((ic << s) == oc);
          ow = (iw >> s);
          oh = ih;
          assert((ow << s) == iw);
          assert(oh == ih);
        } else {
          assert((ic << (s + s)) == oc);
          ow = (iw >> s);
          oh = (ih >> s);
          assert((ow << s) == iw);
          assert((oh << s) == ih);
        }
        break;
      }
    case TYPE_DNS4:
      {
        int s = 4;
        assert(len == 0);
        if (dim == 1) {
          assert((ic << s) == oc);
          ow = (iw >> s);
          oh = ih;
          assert((ow << s) == iw);
          assert(oh == ih);
        } else {
          assert((ic << (s + s)) == oc);
          ow = (iw >> s);
          oh = (ih >> s);
          assert((ow << s) == iw);
          assert((oh << s) == ih);
        }
        break;
      }
    case TYPE_DNS5:
      {
        int s = 5;
        assert(len == 0);
        if (dim == 1) {
          assert((ic << s) == oc);
          ow = (iw >> s);
          oh = ih;
          assert((ow << s) == iw);
          assert(oh == ih);
        } else {
          assert((ic << (s + s)) == oc);
          ow = (iw >> s);
          oh = (ih >> s);
          assert((ow << s) == iw);
          assert((oh << s) == ih);
        }
        break;
      }
    case TYPE_ABSV:
    case TYPE_RELU:
    case TYPE_CLMP:
    case TYPE_SIGM:
    case TYPE_NERF:
    case TYPE_INRF:
      ow = iw;
      oh = ih;
      assert(ic == oc);
      assert(len == 0);
      break;
    case TYPE_GRND:
    case TYPE_LRND:
    case TYPE_PAD1:
      ow = iw;
      oh = ih;
      assert(oc >= ic);
      assert(len == 0);
      break;
    case TYPE_ADDG:
      ow = iw;
      oh = ih;
      assert(oc == ic + 4);
      assert(len == 0);
      break;
    case TYPE_IDEN:
      ow = iw;
      oh = ih;
      assert(ic > 0);
      assert(oc > 0);
      assert(len == 0);
      break;
    case TYPE_ZERO:
      ow = iw;
      oh = ih;
      assert(ic > 0);
      assert(oc > 0);
      assert(len == 0);
      break;
    case TYPE_MEDI:
    case TYPE_BLUR:
      ow = iw;
      oh = ih;
      assert(oc == ic);
      assert(len == 0);
      break;
    case TYPE_MEAN:
    case TYPE_SUMM:
    case TYPE_SMAX:
      ow = iw;
      oh = ih;
      assert(oc > 0);
      assert(len == 0);
      break;
    default:
      assert(0);
    }

    if (vow)
      assert(vow == ow);
    if (voh)
      assert(voh == oh);
    assert(ow > 0);
    assert(oh > 0);
    assert(oc > 0);

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    kbufn += (size_t)ow * (size_t)oh * (size_t)oc * sizeof(double);

    assert(len % 3 == 0);
    *dwnp += len / 3;

    iw = ow;
    ih = oh;
    ic = oc;
    ++i;
  }

  if (owp)
    *owp = iw;
  if (ohp)
    *ohp = ih;
  if (ocp)
    *ocp = ic;

  assert(basei == basen);
  return kbufn;
}


static double *_klnoise(int bufn, double mul = 1.0) {
  double *buf = new double[bufn];
  for (int i = 0; i < bufn; ++i)
    buf[i] = randgauss() * mul;

  double *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static double *_kgnoise(int bufn, int c, double mul = 1.0) {
  double *buf = new double[bufn];
  for (int i = 0; i < c; ++i)
    buf[i] = randgauss() * mul;
  for (int i = c; i < bufn; ++i)
    buf[i] = buf[i % c];

  double *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static double *_kpadone(int bufn) {
  double *buf = new double[bufn];
  for (int i = 0; i < bufn; ++i)
    buf[i] = 1.0;

  double *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static double *pipe_synth(
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf, double rmul
) {
  if (basen == 0)
    return ((double *)kbuf);
  assert(basen > 0);

  double *in = (double *)kbuf;

  Cortex::Segment hdr;
  assert(sizeof(hdr) <= basen);
  memcpy(&hdr, base, sizeof(hdr));
  base += sizeof(hdr);
  kbase += sizeof(hdr);
  basen -= sizeof(hdr);

  uint32_t type;
  int len, ic, oc;
  int viw, vih, vow, voh;
  double mul;
  int rad, freeze, relu;
  int dim;
  _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

  if (viw)
    assert(viw == iw);
  if (vih)
    assert(vih == ih);

  int ow, oh;
  kbuf += iw * ih * ic * sizeof(double);
  double *out = (double *)kbuf;

  switch (type) {
  case TYPE_NORM:
    {
      int wmvn = size_norm(ic, iw, ih, oc);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_norm(in, iw, ih, out, ic, oc, wmv);
      break;
    }
  case TYPE_BIAS:
    {
      int wmvn = size_bias(ic, iw, ih, oc);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_bias(in, iw, ih, out, ic, oc, wmv);
      break;
    }
  case TYPE_LOC0:
    {
      int d = 0;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_local(in, iw, ih, out, d, ic, oc, wmv);
      break;
    }
  case TYPE_LOC1:
    {
      int d = 1;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_local(in, iw, ih, out, d, ic, oc, wmv);
      break;
    }
  case TYPE_LOC2:
    {
      int d = 2;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_local(in, iw, ih, out, d, ic, oc, wmv);
      break;
    }
  case TYPE_CROP:
    {
      int d = rad;

      ow = rad;
      oh = rad;
      assert(ic == oc);
      assert(ow > 0);
      assert(oh > 0);
      assert(iw >= ow);
      assert(ih >= oh);

      int ix0 = randuint() % (iw - ow + 1);
      int iy0 = randuint() % (ih - oh + 1);

      synth_crop(in, iw, ih, out, rad, ic, ix0, iy0);
      break;
    }
  case TYPE_TRIM:
    {
      int d = rad;

      ow = iw - rad * 2;
      oh = ih - rad * 2;
      assert(ic == oc);
      assert(ow > 0);
      assert(oh > 0);

      synth_trim(in, iw, ih, out, rad, ic);
      break;
    }
  case TYPE_PADD:
    {
      int d = rad;

      ow = iw + rad * 2;
      oh = ih + rad * 2;
      assert(ic == oc);

      synth_padd(in, iw, ih, out, rad, ic);
      break;
    }
  case TYPE_CONV:
    {
      int d = rad;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, relu, freeze, dim, wmv);
      break;
    }
  case TYPE_RCN0:
  case TYPE_CON0:
    {
      int d = 0;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, type == TYPE_RCN0, 0, dim, wmv);
      break;
    }
  case TYPE_RCN1:
  case TYPE_CON1:
    {
      int d = 1;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, type == TYPE_RCN1, 0, dim, wmv);
      break;
    }
  case TYPE_RCN2:
  case TYPE_CON2:
    {
      int d = 2;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, type == TYPE_RCN2, 0, dim, wmv);
      break;
    }
  case TYPE_RCN3:
  case TYPE_CON3:
    {
      int d = 3;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, type == TYPE_RCN3, 0, dim, wmv);
      break;
    }
  case TYPE_SCAL:
    {
      int s = _scale_factor(ic, oc);

      ow = (iw << s);
      oh = (ih << s);

      if (s > 0) {
        assert((oc << (s + s)) == ic);
        synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      } else if (s < 0) {
        assert((ic << -(s + s)) == oc);
        synth_downscale(in, iw, ih, out, -s, ic, oc, dim);
      } else {
        assert(ic == oc);
        synth_pad(in, iw, ih, out, ic, oc, NULL);
      }
      break;
    }
  case TYPE_UPS1:
    {
      int s = 1;

      if (dim == 1) {
        ow = (iw << s);
        oh = ih;
        assert((oc << s) == ic);
      } else {
        ow = (iw << s);
        oh = (ih << s);
        assert((oc << (s + s)) == ic);
      }


      synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_UPS2:
    {
      int s = 2;

      if (dim == 1) {
        ow = (iw << s);
        oh = ih;
        assert((oc << s) == ic);
      } else {
        ow = (iw << s);
        oh = (ih << s);
        assert((oc << (s + s)) == ic);
      }


      synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_UPS3:
    {
      int s = 3;

      if (dim == 1) {
        ow = (iw << s);
        oh = ih;
        assert((oc << s) == ic);
      } else {
        ow = (iw << s);
        oh = (ih << s);
        assert((oc << (s + s)) == ic);
      }


      synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_UPS4:
    {
      int s = 4;

      if (dim == 1) {
        ow = (iw << s);
        oh = ih;
        assert((oc << s) == ic);
      } else {
        ow = (iw << s);
        oh = (ih << s);
        assert((oc << (s + s)) == ic);
      }


      synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_UPS5:
    {
      int s = 5;

      if (dim == 1) {
        ow = (iw << s);
        oh = ih;
        assert((oc << s) == ic);
      } else {
        ow = (iw << s);
        oh = (ih << s);
        assert((oc << (s + s)) == ic);
      }


      synth_upscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_DNS1:
    {
      int s = 1;

      if (dim == 1) {
        ow = (iw >> s);
        oh = ih;
        assert(iw == (ow << s));
        assert(ih == oh);
        assert((ic << s) == oc);
      } else {
        ow = (iw >> s);
        oh = (ih >> s);
        assert(iw == (ow << s));
        assert(ih == (oh << s));
        assert((ic << (s + s)) == oc);
      }

      synth_downscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_DNS2:
    {
      int s = 2;

      if (dim == 1) {
        ow = (iw >> s);
        oh = ih;
        assert(iw == (ow << s));
        assert(ih == oh);
        assert((ic << s) == oc);
      } else {
        ow = (iw >> s);
        oh = (ih >> s);
        assert(iw == (ow << s));
        assert(ih == (oh << s));
        assert((ic << (s + s)) == oc);
      }

      synth_downscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_DNS3:
    {
      int s = 3;

      if (dim == 1) {
        ow = (iw >> s);
        oh = ih;
        assert(iw == (ow << s));
        assert(ih == oh);
        assert((ic << s) == oc);
      } else {
        ow = (iw >> s);
        oh = (ih >> s);
        assert(iw == (ow << s));
        assert(ih == (oh << s));
        assert((ic << (s + s)) == oc);
      }

      synth_downscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_DNS4:
    {
      int s = 4;

      if (dim == 1) {
        ow = (iw >> s);
        oh = ih;
        assert(iw == (ow << s));
        assert(ih == oh);
        assert((ic << s) == oc);
      } else {
        ow = (iw >> s);
        oh = (ih >> s);
        assert(iw == (ow << s));
        assert(ih == (oh << s));
        assert((ic << (s + s)) == oc);
      }

      synth_downscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_DNS5:
    {
      int s = 5;

      if (dim == 1) {
        ow = (iw >> s);
        oh = ih;
        assert(iw == (ow << s));
        assert(ih == oh);
        assert((ic << s) == oc);
      } else {
        ow = (iw >> s);
        oh = (ih >> s);
        assert(iw == (ow << s));
        assert(ih == (oh << s));
        assert((ic << (s + s)) == oc);
      }

      synth_downscale(in, iw, ih, out, s, ic, oc, dim);
      break;
    }
  case TYPE_SIGM:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_sigm(in, iw, ih, out, ic);
      break;
    }
  case TYPE_CLMP:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_clamp(in, iw, ih, out, ic);
      break;
    }
  case TYPE_RELU:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_relu(in, iw, ih, out, ic);
      break;
    }
  case TYPE_NERF:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_nerf(in, iw, ih, out, ic);
      break;
    }
  case TYPE_INRF:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_inrf(in, iw, ih, out, ic);
      break;
    }
  case TYPE_ABSV:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_abs(in, iw, ih, out, ic);
      break;
    }
  case TYPE_LRND:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      double *knz = _klnoise((oc - ic) * ow * oh, mul * rmul);
      synth_pad(in, iw, ih, out, ic, oc, knz);
      kfree(knz);
      break;
    }
  case TYPE_GRND:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      double *knz = _kgnoise((oc - ic) * ow * oh, oc - ic, mul * rmul);
      synth_pad(in, iw, ih, out, ic, oc, knz);
      kfree(knz);
      break;
    }
  case TYPE_ADDG:
    {
      assert(len == 0);
      assert(ic + 4 == oc);
      ow = iw;
      oh = ih;

      synth_addgeo(in, iw, ih, out, ic, oc);
      break;
    }
  case TYPE_PAD1:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      double *knz = _kpadone((oc - ic) * ow * oh);
      synth_pad(in, iw, ih, out, ic, oc, knz);
      kfree(knz);
      break;
    }
  case TYPE_ZERO:
    {
      assert(len == 0);
      assert(ic > 0);
      assert(oc > 0);
      ow = iw;
      oh = ih;
      synth_zero(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_IDEN:
    {
      assert(len == 0);
      assert(ic > 0);
      assert(oc > 0);
      ow = iw;
      oh = ih;
      synth_iden(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_SMAX:
    {
      assert(len == 0);
      assert(ic % oc == 0);
      ow = iw;
      oh = ih;
      synth_smax(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_MEAN:
    {
      assert(len == 0);
      assert(ic % oc == 0);
      ow = iw;
      oh = ih;
      synth_mean(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_SUMM:
    {
      assert(len == 0);
      ow = iw;
      oh = ih;
      synth_sum(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_MEDI:
    {
      assert(ic == oc);
      assert(len == 0);
      ow = iw;
      oh = ih;
      synth_median(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_BLUR:
    {
      assert(ic == oc);
      assert(len == 0);
      ow = iw;
      oh = ih;
      synth_blur(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  default:
    assert(0);
  }

  if (vow)
    assert(vow == ow);
  if (vih)
    assert(voh == oh);

  assert(len * sizeof(double) <= basen);
  base += len * sizeof(double);
  kbase += len * sizeof(double);
  basen -= len * sizeof(double);

  return pipe_synth(
    base, basen, kbase,
    ow, oh, kbuf, rmul
  );
}




void pipe_learn(
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf,
  double nu, double b1, double b2, double eps, double clip, uint64_t rounds, double *kdw
) {
  if (basen == 0)
    return;
  assert(basen > 0);

  Cortex::Segment hdr;
  assert(sizeof(hdr) <= basen);
  memcpy(&hdr, base, sizeof(hdr));
  base += sizeof(hdr);
  kbase += sizeof(hdr);
  basen -= sizeof(hdr);

  uint32_t type;
  int len, ic, oc;
  int viw, vih, vow, voh;
  int rad, freeze, relu;
  double mul;
  int dim;
  _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

  if (viw)
    assert(viw == iw);
  if (vih)
    assert(vih == ih);

  int ow, oh;

  switch (type) {
  case TYPE_NORM:
    {
      int wmvn = size_norm(ic, iw, ih, oc);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_BIAS:
    {
      int wmvn = size_bias(ic, iw, ih, oc);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_LOC0:
    {
      int d = 0;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_LOC1:
    {
      int d = 1;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_LOC2:
    {
      int d = 2;
      int wmvn = size_local(d, ic, iw, ih, oc);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_CROP:
    {
      ow = rad;
      oh = rad;
      break;
    }
  case TYPE_TRIM:
    {
      ow = iw - rad * 2;
      oh = ih - rad * 2;
      break;
    }
  case TYPE_PADD:
    {
      ow = iw + rad * 2;
      oh = ih + rad * 2;
      break;
    }
  case TYPE_CONV:
    {
      int d = rad;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_RCN0:
  case TYPE_CON0:
    {
      int d = 0;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_RCN1:
  case TYPE_CON1:
    {
      int d = 1;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_RCN2:
  case TYPE_CON2:
    {
      int d = 2;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_RCN3:
  case TYPE_CON3:
    {
      int d = 3;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_SCAL:
    {
      int s = _scale_factor(ic, oc);
      if (s > 0) {
        ow = (iw << s);
        oh = (ih << s);
      } else if (s < 0) {
        ow = (iw >> -s);
        oh = (ih >> -s);
        assert(iw == (ow << -s));
        assert(ih == (oh << -s));
      } else {
        ow = iw;
        oh = ih;
      }
      break;
    }
  case TYPE_UPS1:
    {
      int s = 1;
      ow = (iw << s);
      oh = (dim == 1) ? ih : (ih << s);
      break;
    }
  case TYPE_UPS2:
    {
      int s = 2;
      ow = (iw << s);
      oh = (dim == 1) ? ih : (ih << s);
      break;
    }
  case TYPE_UPS3:
    {
      int s = 3;
      ow = (iw << s);
      oh = (dim == 1) ? ih : (ih << s);
      break;
    }
  case TYPE_UPS4:
    {
      int s = 4;
      ow = (iw << s);
      oh = (dim == 1) ? ih : (ih << s);
      break;
    }
  case TYPE_UPS5:
    {
      int s = 5;
      ow = (iw << s);
      oh = (dim == 1) ? ih : (ih << s);
      break;
    }
  case TYPE_DNS1:
    {
      int s = 1;
      ow = (iw >> s);
      assert(iw == (ow << s));
      oh = (dim == 1) ? ih : (ih >> s);
      assert(ih == (dim == 1) ? oh : (oh << s));
      break;
    }
  case TYPE_DNS2:
    {
      int s = 2;
      ow = (iw >> s);
      assert(iw == (ow << s));
      oh = (dim == 1) ? ih : (ih >> s);
      assert(ih == (dim == 1) ? oh : (oh << s));
      break;
    }
  case TYPE_DNS3:
    {
      int s = 3;
      ow = (iw >> s);
      assert(iw == (ow << s));
      oh = (dim == 1) ? ih : (ih >> s);
      assert(ih == (dim == 1) ? oh : (oh << s));
      break;
    }
  case TYPE_DNS4:
    {
      int s = 4;
      ow = (iw >> s);
      assert(iw == (ow << s));
      oh = (dim == 1) ? ih : (ih >> s);
      assert(ih == (dim == 1) ? oh : (oh << s));
      break;
    }
  case TYPE_DNS5:
    {
      int s = 5;
      ow = (iw >> s);
      assert(iw == (ow << s));
      oh = (dim == 1) ? ih : (ih >> s);
      assert(ih == (dim == 1) ? oh : (oh << s));
      break;
    }
  case TYPE_SIGM:
  case TYPE_RELU:
  case TYPE_CLMP:
  case TYPE_ABSV:
  case TYPE_NERF:
  case TYPE_INRF:
    assert(ic == oc);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_LRND:
  case TYPE_GRND:
  case TYPE_PAD1:
    assert(ic <= oc);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_ADDG:
    assert(ic + 4 == oc);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_ZERO:
    assert(ic > 0);
    assert(oc > 0);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_IDEN:
    assert(ic > 0);
    assert(oc > 0);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_MEDI:
  case TYPE_BLUR:
    assert(oc == ic);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_SMAX:
  case TYPE_MEAN:
  case TYPE_SUMM:
    assert(oc > 0);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  default:
    assert(0);
  }

  if (vow)
    assert(vow == ow);
  if (voh)
    assert(voh == oh);

  double *in = (double *)kbuf;
  kbuf += iw * ih * ic * sizeof(double);
  double *fout = (double *)kbuf;

  double *wmv = (double *)kbase;
  assert(len * sizeof(double) <= basen);
  base += len * sizeof(double);
  kbase += len * sizeof(double);
  basen -= len * sizeof(double);

  pipe_learn(
    base, basen, kbase,
    ow, oh, kbuf,
    nu, b1, b2, eps, clip, rounds,
    kdw ? (kdw + len / 3) : kdw
  );

  switch (type) {
  case TYPE_NORM:
    learn_norm(in, iw, ih, fout, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_BIAS:
    learn_bias(in, iw, ih, fout, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_LOC0:
    learn_local(in, iw, ih, fout, 0, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_LOC1:
    learn_local(in, iw, ih, fout, 1, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_LOC2:
    learn_local(in, iw, ih, fout, 2, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;

  case TYPE_CROP:
    learn_crop(in, iw, ih, fout, rad, ic);
    break;
  case TYPE_TRIM:
    learn_trim(in, iw, ih, fout, rad, ic);
    break;

  case TYPE_PADD:
    learn_padd(in, iw, ih, fout, rad, ic);
    break;

  case TYPE_CONV:
    learn_conv(in, iw, ih, fout, rad, ic, oc, relu, freeze, dim, wmv, kdw, nu * mul, b1, b2, eps, clip, (double)rounds);
    break;

  case TYPE_CON0:
  case TYPE_RCN0:
    learn_conv(in, iw, ih, fout, 0, ic, oc, type == TYPE_RCN0, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_CON1:
  case TYPE_RCN1:
    learn_conv(in, iw, ih, fout, 1, ic, oc, type == TYPE_RCN1, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_CON2:
  case TYPE_RCN2:
    learn_conv(in, iw, ih, fout, 2, ic, oc, type == TYPE_RCN2, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_CON3:
  case TYPE_RCN3:
    learn_conv(in, iw, ih, fout, 3, ic, oc, type == TYPE_RCN3, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (double)rounds);
    break;
  case TYPE_SCAL:
    {
      int s = _scale_factor(ic, oc);
      if (s > 0)
        learn_upscale(in, iw, ih, fout, s, ic, oc, dim);
      else if (s < 0) 
        learn_downscale(in, iw, ih, fout, -s, ic, oc, dim);
      else
        learn_pad(in, iw, ih, fout, ic, oc);
    }
    break;
  case TYPE_UPS1:
    learn_upscale(in, iw, ih, fout, 1, ic, oc, dim);
    break;
  case TYPE_UPS2:
    learn_upscale(in, iw, ih, fout, 2, ic, oc, dim);
    break;
  case TYPE_UPS3:
    learn_upscale(in, iw, ih, fout, 3, ic, oc, dim);
    break;
  case TYPE_UPS4:
    learn_upscale(in, iw, ih, fout, 4, ic, oc, dim);
    break;
  case TYPE_UPS5:
    learn_upscale(in, iw, ih, fout, 5, ic, oc, dim);
    break;
  case TYPE_DNS1:
    learn_downscale(in, iw, ih, fout, 1, ic, oc, dim);
    break;
  case TYPE_DNS2:
    learn_downscale(in, iw, ih, fout, 2, ic, oc, dim);
    break;
  case TYPE_DNS3:
    learn_downscale(in, iw, ih, fout, 3, ic, oc, dim);
    break;
  case TYPE_DNS4:
    learn_downscale(in, iw, ih, fout, 4, ic, oc, dim);
    break;
  case TYPE_DNS5:
    learn_downscale(in, iw, ih, fout, 5, ic, oc, dim);
    break;
  case TYPE_SIGM:
    learn_sigm(in, iw, ih, fout, ic);
    break;
  case TYPE_CLMP:
    learn_clamp(in, iw, ih, fout, ic);
    break;
  case TYPE_RELU:
    learn_relu(in, iw, ih, fout, ic);
    break;
  case TYPE_NERF:
    learn_nerf(in, iw, ih, fout, ic);
    break;
  case TYPE_INRF:
    learn_inrf(in, iw, ih, fout, ic);
    break;
  case TYPE_ABSV:
    learn_abs(in, iw, ih, fout, ic);
    break;
  case TYPE_LRND:
  case TYPE_GRND:
  case TYPE_PAD1:
    learn_pad(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_ADDG:
    learn_addgeo(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_ZERO:
    learn_zero(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_IDEN:
    learn_iden(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_MEDI:
    learn_median(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_BLUR:
    learn_blur(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_SUMM:
    learn_sum(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_SMAX:
    learn_smax(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_MEAN:
    learn_mean(in, iw, ih, fout, ic, oc);
    break;
  default:
    assert(0);
  }
}







Cortex::Cortex() {
  _clear();
}

Cortex::Cortex(const std::string &_fn, int flags) {
  _clear();
  open(_fn, flags);
}

Cortex::~Cortex() {
  if (is_open)
    close();
}

void Cortex::_clear() {
  is_open = false;
  is_prep = false;
  fn = "";
  fd = -1;
  base = NULL;
  basen = 0;
  kbase = NULL;

  iw = ih = ic = 0;
  ow = oh = oc = 0;
  iwhc = owhc = 0;

  kbufn = 0;
  kbuf = NULL;
  kinp = NULL;
  kout = NULL;

  nu = 1e-4;
  b1 = 0.1;
  b2 = 0.001;
  eps = 1e-8;
  clip = 0;

  decay = 0.0;
  rms = 0.0;
  max = 0.0;
  rounds = 0;

  new_rounds = 0;

  rmul = 1.0;
}

void Cortex::open(const std::string &_fn, int flags) {
  assert(!is_open);
  fn = _fn;

  if (startswith(fn, "//"))
    fn = "/opt/makemore/share/tewel/" + std::string(fn.c_str() + 2);

  assert(flags == O_RDWR || flags == O_RDONLY);
  fd = ::open(fn.c_str(), flags, 0644);
  if (fd < 0)
    error(fmt("%s: %s", fn.c_str(), strerror(errno)));

  struct stat stbuf;
  int ret = ::fstat(fd, &stbuf);
  assert(ret == 0);

  basen = stbuf.st_size;
  assert(basen >= sizeof(Head));

  void *vbase = ::mmap(
    NULL, (basen + sizeof(Head) + 4095) & ~4095,
    PROT_READ | (flags == O_RDWR ? PROT_WRITE : 0), MAP_SHARED,
    fd, 0
  );
  assert(vbase != NULL && vbase != MAP_FAILED);
  head = (Head *)vbase;
  base = (uint8_t *)vbase + sizeof(Head);
  basen -= sizeof(Head);

  assert(!memcmp(head->magic, "MakeMore", 8));
  uint32_t v = ntohl(head->version);
  assert(v == 2);

  for (unsigned int j = 0; j < sizeof(Head::blank); ++j)
    assert(head->blank[j] == 0);

  rounds = ntohll(head->rounds);
  new_rounds = 0;

  rms = head->rms;
  max = head->max;
  decay = head->decay > 0 ? head->decay : 0.01;
  assert(decay > 0);

  nu = head->nu;
  b1 = head->b1;
  b2 = head->b2;
  eps = head->eps;
  clip = head->clip;

  int i = 0;
  unsigned int basei = 0;
  int tic = 0;
  ic = 0;
  oc = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic;
    _parse_header(hdr, &type, &len, &vic, &oc);

    if (i == 0) {
      ic = vic;
      tic = vic;
    } else {
      assert(tic == vic);
    }

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    tic = oc;
    ++i;
  }
  assert(basei == basen);

  is_open = true;
}

void Cortex::_get_head_op(double **vecp, double **matp, int *wp, int *hp) {
  unsigned int basei = 0;
  Cortex::Segment hdr;
  assert(basei + sizeof(hdr) <= basen);
  memcpy(&hdr, base + basei, sizeof(hdr));
  basei += sizeof(hdr);

  uint32_t type;
  int len, vic, voc;
  _parse_header(hdr, &type, &len, &vic, &voc);

  assert(len % 3 == 0);
  int tlen = len / 3;
  *hp = voc;
  assert(tlen > voc);
  tlen -= voc;
  assert(tlen % voc == 0);
  *wp = tlen / voc;

  const double *wmv = (double *)(base + basei);
  *matp = new double[*wp * *hp];
  for (int y = 0; y < *hp; ++y)
    for (int x = 0; x < *wp; ++x)
      (*matp)[x + y * *wp] = wmv[3 * (voc + x + y * *wp)];
  *vecp = new double[*hp];
  for (int y = 0; y < *hp; ++y)
    (*vecp)[y] = wmv[3 * y];
}

void Cortex::_put_head_op(const double *vec, const double *mat, int w, int h) {
  unsigned int basei = 0;
  Cortex::Segment hdr;
  assert(basei + sizeof(hdr) <= basen);
  memcpy(&hdr, base + basei, sizeof(hdr));
  basei += sizeof(hdr);

  uint32_t type;
  int len, vic, voc;
  _parse_header(hdr, &type, &len, &vic, &voc);

  int tlen = len / 3;
  assert(h == voc);
  assert(tlen > voc);
  tlen -= voc;
  assert(tlen % voc == 0);
  assert(w * voc == tlen);

  double *wmv = (double *)(base + basei);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      wmv[3 * (voc + x + y * w)] = mat[x + y * w];
  for (int y = 0; y < h; ++y)
    wmv[3 * y] = vec[y];
}

void Cortex::_get_tail_op(double **vecp, double **matp, int *wp, int *hp) {
  unsigned int basei = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic, voc;
    _parse_header(hdr, &type, &len, &vic, &voc);

    assert(basei + len * sizeof(double) <= basen);
    if (basei + len * sizeof(double) == basen) {
      int tlen = len / 3;
      *hp = voc;
      assert(tlen > voc);
      tlen -= voc;
      assert(tlen % voc == 0);
      *wp = tlen / voc;

      const double *wmv = (double *)(base + basei);
      *matp = new double[*wp * *hp];
      for (int y = 0; y < *hp; ++y)
        for (int x = 0; x < *wp; ++x)
          (*matp)[x + y * *wp] = wmv[3 * (voc + x + y * *wp)];
      *vecp = new double[*hp];
      for (int y = 0; y < *hp; ++y)
        (*vecp)[y] = wmv[3 * y];
      return;
    }

    basei += len * sizeof(double);
  }
  assert(0);
}

void Cortex::_put_tail_op(const double *vec, const double *mat, int w, int h) {
  unsigned int basei = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic, voc;
    _parse_header(hdr, &type, &len, &vic, &voc);

    assert(basei + len * sizeof(double) <= basen);
    if (basei + len * sizeof(double) == basen) {
      int tlen = len / 3;
      assert(h == voc);
      assert(tlen > voc);
      tlen -= voc;
      assert(tlen % voc == 0);
      assert(w * voc == tlen);

      double *wmv = (double *)(base + basei);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          wmv[3 * (voc + x + y * w)] = mat[x + y * w];
      for (int y = 0; y < h; ++y)
        wmv[3 * y] = vec[y];

      return;
    }

    basei += len * sizeof(double);
  }
  assert(0);
}


void Cortex::close() {
  assert(is_open);

  if (is_prep)
    unprepare();
  assert(!is_prep);

  int ret = ::munmap(base - sizeof(Head), (basen + sizeof(Head) + 4095) & ~4095);
  assert(ret == 0);
  base = NULL;

  if (kbase)
    kfree(kbase);
  kbase = NULL;

  if (kbuf)
    kfree(kbuf);
  kbuf = NULL;
  kbufn = 0;

  assert(fd != -1);
  ::close(fd);
  fd = -1;

  kinp = NULL;
  kout = NULL;
  fn = "";

  is_open = false;
  new_rounds = 0;
  rounds = 0;
}

// static
void Cortex::create(const std::string &_fn, bool clobber) {
  std::string fn = _fn;

  if (startswith(fn, "//"))
    fn = "/opt/makemore/share/tewel/" + std::string(fn.c_str() + 2);

  int fd = ::open(
    fn.c_str(),
    O_RDWR | O_CREAT | (clobber ? O_TRUNC : O_EXCL),
    0644
  );
  if (fd == -1) {
    if (errno == EEXIST)
      error(fn + " already exists, use clobber=1 to overwrite");
    error(std::string(fn) + ": " + strerror(errno));
  }

  Head head;
  assert(sizeof(Head) == 4096);
  memset(&head, 0, sizeof(head));
  memcpy(head.magic, "MakeMore", 8);
  head.version = htonl(2);
  head.rounds = 0;
  head.decay = 0.01;
  head.rdev = 0.1;
  head.rms = 0;
  head.max = 0;
  head.nu = 1e-4;
  head.b1 = 0.1;
  head.b2 = 0.001;
  head.eps = 1e-8;
  head.clip = 0;
  memset(head.blank, 0, sizeof(head.blank));

  int ret = ::write(fd, &head, sizeof(Head));
  assert(ret == sizeof(Head));

  ret = ::close(fd);
  assert(ret == 0);
}

void Cortex::unfreeze(int last) {
  int i = 1;
  unsigned int basei = 0;

  int got = 0;

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));

    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc;
    int viw, vih, vow, voh;
    int rad, freeze, relu;
    double mul;
    int dim;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    if (hdr.freeze) ++got;
    if (!last && got) {
      hdr.freeze = 0;
      memcpy(base + basei - sizeof(hdr), &hdr, sizeof(hdr));
      return;
    }

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    ++i;
  }
  assert(basei == basen);

  if (!got)
    return;
  basei = 0;

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc;
    int viw, vih, vow, voh;
    int rad, freeze, relu;
    double mul;
    int dim;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    if (hdr.freeze) --got;
    if (got == 0) {
      hdr.freeze = 0;
      memcpy(base + basei - sizeof(hdr), &hdr, sizeof(hdr));
      return;
    }


    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    ++i;
  }

  assert(0);
}

void Cortex::dump(FILE *outfp) {
  int i = 1;
  unsigned int basei = 0;

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc;
    int viw, vih, vow, voh;
    int rad, freeze, relu;
    double mul;
    int dim;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    uint32_t ntype = htonl(type);
    char stype[5];
    memcpy(stype, &ntype, 4);
    stype[4] = 0;

    fprintf(outfp, "type=%s ic=%d oc=%d", stype, ic, oc);
    if (viw) fprintf(outfp, " iw=%d", viw);
    if (vih) fprintf(outfp, " ih=%d", vih);
    if (vow) fprintf(outfp, " ow=%d", vow);
    if (voh) fprintf(outfp, " oh=%d", voh);

    if (type == TYPE_LRND || type == TYPE_GRND) {
      fprintf(outfp, " mul=%g", mul);
    } else if (type == TYPE_CONV) {
      fprintf(outfp, " rad=%d", rad);
      fprintf(outfp, " freeze=%d", freeze);
      fprintf(outfp, " relu=%d", relu);
      fprintf(outfp, " dim=%d", dim);
    } else if (type == TYPE_TRIM || type == TYPE_PADD || type == TYPE_CROP) {
      fprintf(outfp, " rad=%d", rad);
    }

    fprintf(outfp, "\n");

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    ++i;
  }
  assert(basei == basen);
}

void Cortex::bindump(FILE *outfp, unsigned int a, unsigned int b) {
  int i = 1;
  unsigned int basei = 0;

  if (a == 0) {
    int ret = fwrite(head, 1, sizeof(Head), outfp);
    assert(ret == sizeof(Head));
  }

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len;
    _parse_header(hdr, &type, &len);

    uint32_t ntype = htonl(type);
    char stype[5];
    memcpy(stype, &ntype, 4);
    stype[4] = 0;

    assert(basei + len * sizeof(double) <= basen);
    assert(i <= 4096);

    if (i >= a && i <= b) {
      int ret = fwrite(&hdr, 1, sizeof(hdr), outfp);
      assert(ret == sizeof(hdr));
      ret = fwrite(base + basei, 1, len * sizeof(double), outfp);
      assert(ret == len * sizeof(double));
    }

    basei += len * sizeof(double);
    ++i;
  }
  assert(basei == basen);
}

void Cortex::unprepare() {
  assert(is_prep);

  assert(kbuf);
  kfree(kbuf);
  kbuf = NULL;
  kbufn = 0;
  kinp = NULL;
  kout = NULL;

  assert(kbase);
  kfree(kbase);
  kbase = NULL;

  iw = ih = ic = 0;
  ow = oh = oc = 0;
  iwhc = owhc = 0;
  rounds = 0;
  is_prep = false;
}

bool Cortex::prepare(int _iw, int _ih) {
  assert(is_open);

  if (is_prep) {
    if (_iw == iw && ih == _ih) {
      return true;
    }
    unprepare();
  }

  assert(!is_prep);
  assert(_iw > 0);
  assert(_ih > 0);

  int vic, voc;
  assert(ic > 0);
  assert(oc > 0);

  dwn = 0;

  kbufn = pipe_prep(base, basen, _iw, _ih, &vic, &ow, &oh, &voc, &dwn);
  if (kbufn == 0)
    return false;
  assert(kbufn > 0);
  kmake(&kbuf, kbufn);
  kinp = (double *)kbuf;
  kout = NULL;

  assert(vic == ic);
  assert(voc == oc);

  iw = _iw;
  ih = _ih;

  assert(ic > 0);
  assert(ow > 0);
  assert(oh > 0);
  assert(oc > 0);

  iwhc = iw * ih * ic;
  owhc = ow * oh * oc;

  assert(!kbase);
  kmake(&kbase, basen);
  enk(base, basen, kbase);

  if (dwn > 0) {
    kmake(&kdw, dwn);
    kfill(kdw, dwn, 0);
  }

  is_prep = true;
  return true;
}

double *Cortex::synth() {
  assert(is_open);
  assert(is_prep);
  kout = pipe_synth(base, basen, kbase, iw, ih, kbuf, rmul);
  return kout;
}

double *Cortex::synth(const double *_kinp) {
  assert(is_open);
  assert(is_prep);
  kcopy(_kinp, iwhc, kinp);
  return synth();
}

void Cortex::stats() {
  double nrms = sqrt(ksumsq(kout, owhc) / (double)owhc);

  double z = pow(1.0 - decay, (double)rounds);
  rms *= (1.0 - z);
  rms *= (1.0 - decay);
  rms += decay * nrms;
  rms *= 1.0 / (1.0 - z * (1.0 - decay));

  double nmax = kmaxabs(kout, owhc);
  max *= (1.0 - z);
  max *= (1.0 - decay);
  max += decay * nmax;
  max *= 1.0 / (1.0 - z * (1.0 - decay));

  ++rounds;
  ++new_rounds;
}

void Cortex::target(const double *ktgt) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  assert(owhc > 0);
  ksubvec(ktgt, kout, owhc, kout);
}

void Cortex::setloss(const double *ktgt) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  assert(owhc > 0);
  kcopy(ktgt, owhc, kout);
}

double *Cortex::propback() {
  assert(is_open);
  assert(is_prep);
  pipe_learn(base, basen, kbase, iw, ih, kbuf, 0, b1, b2, eps, clip, rounds, NULL);
  return kinp;
}

double *Cortex::accumulate() {
  assert(is_open);
  assert(is_prep);
  stats();
  pipe_learn(base, basen, kbase, iw, ih, kbuf, 0, b1, b2, eps, clip, rounds, kdw);
  return kinp;
}

double *Cortex::learn(double mul) {
  assert(is_open);
  assert(is_prep);
  stats();
  pipe_learn(base, basen, kbase, iw, ih, kbuf, nu * mul, b1, b2, eps, clip, rounds, kdw);
  return kinp;
}

double *Cortex::learn(const double *_kout, double mul) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  kcopy(_kout, owhc, kout);
  return learn(mul);
}

void Cortex::load() {
  assert(is_open);
  assert(is_prep);
  enk(base, basen, kbase);
}

void Cortex::save() {
  assert(is_open);
//  assert(is_prep);

  rounds = ntohll(head->rounds) + new_rounds;
  new_rounds = 0;
  head->rounds = htonll(rounds);

  head->max = max;
  head->rms = rms;

  if (head->nu != nu) {
    nu = head->nu;
    warning("nu changed");
  }
  if (head->b1 != b1) {
    b1 = head->b1;
    warning("b1 changed");
  }
  if (head->b2 != b2) {
    b2 = head->b2;
    warning("b2 changed");
  }
  if (head->eps != eps) {
    eps = head->eps;
    warning("eps changed");
  }
  if (head->clip != clip) {
    clip = head->clip;
    warning("clip changed");
  }

  if (head->decay == 0) {
    assert(decay > 0);
    head->decay = decay;
  }
  if (head->decay != decay) {
    decay = head->decay;
    warning("decay changed");
  }

  dek(kbase, basen, base);
  ::msync(base - sizeof(Head), (basen + sizeof(Head) + 4095) & ~4095, MS_ASYNC);
}

void Cortex::scram(double dev) {
  info(fmt("scrambling %s with dev=%g", fn.c_str(), dev));

  assert(is_open);
  assert(!is_prep);

  unsigned int basei = 0;

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len;
    _parse_header(hdr, &type, &len);

    if (len == 0)
      continue;
    assert(len > 0);

    uint8_t *bwmv = base + basei;
    double *wmv = (double *)bwmv;
    assert(len % 3 == 0);

    for (int i = 0; i < len; i += 3) {
      wmv[i + 0] = randgauss() * dev;
      wmv[i + 1] = 0.0;
      wmv[i + 2] = 1.0;
    }

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);
  }

  assert(basei == basen);
}

void Cortex::push(const std::string &stype, int nic, int noc, int niw, int nih, int now, int noh, double mul, int rad, int freeze, int relu, int dim) {
  if (noc <= 0) {
    int ric = nic;

    if      (stype == "dns1")	{ noc = (nic << 2); }
    else if (stype == "dns2")	{ noc = (nic << 4); }
    else if (stype == "dns3")	{ noc = (nic << 6); }
    else if (stype == "dns4")	{ noc = (nic << 8); }
    else if (stype == "ups1")	{ noc = (nic >> 2); ric = (noc << 2); }
    else if (stype == "ups2")	{ noc = (nic >> 4); ric = (noc << 4); }
    else if (stype == "ups3")	{ noc = (nic >> 6); ric = (noc << 6); }
    else if (stype == "ups4")	{ noc = (nic >> 8); ric = (noc << 8); }
    else if (stype == "grnd")	{ noc = (nic + 1); }
    else if (stype == "lrnd")	{ noc = (nic + 1); }
    else if (stype == "pad1")	{ noc = (nic + 1); }
    else if (stype == "iden")	{ noc = nic; }
    else			{ noc = nic; }

    if (ric != nic)
      error("input channels can't divide to upscale");
  }
  assert(noc > 0);

  this->oc = noc;

  assert(stype.length() == 4);
  uint32_t type;
  memcpy(&type, stype.data(), 4);
  type = ntohl(type);

  assert(is_open);
  assert(!is_prep);
  assert(nic > 0);
  assert(oc > 0);


  int len;
  switch (type) {
  case TYPE_NORM:
    assert(now > 0);
    assert(noh > 0);
    len = size_norm(nic, now, noh, oc);
    break;
  case TYPE_BIAS:
    assert(now > 0);
    assert(noh > 0);
    len = size_bias(nic, now, noh, oc);
    break;
  case TYPE_LOC0:
    assert(now > 0);
    assert(noh > 0);
    len = size_local(0, nic, now, noh, oc);
    break;
  case TYPE_LOC1:
    assert(now > 0);
    assert(noh > 0);
    len = size_local(1, nic, now, noh, oc);
    break;
  case TYPE_LOC2:
    assert(now > 0);
    assert(noh > 0);
    len = size_local(2, nic, now, noh, oc);
    break;
  case TYPE_CROP:
  case TYPE_TRIM:
  case TYPE_PADD:
    len = 0;
    break;
  case TYPE_CONV:
    len = size_conv(rad, nic, oc, dim);
    break;
  case TYPE_CON0:
  case TYPE_RCN0:
    len = size_conv(0, nic, oc, dim);
    break;
  case TYPE_CON1:
  case TYPE_RCN1:
    len = size_conv(1, nic, oc, dim);
    break;
  case TYPE_CON2:
  case TYPE_RCN2:
    len = size_conv(2, nic, oc, dim);
    break;
  case TYPE_CON3:
  case TYPE_RCN3:
    len = size_conv(3, nic, oc, dim);
    break;
  case TYPE_SCAL:
  case TYPE_UPS1:
  case TYPE_UPS2:
  case TYPE_UPS3:
  case TYPE_UPS4:
  case TYPE_UPS5:
  case TYPE_DNS1:
  case TYPE_DNS2:
  case TYPE_DNS3:
  case TYPE_DNS4:
  case TYPE_DNS5:
  case TYPE_SIGM:
  case TYPE_RELU:
  case TYPE_CLMP:
  case TYPE_NERF:
  case TYPE_INRF:
  case TYPE_ABSV:
  case TYPE_GRND:
  case TYPE_LRND:
  case TYPE_PAD1:
  case TYPE_ADDG:
  case TYPE_IDEN:
  case TYPE_ZERO:
  case TYPE_MEAN:
  case TYPE_SMAX:
  case TYPE_SUMM:
  case TYPE_BLUR:
  case TYPE_MEDI:
    len = 0;
    break;
  default:
    assert(0);
  }

  size_t new_basen = basen + sizeof(Segment) + len * sizeof(double);

  int ret = ::ftruncate(fd, new_basen + sizeof(Head));
  assert(ret == 0);

  void *vbase = ::mremap(
    base - sizeof(Head),
    (basen + sizeof(Head) + 4095) & ~4095,
    (new_basen + sizeof(Head) + 4095) & ~4095,
    MREMAP_MAYMOVE
  );
  assert(vbase != NULL && vbase != MAP_FAILED);
  head = (Head *)vbase;
  base = (uint8_t *)vbase + sizeof(Head);

  Segment hdr;
  memset(&hdr, 0, sizeof(hdr));
  assert(sizeof(hdr) >= 64);
  memcpy(hdr.magic, "MakeMore", 8);
  hdr.type = htonl(type);
  hdr.len = htonl(len);
  hdr.ic = htonl(nic);
  hdr.oc = htonl(noc);
  hdr.iw = htonl(niw);
  hdr.ih = htonl(nih);
  hdr.ow = htonl(now);
  hdr.oh = htonl(noh);
  hdr.mul = mul - 1.0;
  hdr.freeze = htonl(freeze);
  hdr.rad = htonl(rad);
  hdr.relu = htonl(relu);
  hdr.dim = htonl(dim);

  memcpy(base + basen, &hdr, sizeof(hdr));
  basen = new_basen;
}

void Cortex::pushbuf() {
  assert(kbufn > 0);

  uint8_t *lbuf;
  kmake(&lbuf, kbufn);
  kcopy(kbuf, kbufn, lbuf);

  bufstack.push_front(lbuf);
}

void Cortex::popbuf() {
  assert(kbufn > 0);

  assert(bufstack.begin() != bufstack.end());
  uint8_t *lbuf = bufstack.front();
  bufstack.pop_front();

  kcopy(lbuf, kbufn, kbuf);
  kfree(lbuf);
}


}
