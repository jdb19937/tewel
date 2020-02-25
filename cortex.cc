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

const uint32_t TYPE_CON0 = CC4('c','o','n','0');
const uint32_t TYPE_CON1 = CC4('c','o','n','1');
const uint32_t TYPE_CON2 = CC4('c','o','n','2');
const uint32_t TYPE_UPS1 = CC4('u','p','s','1');
const uint32_t TYPE_UPS2 = CC4('u','p','s','2');
const uint32_t TYPE_UPS3 = CC4('u','p','s','3');
const uint32_t TYPE_UPS4 = CC4('u','p','s','4');
const uint32_t TYPE_DNS1 = CC4('d','n','s','1');
const uint32_t TYPE_DNS2 = CC4('d','n','s','2');
const uint32_t TYPE_DNS3 = CC4('d','n','s','3');
const uint32_t TYPE_DNS4 = CC4('d','n','s','4');
const uint32_t TYPE_RELU = CC4('r','e','l','u');
const uint32_t TYPE_ABSV = CC4('a','b','s','v');
const uint32_t TYPE_GRND = CC4('g','r','n','d');
const uint32_t TYPE_LRND = CC4('l','r','n','d');
const uint32_t TYPE_PAD1 = CC4('p','a','d','1');
const uint32_t TYPE_IDEN = CC4('i','d','e','n');

static void _check_intro(uint8_t *base, size_t basen) {
  assert(basen >= 4096);
  assert(!memcmp(base, "MakeMore", 8));
  uint32_t v;
  memcpy(&v, base + 8, 4);
  v = ntohl(v);

  assert(v != 0);
  assert(v < 256);
  if (v > 1)
    fprintf(stderr, "Warning: future version %u\n", v);

  char blank[4076];
  memset(blank, 0, 4076);
  assert(!memcmp(blank, base + 20, 4076));
}


static void _place_intro(uint8_t *base, size_t basen) {
  assert(basen >= 4096);
  memcpy(base, "MakeMore", 8);
  uint32_t v = htonl(1);
  memcpy(base + 8, &v, 4);
  memset(base + 12, 0, 4084);
}

static void pipe_grow(int fd, uint8_t **basep, size_t basen, size_t new_basen) {
  assert(new_basen >= basen);

  int ret = ::ftruncate(fd, new_basen + 4096);
  assert(ret == 0);

  void *vbase = mremap(
    *basep - 4096,
    (basen + 4096 + 4095) & ~4095,
    (new_basen + 4096 + 4095) & ~4095,
    MREMAP_MAYMOVE
  );
  assert(vbase != NULL && vbase != MAP_FAILED);
  uint8_t *base = (uint8_t *)vbase;
  base += 4096;

  *basep = (uint8_t *)base;
}

static void _make_header(
  layer_header_t& hdr, uint32_t type, int len, int ic, int oc
) {
  memset(hdr, 0, sizeof(hdr));
  assert(sizeof(hdr[0]) == 4);
  assert(sizeof(hdr) >= 64);
  memcpy((char *)hdr, "MakeMore", 8);
  hdr[2] = htonl(1);
  hdr[3] = htonl(type);
  hdr[4] = htonl(len);
  hdr[5] = htonl(ic);
  hdr[6] = htonl(oc);
}

static void _parse_header(
  layer_header_t& hdr, uint32_t *typep, int *lenp, int *icp, int *ocp
) {
  assert(sizeof(hdr[0]) == 4);
  assert(sizeof(hdr) >= 64);

  assert(!memcmp((char *)hdr, "MakeMore", 8));
  uint32_t v = ntohl(hdr[2]);

  assert(v != 0);
  assert(v < 256);
  if (v > 1)
    fprintf(stderr, "Warning: future version %u\n", v);

  *typep = ntohl(hdr[3]);
  *lenp = ntohl(hdr[4]);
  *icp = ntohl(hdr[5]);
  *ocp = ntohl(hdr[6]);
}


static void pipe_push(
  int fd, uint8_t **basep, size_t *basenp, uint32_t type, int ic, int oc
) {
  int len;

  switch (type) {
  case TYPE_CON0:
    len = size_conv(0, ic, oc, false);
    break;
  case TYPE_CON1:
    len = size_conv(1, ic, oc, false);
    break;
  case TYPE_CON2:
    len = size_conv(2, ic, oc, false);
    break;
  case TYPE_UPS1:
  case TYPE_UPS2:
  case TYPE_UPS3:
  case TYPE_UPS4:
  case TYPE_DNS1:
  case TYPE_DNS2:
  case TYPE_DNS3:
  case TYPE_DNS4:
  case TYPE_RELU:
  case TYPE_ABSV:
  case TYPE_GRND:
  case TYPE_LRND:
  case TYPE_PAD1:
  case TYPE_IDEN:
    len = 0;
    break;
  default:
    assert(0);
  }

  size_t new_basen = *basenp + sizeof(layer_header_t) + len * sizeof(double);
  pipe_grow(fd, basep, *basenp, new_basen);

  layer_header_t hdr;
  _make_header(hdr, type, len, ic, oc);

  memcpy((char *)*basep + *basenp, &hdr, sizeof(hdr));
  *basenp = new_basen;
}

static bool pipe_scan(uint8_t *base, size_t basen, int *icp, int *ocp) {
  int i = 0;
  unsigned int basei = 0;
  int ic = 0, oc = 0;

  *icp = 0;
  *ocp = 0;

  if (basei == basen)
    return false;
  assert(basei < basen);

  while (basei < basen) {
    layer_header_t hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic;
    _parse_header(hdr, &type, &len, &vic, &oc);

    if (i == 0) {
      ic = vic;
      *icp = ic;
    } else {
      assert(ic == vic);
    }

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    ic = oc;
    ++i;
  }

  if (ocp)
    *ocp = ic;
  assert(basei == basen);

  return true;
}

static size_t pipe_prep(uint8_t *base, size_t basen, int iw, int ih, int *icp, int *owp, int *ohp, int *ocp, bool stripped) {
  int i = 0;
  size_t kbufn = 0;
  unsigned int basei = 0;
  int ow = 0, oh = 0, ic = 0, oc = 0;

  if (basei == basen)
    return 0;
  assert(basei < basen);

  while (basei < basen) {
    layer_header_t hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic;
    _parse_header(hdr, &type, &len, &vic, &oc);

    if (i == 0) {
      ic = vic;
      kbufn += iw * ih * ic * sizeof(double);
      if (icp)
        *icp = ic;
    } else {
      assert(ic == vic);
    }

    switch (type) {
    case TYPE_CON0:
      {
        int wmvn = size_conv(0, ic, oc, stripped);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_CON1:
      {
        int wmvn = size_conv(1, ic, oc, stripped);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_CON2:
      {
        int wmvn = size_conv(2, ic, oc, stripped);
        assert(wmvn == len);
        ow = iw;
        oh = ih;
        break;
      }
    case TYPE_UPS1:
      {
        int s = 1;
        assert(len == 0);
        assert((oc << (s + s)) == ic);
        ow = (iw << s);
        oh = (ih << s);
        break;
      }
    case TYPE_UPS2:
      {
        int s = 2;
        assert(len == 0);
        assert((oc << (s + s)) == ic);
        ow = (iw << s);
        oh = (ih << s);
        break;
      }
    case TYPE_UPS3:
      {
        int s = 3;
        assert(len == 0);
        assert((oc << (s + s)) == ic);
        ow = (iw << s);
        oh = (ih << s);
        break;
      }
    case TYPE_UPS4:
      {
        int s = 4;
        assert(len == 0);
        assert((oc << (s + s)) == ic);
        ow = (iw << s);
        oh = (ih << s);
        break;
      }
    case TYPE_DNS1:
      {
        int s = 1;
        assert(len == 0);
        assert((ic << (s + s)) == oc);
        ow = (iw >> s);
        oh = (ih >> s);
        assert((ow << s) == iw);
        assert((oh << s) == ih);
        break;
      }
    case TYPE_DNS2:
      {
        int s = 2;
        assert(len == 0);
        assert((ic << (s + s)) == oc);
        ow = (iw >> s);
        oh = (ih >> s);
        assert((ow << s) == iw);
        assert((oh << s) == ih);
        break;
      }
    case TYPE_DNS3:
      {
        int s = 3;
        assert(len == 0);
        assert((ic << (s + s)) == oc);
        ow = (iw >> s);
        oh = (ih >> s);
        assert((ow << s) == iw);
        assert((oh << s) == ih);
        break;
      }
    case TYPE_DNS4:
      {
        int s = 4;
        assert(len == 0);
        assert((ic << (s + s)) == oc);
        ow = (iw >> s);
        oh = (ih >> s);
        assert((ow << s) == iw);
        assert((oh << s) == ih);
        break;
      }
    case TYPE_ABSV:
    case TYPE_RELU:
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
    case TYPE_IDEN:
      ow = iw;
      oh = ih;
      assert(oc == ic);
      assert(len == 0);
      break;
    default:
      assert(0);
    }

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);

    kbufn += (size_t)ow * (size_t)oh * (size_t)oc * sizeof(double);

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


static double *_klnoise(int bufn) {
  double *buf = new double[bufn];
  for (int i = 0; i < bufn; ++i)
    buf[i] = randgauss();

  double *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static double *_kgnoise(int bufn, int c) {
  double *buf = new double[bufn];
  for (int i = 0; i < c; ++i)
    buf[i] = randgauss();
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
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf,
  bool stripped
) {
  if (basen == 0)
    return ((double *)kbuf);
  assert(basen > 0);

  const double *in = (const double *)kbuf;

  layer_header_t hdr;
  assert(sizeof(hdr) <= basen);
  memcpy(hdr, base, sizeof(hdr));
  base += sizeof(hdr);
  kbase += sizeof(hdr);
  basen -= sizeof(hdr);

  uint32_t type;
  int len, ic, oc;
  _parse_header(hdr, &type, &len, &ic, &oc);

  int ow, oh;
  kbuf += iw * ih * ic * sizeof(double);
  double *out = (double *)kbuf;

  switch (type) {
  case TYPE_CON0:
    {
      int d = 0;
      int wmvn = size_conv(d, ic, oc, stripped);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, wmv, stripped);
      break;
    }
  case TYPE_CON1:
    {
      int d = 1;
      int wmvn = size_conv(d, ic, oc, stripped);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, wmv, stripped);
      break;
    }
  case TYPE_CON2:
    {
      int d = 2;
      int wmvn = size_conv(d, ic, oc, stripped);
      assert(wmvn == len);
      double *wmv = (double *)kbase;

      ow = iw;
      oh = ih;

      synth_conv(in, iw, ih, out, d, ic, oc, wmv, stripped);
      break;
    }
  case TYPE_UPS1:
    {
      int s = 1;

      ow = (iw << s);
      oh = (ih << s);
      assert((oc << (s + s)) == ic);

      synth_upscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_UPS2:
    {
      int s = 2;

      ow = (iw << s);
      oh = (ih << s);
      assert((oc << (s + s)) == ic);

      synth_upscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_UPS3:
    {
      int s = 3;

      ow = (iw << s);
      oh = (ih << s);
      assert((oc << (s + s)) == ic);

      synth_upscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_UPS4:
    {
      int s = 4;

      ow = (iw << s);
      oh = (ih << s);
      assert((oc << (s + s)) == ic);

      synth_upscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_DNS1:
    {
      int s = 1;

      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      assert((ic << (s + s)) == oc);

      synth_downscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_DNS2:
    {
      int s = 2;

      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      assert((ic << (s + s)) == oc);

      synth_downscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_DNS3:
    {
      int s = 3;

      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      assert((ic << (s + s)) == oc);

      synth_downscale(in, iw, ih, out, s, ic, oc);
      break;
    }
  case TYPE_DNS4:
    {
      int s = 4;

      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      assert((ic << (s + s)) == oc);

      synth_downscale(in, iw, ih, out, s, ic, oc);
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

      double *knz = _klnoise((oc - ic) * ow * oh);
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

      double *knz = _kgnoise((oc - ic) * ow * oh, oc - ic);
      synth_pad(in, iw, ih, out, ic, oc, knz);
      kfree(knz);
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
  case TYPE_IDEN:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      synth_pad(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  default:
    assert(0);
  }

  assert(len * sizeof(double) <= basen);
  base += len * sizeof(double);
  kbase += len * sizeof(double);
  basen -= len * sizeof(double);

  return pipe_synth(
    base, basen, kbase,
    ow, oh, kbuf, stripped
  );
}




void pipe_learn(
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf,
  double nu, double b1, double b2, double eps, uint64_t rounds
) {
  if (basen == 0)
    return;
  assert(basen > 0);

  layer_header_t hdr;
  assert(sizeof(hdr) <= basen);
  memcpy(hdr, base, sizeof(hdr));
  base += sizeof(hdr);
  kbase += sizeof(hdr);
  basen -= sizeof(hdr);

  uint32_t type;
  int len, ic, oc;
  _parse_header(hdr, &type, &len, &ic, &oc);

  int ow, oh;

  switch (type) {
  case TYPE_CON0:
    {
      int d = 0;
      int wmvn = size_conv(d, ic, oc, false);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_CON1:
    {
      int d = 1;
      int wmvn = size_conv(d, ic, oc, false);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_CON2:
    {
      int d = 2;
      int wmvn = size_conv(d, ic, oc, false);
      assert(wmvn == len);

      ow = iw;
      oh = ih;

      break;
    }
  case TYPE_UPS1:
    {
      int s = 1;
      ow = (iw << s);
      oh = (ih << s);
      break;
    }
  case TYPE_UPS2:
    {
      int s = 2;
      ow = (iw << s);
      oh = (ih << s);
      break;
    }
  case TYPE_UPS3:
    {
      int s = 3;
      ow = (iw << s);
      oh = (ih << s);
      break;
    }
  case TYPE_UPS4:
    {
      int s = 4;
      ow = (iw << s);
      oh = (ih << s);
      break;
    }
  case TYPE_DNS1:
    {
      int s = 1;
      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      break;
    }
  case TYPE_DNS2:
    {
      int s = 2;
      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      break;
    }
  case TYPE_DNS3:
    {
      int s = 3;
      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      break;
    }
  case TYPE_DNS4:
    {
      int s = 4;
      ow = (iw >> s);
      oh = (ih >> s);
      assert(iw == (ow << s));
      assert(ih == (oh << s));
      break;
    }
  case TYPE_RELU:
  case TYPE_ABSV:
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
  case TYPE_IDEN:
    assert(ic == oc);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  default:
    assert(0);
  }

  double *in = (double *)kbuf;
  kbuf += iw * ih * ic * sizeof(double);
  double *fout = (double *)kbuf;

  double *wmv = (double *)kbase;
  assert(len * sizeof(double) <= basen);
  base += len * sizeof(double);
  kbase += len * sizeof(double);
  basen -= len * sizeof(double);

  //fprintf(stderr, "rrr=%lf\n", ksumsq(in, iw * ih * ic));

  pipe_learn(
    base, basen, kbase,
    ow, oh, kbuf,
    nu, b1, b2, eps, rounds
  );

  switch (type) {
  case TYPE_CON0:
    learn_conv(in, iw, ih, fout, 0, ic, oc, wmv, nu, b1, b2, eps, (double)rounds);
    break;
  case TYPE_CON1:
    learn_conv(in, iw, ih, fout, 1, ic, oc, wmv, nu, b1, b2, eps, (double)rounds);
    break;
  case TYPE_CON2:
    learn_conv(in, iw, ih, fout, 2, ic, oc, wmv, nu, b1, b2, eps, (double)rounds);
    break;
  case TYPE_UPS1:
    learn_upscale(in, iw, ih, fout, 1, ic, oc);
    break;
  case TYPE_UPS2:
    learn_upscale(in, iw, ih, fout, 2, ic, oc);
    break;
  case TYPE_UPS3:
    learn_upscale(in, iw, ih, fout, 3, ic, oc);
    break;
  case TYPE_UPS4:
    learn_upscale(in, iw, ih, fout, 4, ic, oc);
    break;
  case TYPE_DNS1:
    learn_downscale(in, iw, ih, fout, 1, ic, oc);
    break;
  case TYPE_DNS2:
    learn_downscale(in, iw, ih, fout, 2, ic, oc);
    break;
  case TYPE_DNS3:
    learn_downscale(in, iw, ih, fout, 3, ic, oc);
    break;
  case TYPE_DNS4:
    learn_downscale(in, iw, ih, fout, 4, ic, oc);
    break;
  case TYPE_RELU:
    learn_relu(in, iw, ih, fout, ic);
    break;
  case TYPE_ABSV:
    learn_abs(in, iw, ih, fout, ic);
    break;
  case TYPE_LRND:
  case TYPE_GRND:
  case TYPE_PAD1:
  case TYPE_IDEN:
    learn_pad(in, iw, ih, fout, ic, oc);
    break;
  default:
    assert(0);
  }
}







Cortex::Cortex() {
  _clear();
}

Cortex::Cortex(const std::string &_fn) {
  _clear();
  open(_fn);
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

  kbuf = NULL;
  kfakebuf = NULL;
  kinp = NULL;
  kout = NULL;

  nu = 1e-4;
  b1 = 0.1;
  b2 = 0.001;
  eps = 1e-8;

  decay = 0.01;
  rms = 0.0;
  max = 0.0;
  rounds = 0;
  stripped = false;
}

void Cortex::open(const std::string &_fn) {
  assert(!is_open);
  fn = _fn;

  fd = ::open(fn.c_str(), O_RDWR, 0700);
  assert(fd >= 0);

  struct stat stbuf;
  int ret = ::fstat(fd, &stbuf);
  assert(ret == 0);

  basen = stbuf.st_size;
  assert(basen >= 4096);

  void *vbase = ::mmap(
    NULL, (basen + 4095) & ~4095,
    PROT_READ | PROT_WRITE, MAP_SHARED,
    fd, 0
  );
  assert(vbase != NULL && vbase != MAP_FAILED);
  base = (uint8_t *)vbase;

  _check_intro(base, basen);
  base += 4096;
  basen -= 4096;

  pipe_scan(
    base, basen,
    &ic, &oc
  );

  is_open = true;
}

void Cortex::close() {
  assert(is_open);

  if (is_prep)
    unprepare();
  assert(!is_prep);

  int ret = ::munmap(base - 4096, (basen + 4096 + 4095) & ~4095);
  assert(ret == 0);
  base = NULL;

  if (kbase)
    kfree(kbase);
  kbase = NULL;

  if (kbuf)
    kfree(kbuf);
  kbuf = NULL;

  assert(fd != -1);
  ::close(fd);
  fd = -1;

  kinp = NULL;
  kout = NULL;
  fn = "";

  is_open = false;
}

// static
void Cortex::create(const std::string &fn) {
  int fd = ::open(fn.c_str(), O_RDWR | O_CREAT | O_EXCL, 0700);
  assert(fd != -1);

  int ret = ::write(fd, "MakeMore", 8);
  assert(ret == 8);

  uint32_t v = htonl(1);
  ret = ::write(fd, &v, 4);
  assert(ret == 4);

  char rest[4084];
  memset(rest, 0, sizeof(rest));
  ret = ::write(fd, rest, sizeof(rest));
  assert(ret == sizeof(rest));

  ret = ::close(fd);
  assert(ret == 0);
}

void Cortex::dump(FILE *outfp) {
  int i = 0;
  unsigned int basei = 0;

  while (basei < basen) {
    layer_header_t hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc;
    _parse_header(hdr, &type, &len, &ic, &oc);

    uint32_t ntype = htonl(type);
    char stype[5];
    memcpy(stype, &ntype, 4);
    stype[4] = 0;

    fprintf(outfp, "t=%s len=%d ic=%d oc=%d\n", stype, len, ic, oc);

    assert(basei + len * sizeof(double) <= basen);
    basei += len * sizeof(double);
  }
  assert(basei == basen);
}

void Cortex::unprepare() {
  assert(is_prep);

  assert(kbuf);
  kfree(kbuf);
  kbuf = NULL;
  kinp = NULL;
  kout = NULL;

  assert(kbase);
  kfree(kbase);
  kbase = NULL;

  if (kfakebuf) {
    kfree(kfakebuf);
    kfakebuf = NULL;
  }

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

  uint64_t nrounds;
  memcpy(&nrounds, base - 4096 + 12, 8);
  nrounds = ntohl(nrounds);
  rounds = nrounds;

  uint64_t nstripped;
  memcpy(&nstripped, base - 4096 + 20, 8);
  nstripped = ntohl(nstripped);
  assert(nstripped == 0 || nstripped == 1);
  stripped = nstripped;

  int vic, voc;
  assert(ic > 0);
  assert(oc > 0);

  size_t kbufn = pipe_prep(base, basen, _iw, _ih, &vic, &ow, &oh, &voc, stripped);
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

  is_prep = true;
  return true;
}

double *Cortex::synth() {
  assert(is_open);
  assert(is_prep);
  kout = pipe_synth(base, basen, kbase, iw, ih, kbuf, stripped);
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
}

void Cortex::target(const double *ktgt) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  assert(owhc > 0);
  ksubvec(ktgt, kout, owhc, kout);
}

double *Cortex::propback() {
  assert(is_open);
  assert(is_prep);
  assert(!stripped);
  pipe_learn(base, basen, kbase, iw, ih, kbuf, 0, b1, b2, eps, rounds);
  return kinp;
}

double *Cortex::learn(double mul) {
  assert(is_open);
  assert(is_prep);
  assert(!stripped);
  stats();
  pipe_learn(base, basen, kbase, iw, ih, kbuf, nu * mul, b1, b2, eps, rounds);
  return kinp;
}

double *Cortex::learn(const double *_kout, double mul) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  kcopy(_kout, owhc, kout);
  return learn(mul);
}

#if 0
void Cortex::learnauto(const double *kintgt, double mul) {
  assert(is_open);
  assert(is_prep);
  learnfunc(kintgt, kintgt, mul);
}

void Cortex::learnfunc(const double *kin, const double *ktgt, double mul) {
  assert(is_open);
  assert(is_prep);

  double *kout = _synth(kin);
  ksubvec(ktgt, kout, owhc, kout);
  _stats(kout);
  _learn(mul);
}


void Cortex::_target_fake(double *kfakeout) {
  double *kfaketgt;
  kmake(&kfaketgt, owhc);
  kunit(kfaketgt, owhc);
  ksubvec(kfaketgt, kfakeout, iwhc, kfakeout);
  kfree(kfaketgt);

  _stats(kfakeout);
}

void Cortex::_target_real(double *krealout) {
  double *krealtgt;
  kmake(&krealtgt, owhc);
  kzero(krealtgt, owhc);
  ksubvec(krealtgt, krealout, iwhc, krealout);
  kfree(krealtgt);

  _stats(krealout);
}

void Cortex::learnreal(const double *kfake, const double *kreal, double mul) {
  _target_fake(_synth(kfake));
  _learn(mul);
  _target_real(_synth(kreal));
  _learn(mul);
}

double *Cortex::helpfake(const double *kfake) {
  _target_real(_synth(kfake));
  return _learn(0);
}

void Cortex::learnstyl(const double *kin, const double *kreal, Cortex *dis, double mul, double dismul) {
  if (!kfakebuf)
    kmake(&kfakebuf, owhc);

  double *kout = _synth(kin);
  kcopy(kout, owhc, kfakebuf);
  kcopy(dis->helpfake(kfakebuf), owhc, kout);
  _learn(mul);

  learnreal(kfakebuf, kreal, dismul);
}

void Cortex::learnhans(const double *kin, const double *ktgt, Cortex *dis, double nz, double mul, double dismul) {
#if 0
  if (!kfakebuf)
    kmake(&kfakebuf, owhc);

  double *kout = _synth(kin);
  kcopy(kout, owhc, kfakebuf);
  kcopy(dis->helpfake(kfakebuf +++ kin), owhc, kout);
  _learn(mul);

  learnreal(kfakebuf +++ kin, ktgt +++ kin, dismul);
#endif
}
#endif

void Cortex::load() {
  assert(is_open);
  assert(is_prep);
  enk(base, basen, kbase);
}

void Cortex::save() {
  assert(is_open);
  assert(is_prep);

  uint64_t nrounds = htonl(rounds);
  memcpy(base - 4096 + 12, &nrounds, 8);

  dek(kbase, basen, base);
  ::msync(base - 4096, (basen + 4096 + 4095) & ~4095, MS_ASYNC);
}

void Cortex::scram(double dev) {
  assert(is_open);
  assert(!is_prep);

  unsigned int basei = 0;

  while (basei < basen) {
    layer_header_t hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc;
    _parse_header(hdr, &type, &len, &ic, &oc);

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

void Cortex::push(const std::string &stype, int ic, int oc) {
  assert(!stripped);

  assert(stype.length() == 4);
  uint32_t type;
  memcpy(&type, stype.data(), 4);
  type = ntohl(type);

  assert(is_open);
  assert(!is_prep);
  if (oc == 0)
    oc = ic;
  assert(ic > 0);
  assert(oc > 0);
  pipe_push(fd, &base, &basen, type, ic, oc);
}

}
