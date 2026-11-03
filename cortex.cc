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
const uint32_t TYPE_CENT = CC4('c','e','n','t');
const uint32_t TYPE_RELU = CC4('r','e','l','u');
const uint32_t TYPE_LRLU = CC4('l','r','l','u');
const uint32_t TYPE_PNRM = CC4('p','n','r','m');
const uint32_t TYPE_UPBL = CC4('u','p','b','l');
const uint32_t TYPE_DNAV = CC4('d','n','a','v');
const uint32_t TYPE_MBSD = CC4('m','b','s','d');
const uint32_t TYPE_STYL = CC4('s','t','y','l');
const uint32_t TYPE_MCON = CC4('m','c','o','n');
// skip connections: keep (out = in, stash slot), rcal (out = stash slot; oc may differ from ic),
// radd (out = (in + stash slot) * mul).  Slots live in the layer's rad field, each keep has one
// consumer (the next rcal/radd of that slot) and the consumer must precede the slot's next keep.
const uint32_t TYPE_KEEP = CC4('k','e','e','p');
const uint32_t TYPE_RCAL = CC4('r','c','a','l');
const uint32_t TYPE_RADD = CC4('r','a','d','d');
static const int g_nslots = 16;
static float *g_slot_out[g_nslots];        // forward: the kept tensor
static size_t g_slot_n[g_nslots];
static const float *g_slot_grad[g_nslots]; // backward: gradient the consumer hands back to keep
static float g_slot_gmul[g_nslots];
static int g_slot_w[g_nslots], g_slot_h[g_nslots], g_slot_c[g_nslots];   // kept shape (rcal takes its output shape from it)

// per-layer stash buffers (mcon keeps its forward output for the backward pass)
static std::vector<float *> *g_stash = NULL;
static int g_li = 0;
static std::vector<size_t> g_stash_sizes;

// style side input for the pipeline currently being run (set by Cortex before pipe_*)
static const float *g_kstyle = NULL;
static float *g_kstylegrad = NULL;
const uint32_t TYPE_CLMP = CC4('c','l','m','p');
const uint32_t TYPE_ABSV = CC4('a','b','s','v');
const uint32_t TYPE_GRND = CC4('g','r','n','d');
const uint32_t TYPE_LRND = CC4('l','r','n','d');
const uint32_t TYPE_DRND = CC4('d','r','n','d');
const uint32_t TYPE_PAD1 = CC4('p','a','d','1');
const uint32_t TYPE_ADDG = CC4('a','d','d','g');
const uint32_t TYPE_IDEN = CC4('i','d','e','n');
const uint32_t TYPE_POPP = CC4('p','o','p','p');
const uint32_t TYPE_TLAB = CC4('t','l','a','b');
const uint32_t TYPE_FLAB = CC4('f','l','a','b');
const uint32_t TYPE_MEAN = CC4('m','e','a','n');
const uint32_t TYPE_SMAX = CC4('s','m','a','x');
const uint32_t TYPE_SUMM = CC4('s','u','m','m');
const uint32_t TYPE_DIFF = CC4('d','i','f','f');
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
  float *mulp = NULL, int *radp = NULL, int *freezep = NULL, int *relup = NULL, int *dimp = NULL
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


static size_t pipe_prep(uint8_t *base, size_t basen, int iw, int ih, int *icp, int *owp, int *ohp, int *ocp, unsigned long *dwnp, int bn, int *sdimp) {
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
    float _mul;
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
      kbufn += (size_t)bn * iw * ih * ic * sizeof(float);
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
    case TYPE_MBSD:
      assert(len == 0);
      assert(ic + 1 == oc);
      ow = iw;
      oh = ih;
      break;
    case TYPE_STYL:
      assert(ic == oc);
      assert(len == size_styl(ic, rad));
      assert(*sdimp == 0 || *sdimp == rad);
      *sdimp = rad;
      ow = iw;
      oh = ih;
      break;
    case TYPE_MCON:
      assert(len == size_mcon(ic, oc, rad));
      assert(*sdimp == 0 || *sdimp == rad);
      *sdimp = rad;
      ow = iw;
      oh = ih;
      g_stash_sizes.push_back((size_t)bn * ow * oh * oc);
      break;
    case TYPE_UPBL:
      assert(len == 0);
      assert(ic == oc);
      ow = iw * 2;
      oh = ih * 2;
      break;
    case TYPE_DNAV:
      assert(len == 0);
      assert(ic == oc);
      assert(iw % 2 == 0 && ih % 2 == 0);
      ow = iw / 2;
      oh = ih / 2;
      break;
    case TYPE_KEEP:
      assert(len == 0);
      assert(ic == oc);
      assert(rad >= 0 && rad < g_nslots);
      g_slot_w[rad] = iw; g_slot_h[rad] = ih; g_slot_c[rad] = ic;
      ow = iw;
      oh = ih;
      break;
    case TYPE_RADD:
      assert(len == 0);
      assert(ic == oc);
      assert(rad >= 0 && rad < g_nslots);
      ow = iw;
      oh = ih;
      break;
    case TYPE_RCAL:
      assert(len == 0);
      assert(rad >= 0 && rad < g_nslots);
      assert(g_slot_c[rad] == oc);
      ow = g_slot_w[rad];
      oh = g_slot_h[rad];
      break;
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
    case TYPE_LRLU:
    case TYPE_PNRM:
    case TYPE_CLMP:
    case TYPE_SIGM:
    case TYPE_CENT:
    case TYPE_NERF:
    case TYPE_INRF:
      ow = iw;
      oh = ih;
      assert(ic == oc);
      assert(len == 0);
      break;
    case TYPE_GRND:
    case TYPE_LRND:
    case TYPE_DRND:
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
    case TYPE_TLAB:
    case TYPE_FLAB:
      ow = iw;
      oh = ih;
      assert(ic == 3);
      assert(oc == 3);
      assert(len == 0);
      break;
    case TYPE_POPP:
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
    case TYPE_DIFF:
      ow = iw;
      oh = ih;
      assert(oc * 2 == ic);
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

    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);

    kbufn += (size_t)bn * (size_t)ow * (size_t)oh * (size_t)oc * sizeof(float);

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


// device-side pool of unit gaussians, filled once; noise buffers are random windows into it
static float *g_noisepool = NULL;
static const long g_noisepooln = 1L << 24;   // 16M floats, 64 MB

// deterministic inference: when a noise seed is set (>= 0), every noise window is a hash of the
// seed and the call index within the pass (reset at each Cortex::synth), so equal inputs give
// equal outputs; training leaves it at -1 (fresh random windows)
static long g_noiseseed = -1;
static long g_noisecall = 0;
void Cortex::set_noise_seed(long seed) { g_noiseseed = seed; g_noisecall = 0; }
long Cortex::hash_input(const float *x, long n) {
  uint64_t h = 0xcbf29ce484222325ULL;
  const uint32_t *u = (const uint32_t *)x;
  for (long i = 0; i < n; ++i) { h ^= u[i]; h *= 0x100000001b3ULL; }
  return (long)(h & 0x7fffffffffffffffULL);
}

static float *_klnoise(int bufn, float mul = 1.0) {
  if (!g_noisepool) {
    float *buf = new float[g_noisepooln];
    for (long i = 0; i < g_noisepooln; ++i)
      buf[i] = randgauss();
    kmake(&g_noisepool, g_noisepooln);
    enk(buf, g_noisepooln, g_noisepool);
    delete[] buf;
  }
  assert(bufn <= g_noisepooln);
  long off;
  if (g_noiseseed >= 0) {
    uint64_t h = (uint64_t)g_noiseseed * 0x9E3779B97F4A7C15ULL + (uint64_t)(g_noisecall + 1) * 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31; h *= 0x94D049BB133111EBULL; h ^= h >> 29;
    off = (long)(h % (uint64_t)(g_noisepooln - bufn + 1));
  } else {
    off = (long)(randuint() % (uint32_t)(g_noisepooln - bufn + 1));
  }
  ++g_noisecall;
  float *kbuf;
  kmake(&kbuf, bufn);
  kcopy(g_noisepool + off, bufn, kbuf);
  if (mul != 1.0)
    kmul(kbuf, mul, bufn, kbuf);
  return kbuf;
}

static float *_kgnoise(int bufn, int c, float mul = 1.0) {
  float *buf = new float[bufn];
  for (int i = 0; i < c; ++i)
    buf[i] = randgauss() * mul;
  for (int i = c; i < bufn; ++i)
    buf[i] = buf[i % c];

  float *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static float *_kpadone(int bufn) {
  float *buf = new float[bufn];
  for (int i = 0; i < bufn; ++i)
    buf[i] = 1.0;

  float *kbuf;
  kmake(&kbuf, bufn);
  enk(buf, bufn, kbuf);
  delete[] buf;

  return kbuf;
}

static float *pipe_synth(
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf, float rmul, float rounds, int bn
) {
  if (basen == 0)
    return ((float *)kbuf);
  assert(basen > 0);

  float *in = (float *)kbuf;

  Cortex::Segment hdr;
  assert(sizeof(hdr) <= basen);
  memcpy(&hdr, base, sizeof(hdr));
  base += sizeof(hdr);
  kbase += sizeof(hdr);
  basen -= sizeof(hdr);

  uint32_t type;
  int len, ic, oc;
  int viw, vih, vow, voh;
  float mul;
  int rad, freeze, relu;
  int dim;
  _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

  if (viw)
    assert(viw == iw);
  if (vih)
    assert(vih == ih);

  int ow, oh;
  kbuf += (size_t)bn * iw * ih * ic * sizeof(float);
  float *out = (float *)kbuf;

  // batch: activations are laid out [bn][ih][iw][ic]; each kernel runs per sample
  float *in0 = in, *out0 = out;
  for (int __b = 0; __b < bn; ++__b) {
  in = in0 + (size_t)__b * iw * ih * ic;
  out = (__b == 0) ? out0 : out0 + (size_t)__b * ow * oh * oc;

  switch (type) {
  case TYPE_NORM:
    {
      int wmvn = size_norm(ic, iw, ih, oc);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      synth_norm(in, iw, ih, out, ic, oc, wmv);
      break;
    }
  case TYPE_BIAS:
    {
      int wmvn = size_bias(ic, iw, ih, oc);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

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
      float *wmv = (float *)kbase;

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
      float *wmv = (float *)kbase;

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
      float *wmv = (float *)kbase;

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
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      if (__b == 0) synth_conv(in0, iw, ih, out0, d, ic, oc, relu, freeze, dim, wmv, bn);
      break;
    }
  case TYPE_RCN0:
  case TYPE_CON0:
    {
      int d = 0;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      if (__b == 0) synth_conv(in0, iw, ih, out0, d, ic, oc, type == TYPE_RCN0, 0, dim, wmv, bn, (relu & 2) ? mul : 1.0f);
      break;
    }
  case TYPE_RCN1:
  case TYPE_CON1:
    {
      int d = 1;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      if (__b == 0) synth_conv(in0, iw, ih, out0, d, ic, oc, type == TYPE_RCN1, 0, dim, wmv, bn, (relu & 2) ? mul : 1.0f);
      break;
    }
  case TYPE_RCN2:
  case TYPE_CON2:
    {
      int d = 2;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      if (__b == 0) synth_conv(in0, iw, ih, out0, d, ic, oc, type == TYPE_RCN2, 0, dim, wmv, bn, (relu & 2) ? mul : 1.0f);
      break;
    }
  case TYPE_RCN3:
  case TYPE_CON3:
    {
      int d = 3;
      int wmvn = size_conv(d, ic, oc, dim);
      assert(wmvn == len);
      float *wmv = (float *)kbase;

      ow = iw;
      oh = ih;

      if (__b == 0) synth_conv(in0, iw, ih, out0, d, ic, oc, type == TYPE_RCN3, 0, dim, wmv, bn, (relu & 2) ? mul : 1.0f);
      break;
    }
  case TYPE_SCAL:
    {
      int s = _scale_factor(ic, oc);

      ow = (iw << s);
      oh = (ih << s);

      if (s > 0) {
        assert((oc << (s + s)) == ic);
        if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
      } else if (s < 0) {
        assert((ic << -(s + s)) == oc);
        if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, -s, ic, oc, dim);
      } else {
        assert(ic == oc);
        if (__b == 0) synth_pad(in0, iw, ih * bn, out0, ic, oc, NULL);
      }
      break;
    }
  case TYPE_MBSD:
    {
      assert(len == 0);
      assert(ic + 1 == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_mbsd(in0, iw, ih, out0, ic, bn);
      break;
    }
  case TYPE_STYL:
    {
      assert(ic == oc);
      ow = iw;
      oh = ih;
      assert(g_kstyle);
      if (__b == 0) synth_styl(in0, iw, ih, out0, ic, bn, (const float *)kbase, g_kstyle, rad, (relu & 2) ? mul : 1.0f);
      break;
    }
  case TYPE_MCON:
    {
      ow = iw;
      oh = ih;
      assert(g_kstyle);
      if (__b == 0) synth_mcon(in0, iw, ih, out0, ic, oc, bn, (const float *)kbase, g_kstyle, rad,
        (relu & 2) ? mul : 1.0f, (relu & 2) ? 1.0f / sqrtf((float)rad) : 1.0f, (*g_stash)[g_li++]);
      break;
    }
  case TYPE_UPBL:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw * 2;
      oh = ih * 2;
      if (__b == 0) synth_upbl(in0, iw, ih, out0, ic, bn);
      break;
    }
  case TYPE_DNAV:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw / 2;
      oh = ih / 2;
      if (__b == 0) synth_dnav(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_KEEP:
    {
      ow = iw; oh = ih;
      size_t n = (size_t)bn * iw * ih * ic;
      if (__b == 0) { kcopy(in0, n, out0); g_slot_out[rad] = out0; g_slot_n[rad] = n; g_slot_w[rad] = iw; g_slot_h[rad] = ih; g_slot_c[rad] = ic; }
      break;
    }
  case TYPE_RCAL:
    {
      ow = g_slot_w[rad]; oh = g_slot_h[rad];
      assert(g_slot_c[rad] == oc);
      size_t n = (size_t)bn * ow * oh * oc;
      if (__b == 0) { assert(g_slot_out[rad] && g_slot_n[rad] == n); kcopy(g_slot_out[rad], n, out0); }
      break;
    }
  case TYPE_RADD:
    {
      ow = iw; oh = ih;
      size_t n = (size_t)bn * iw * ih * ic;
      if (__b == 0) {
        if (g_slot_out[rad]) { assert(g_slot_n[rad] == n); kaddvec(in0, g_slot_out[rad], n, out0); }
        else kcopy(in0, n, out0);
        if (mul != 1.0f) kmul(out0, mul, n, out0);
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


      if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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


      if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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


      if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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


      if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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


      if (__b == 0) synth_upscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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

      if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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

      if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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

      if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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

      if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
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

      if (__b == 0) synth_downscale(in0, iw, ih * bn, out0, s, ic, oc, dim);
      break;
    }
  case TYPE_CENT:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_cent(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_SIGM:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_sigm(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_CLMP:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_clamp(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_RELU:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_relu(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_LRLU:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_lrlu(in0, iw, ih * bn, out0, ic, mul);
      break;
    }
  case TYPE_PNRM:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_pnrm(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_NERF:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_nerf(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_INRF:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_inrf(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_ABSV:
    {
      assert(len == 0);
      assert(ic == oc);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_abs(in0, iw, ih * bn, out0, ic);
      break;
    }
  case TYPE_DRND:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      float k = 65536.0;
      float z = mul * k / ((float)rounds + k);

      float *knz = _klnoise((oc - ic) * ow * oh, z);
      synth_pad(in, iw, ih, out, ic, oc, knz);
      kfree(knz);
      break;
    }
  case TYPE_LRND:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      if (__b == 0) {
        float *knz = _klnoise((oc - ic) * ow * oh * bn, mul * rmul);
        synth_pad(in0, iw, ih * bn, out0, ic, oc, knz);
        kfree(knz);
      }
      break;
    }
  case TYPE_GRND:
    {
      assert(len == 0);
      assert(ic <= oc);
      ow = iw;
      oh = ih;

      float *knz = _kgnoise((oc - ic) * ow * oh, oc - ic, mul * rmul);
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

      float *knz = _kpadone((oc - ic) * ow * oh);
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
  case TYPE_TLAB:
    {
      assert(len == 0);
      assert(ic == 3);
      assert(oc == 3);
      ow = iw;
      oh = ih;
      synth_tlab(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_FLAB:
    {
      assert(len == 0);
      assert(ic == 3);
      assert(oc == 3);
      ow = iw;
      oh = ih;
      synth_flab(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_POPP:
    {
      assert(len == 0);
      assert(ic > 0);
      assert(oc > 0);
      ow = iw;
      oh = ih;
      synth_popp(in, iw, ih, out, ic, oc, NULL);
      break;
    }
  case TYPE_IDEN:
    {
      assert(len == 0);
      assert(ic > 0);
      assert(oc > 0);
      ow = iw;
      oh = ih;
      if (__b == 0) synth_iden(in0, iw, ih * bn, out0, ic, oc, NULL);
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
      if (__b == 0) synth_mean(in0, iw, ih * bn, out0, ic, oc, NULL);
      break;
    }
  case TYPE_DIFF:
    {
      assert(len == 0);
      ow = iw;
      oh = ih;
      synth_diff(in, iw, ih, out, ic, oc, NULL);
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
  }
  out = out0;

  if (vow)
    assert(vow == ow);
  if (vih)
    assert(voh == oh);

  assert(len * sizeof(float) <= basen);
  base += len * sizeof(float);
  kbase += len * sizeof(float);
  basen -= len * sizeof(float);

  return pipe_synth(
    base, basen, kbase,
    ow, oh, kbuf, rmul, rounds, bn
  );
}




void pipe_learn(
  uint8_t *base, size_t basen, uint8_t *kbase, int iw, int ih, uint8_t *kbuf,
  float nu, float b1, float b2, float eps, float clip, uint64_t rounds, float *kdw, int bn
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
  float mul;
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
  case TYPE_MBSD:
    assert(len == 0);
    assert(ic + 1 == oc);
    ow = iw;
    oh = ih;
    break;
  case TYPE_STYL:
    assert(ic == oc);
    assert(len == size_styl(ic, rad));
    ow = iw;
    oh = ih;
    break;
  case TYPE_MCON:
    assert(len == size_mcon(ic, oc, rad));
    ow = iw;
    oh = ih;
    break;
  case TYPE_UPBL:
    assert(len == 0);
    ow = iw * 2;
    oh = ih * 2;
    break;
  case TYPE_DNAV:
    assert(len == 0);
    ow = iw / 2;
    oh = ih / 2;
    break;
  case TYPE_KEEP:
    assert(len == 0);
    g_slot_w[rad] = iw; g_slot_h[rad] = ih; g_slot_c[rad] = ic;
    ow = iw;
    oh = ih;
    break;
  case TYPE_RADD:
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_RCAL:
    assert(len == 0);
    ow = g_slot_w[rad];
    oh = g_slot_h[rad];
    break;
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
  case TYPE_CENT:
  case TYPE_SIGM:
  case TYPE_RELU:
  case TYPE_LRLU:
  case TYPE_PNRM:
  case TYPE_CLMP:
  case TYPE_ABSV:
  case TYPE_NERF:
  case TYPE_INRF:
    assert(ic == oc);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_DRND:
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
  case TYPE_TLAB:
  case TYPE_FLAB:
    assert(ic == 3);
    assert(oc == 3);
    assert(len == 0);
    ow = iw;
    oh = ih;
    break;
  case TYPE_POPP:
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
  case TYPE_DIFF:
    assert(oc * 2 == ic);
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

  float *in = (float *)kbuf;
  kbuf += (size_t)bn * iw * ih * ic * sizeof(float);
  float *fout = (float *)kbuf;

  float *wmv = (float *)kbase;
  assert(len * sizeof(float) <= basen);
  base += len * sizeof(float);
  kbase += len * sizeof(float);
  basen -= len * sizeof(float);

  pipe_learn(
    base, basen, kbase,
    ow, oh, kbuf,
    nu, b1, b2, eps, clip, rounds,
    kdw ? (kdw + len / 3) : kdw,
    bn
  );

  // batch: gradients accumulate across samples (nu=0), the weight update
  // (nu>0) is applied once on the last sample
  float *in0 = in, *fout0 = fout;
  float nu0 = nu;
  for (int __b = 0; __b < bn; ++__b) {
  in = in0 + (size_t)__b * iw * ih * ic;
  fout = fout0 + (size_t)__b * ow * oh * oc;
  nu = (__b + 1 < bn) ? 0.0f : nu0;

  switch (type) {
  case TYPE_NORM:
    learn_norm(in, iw, ih, fout, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_BIAS:
    learn_bias(in, iw, ih, fout, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_LOC0:
    learn_local(in, iw, ih, fout, 0, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_LOC1:
    learn_local(in, iw, ih, fout, 1, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_LOC2:
    learn_local(in, iw, ih, fout, 2, ic, oc, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds);
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
    if (__b + 1 == bn) learn_conv(in0, iw, ih, fout0, rad, ic, oc, relu, freeze, dim, wmv, kdw, nu * mul, b1, b2, eps, clip, (float)rounds, bn);
    break;

  case TYPE_CON0:
  case TYPE_RCN0:
    if (__b + 1 == bn) learn_conv(in0, iw, ih, fout0, 0, ic, oc, type == TYPE_RCN0, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds, bn, (relu & 2) ? mul : 1.0f);
    break;
  case TYPE_CON1:
  case TYPE_RCN1:
    if (__b + 1 == bn) learn_conv(in0, iw, ih, fout0, 1, ic, oc, type == TYPE_RCN1, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds, bn, (relu & 2) ? mul : 1.0f);
    break;
  case TYPE_CON2:
  case TYPE_RCN2:
    if (__b + 1 == bn) learn_conv(in0, iw, ih, fout0, 2, ic, oc, type == TYPE_RCN2, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds, bn, (relu & 2) ? mul : 1.0f);
    break;
  case TYPE_CON3:
  case TYPE_RCN3:
    if (__b + 1 == bn) learn_conv(in0, iw, ih, fout0, 3, ic, oc, type == TYPE_RCN3, 0, dim, wmv, kdw, nu, b1, b2, eps, clip, (float)rounds, bn, (relu & 2) ? mul : 1.0f);
    break;
  case TYPE_SCAL:
    {
      int s = _scale_factor(ic, oc);
      if (s > 0)
        if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, s, ic, oc, dim);
      else if (s < 0) 
        if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, -s, ic, oc, dim);
      else
        if (__b + 1 == bn) learn_pad(in0, iw, ih * bn, fout0, ic, oc);
    }
    break;
  case TYPE_MBSD:
    if (__b + 1 == bn) learn_mbsd(in0, iw, ih, fout0, ic, bn);
    break;
  case TYPE_STYL:
    assert(g_kstyle);
    if (__b + 1 == bn) learn_styl(in0, iw, ih, fout0, ic, bn, wmv, kdw, g_kstyle, g_kstylegrad, rad, (relu & 2) ? mul : 1.0f, nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_MCON:
    assert(g_kstyle);
    // layers are visited tail-first here; the stash index counts down
    if (__b + 1 == bn) learn_mcon(in0, iw, ih, fout0, ic, oc, bn, wmv, kdw, g_kstyle, g_kstylegrad, rad,
      (relu & 2) ? mul : 1.0f, (relu & 2) ? 1.0f / sqrtf((float)rad) : 1.0f, (*g_stash)[--g_li], nu, b1, b2, eps, clip, (float)rounds);
    break;
  case TYPE_UPBL:
    if (__b + 1 == bn) learn_upbl(in0, iw, ih, fout0, ic, bn);
    break;
  case TYPE_DNAV:
    if (__b + 1 == bn) learn_dnav(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_KEEP:
    if (__b + 1 == bn) {
      size_t n = (size_t)bn * iw * ih * ic;
      kcopy(fout0, n, in0);
      if (g_slot_grad[rad]) { kaxpy(g_slot_grad[rad], g_slot_gmul[rad], n, in0); g_slot_grad[rad] = NULL; }
    }
    break;
  case TYPE_RCAL:
    if (__b + 1 == bn) {
      kzero(in0, (size_t)bn * iw * ih * ic);
      g_slot_grad[rad] = fout0; g_slot_gmul[rad] = 1.0f;
    }
    break;
  case TYPE_RADD:
    if (__b + 1 == bn) {
      size_t n = (size_t)bn * iw * ih * ic;
      kcopy(fout0, n, in0);
      if (mul != 1.0f) kmul(in0, mul, n, in0);
      g_slot_grad[rad] = fout0; g_slot_gmul[rad] = mul;
    }
    break;
  case TYPE_UPS1:
    if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, 1, ic, oc, dim);
    break;
  case TYPE_UPS2:
    if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, 2, ic, oc, dim);
    break;
  case TYPE_UPS3:
    if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, 3, ic, oc, dim);
    break;
  case TYPE_UPS4:
    if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, 4, ic, oc, dim);
    break;
  case TYPE_UPS5:
    if (__b + 1 == bn) learn_upscale(in0, iw, ih * bn, fout0, 5, ic, oc, dim);
    break;
  case TYPE_DNS1:
    if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, 1, ic, oc, dim);
    break;
  case TYPE_DNS2:
    if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, 2, ic, oc, dim);
    break;
  case TYPE_DNS3:
    if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, 3, ic, oc, dim);
    break;
  case TYPE_DNS4:
    if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, 4, ic, oc, dim);
    break;
  case TYPE_DNS5:
    if (__b + 1 == bn) learn_downscale(in0, iw, ih * bn, fout0, 5, ic, oc, dim);
    break;
  case TYPE_CENT:
    if (__b + 1 == bn) learn_cent(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_SIGM:
    if (__b + 1 == bn) learn_sigm(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_CLMP:
    if (__b + 1 == bn) learn_clamp(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_RELU:
    if (__b + 1 == bn) learn_relu(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_LRLU:
    if (__b + 1 == bn) learn_lrlu(in0, iw, ih * bn, fout0, ic, mul);
    break;
  case TYPE_PNRM:
    if (__b + 1 == bn) learn_pnrm(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_NERF:
    if (__b + 1 == bn) learn_nerf(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_INRF:
    if (__b + 1 == bn) learn_inrf(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_ABSV:
    if (__b + 1 == bn) learn_abs(in0, iw, ih * bn, fout0, ic);
    break;
  case TYPE_DRND:
  case TYPE_LRND:
  case TYPE_GRND:
  case TYPE_PAD1:
    if (__b + 1 == bn) learn_pad(in0, iw, ih * bn, fout0, ic, oc);
    break;
  case TYPE_ADDG:
    learn_addgeo(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_ZERO:
    learn_zero(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_TLAB:
  case TYPE_FLAB:
    assert(0);
    break;
  case TYPE_POPP:
    learn_popp(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_IDEN:
    if (__b + 1 == bn) learn_iden(in0, iw, ih * bn, fout0, ic, oc);
    break;
  case TYPE_MEDI:
    learn_median(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_BLUR:
    learn_blur(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_DIFF:
    learn_diff(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_SUMM:
    learn_sum(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_SMAX:
    learn_smax(in, iw, ih, fout, ic, oc);
    break;
  case TYPE_MEAN:
    if (__b + 1 == bn) learn_mean(in0, iw, ih * bn, fout0, ic, oc);
    break;
  default:
    assert(0);
  }
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
  kstyle = NULL;
  kstylegrad = NULL;
  sdim = 0;
  kdw = NULL;
  dwn = 0;

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
  if (v != 3)
    error(fmt("%s: ctx version %u, this build needs version 3 (float weights); re-make it from its spec", fn.c_str(), v));

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

    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);

    tic = oc;
    ++i;
  }
  assert(basei == basen);

  is_open = true;
}

void Cortex::_get_head_op(float **vecp, float **matp, int *wp, int *hp) {
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

  const float *wmv = (float *)(base + basei);
  *matp = new float[*wp * *hp];
  for (int y = 0; y < *hp; ++y)
    for (int x = 0; x < *wp; ++x)
      (*matp)[x + y * *wp] = wmv[voc + x + y * *wp];
  *vecp = new float[*hp];
  for (int y = 0; y < *hp; ++y)
    (*vecp)[y] = wmv[y];
}

void Cortex::_put_head_op(const float *vec, const float *mat, int w, int h) {
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

  float *wmv = (float *)(base + basei);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      wmv[voc + x + y * w] = mat[x + y * w];
  for (int y = 0; y < h; ++y)
    wmv[y] = vec[y];
}

void Cortex::_get_tail_op(float **vecp, float **matp, int *wp, int *hp) {
  unsigned int basei = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic, voc;
    _parse_header(hdr, &type, &len, &vic, &voc);

    assert(basei + len * sizeof(float) <= basen);
    if (basei + len * sizeof(float) == basen) {
      int tlen = len / 3;
      *hp = voc;
      assert(tlen > voc);
      tlen -= voc;
      assert(tlen % voc == 0);
      *wp = tlen / voc;

      const float *wmv = (float *)(base + basei);
      *matp = new float[*wp * *hp];
      for (int y = 0; y < *hp; ++y)
        for (int x = 0; x < *wp; ++x)
          (*matp)[x + y * *wp] = wmv[voc + x + y * *wp];
      *vecp = new float[*hp];
      for (int y = 0; y < *hp; ++y)
        (*vecp)[y] = wmv[y];
      return;
    }

    basei += len * sizeof(float);
  }
  assert(0);
}

void Cortex::_put_tail_op(const float *vec, const float *mat, int w, int h) {
  unsigned int basei = 0;

  while (basei < basen) {
    Cortex::Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, vic, voc;
    _parse_header(hdr, &type, &len, &vic, &voc);

    assert(basei + len * sizeof(float) <= basen);
    if (basei + len * sizeof(float) == basen) {
      int tlen = len / 3;
      assert(h == voc);
      assert(tlen > voc);
      tlen -= voc;
      assert(tlen % voc == 0);
      assert(w * voc == tlen);

      float *wmv = (float *)(base + basei);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          wmv[voc + x + y * w] = mat[x + y * w];
      for (int y = 0; y < h; ++y)
        wmv[y] = vec[y];

      return;
    }

    basei += len * sizeof(float);
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
  head.version = htonl(3);
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
    float mul;
    int dim;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    if (hdr.freeze) ++got;
    if (!last && got) {
      hdr.freeze = 0;
      memcpy(base + basei - sizeof(hdr), &hdr, sizeof(hdr));
      return;
    }

    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);

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
    float mul;
    int dim;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    if (hdr.freeze) --got;
    if (got == 0) {
      hdr.freeze = 0;
      memcpy(base + basei - sizeof(hdr), &hdr, sizeof(hdr));
      return;
    }


    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);

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
    float mul;
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

    if (type == TYPE_LRND || type == TYPE_GRND || type == TYPE_DRND) {
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

    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);

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

    assert(basei + len * sizeof(float) <= basen);
    assert(i <= 4096);

    if (i >= a && i <= b) {
      int ret = fwrite(&hdr, 1, sizeof(hdr), outfp);
      assert(ret == sizeof(hdr));
      ret = fwrite(base + basei, 1, len * sizeof(float), outfp);
      assert(ret == len * sizeof(float));
    }

    basei += len * sizeof(float);
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

  if (kstylegrad) {
    kfree(kstylegrad);
    kstylegrad = NULL;
  }
  sdim = 0;
  for (auto k : kstash) kfree(k);
  kstash.clear();
  if (kdw) {
    kfree(kdw);
    kdw = NULL;
  }
  dwn = 0;

  iw = ih = ic = 0;
  ow = oh = oc = 0;
  iwhc = owhc = 0;
  rounds = 0;
  is_prep = false;
}

bool Cortex::prepare(int _iw, int _ih, int _bn) {
  assert(_bn > 0);
  if (is_prep && _bn != bn)
    unprepare();
  bn = _bn;
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

  sdim = 0;
  g_stash_sizes.clear();
  kbufn = pipe_prep(base, basen, _iw, _ih, &vic, &ow, &oh, &voc, &dwn, bn, &sdim);
  for (auto n : g_stash_sizes) { float *k; kmake(&k, n); kstash.push_back(k); }
  if (kbufn == 0)
    return false;
  assert(kbufn > 0);
  kmake(&kbuf, kbufn);
  kinp = (float *)kbuf;
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

  if (sdim > 0) {
    kmake(&kstylegrad, (unsigned int)sdim * bn);
    kfill(kstylegrad, (unsigned int)sdim * bn, 0);
  }

  is_prep = true;
  return true;
}

float *Cortex::synth() {
  g_noisecall = 0;
  assert(is_open);
  assert(is_prep);
  assert(sdim == 0 || kstyle);
  g_kstyle = kstyle;
  g_kstylegrad = kstylegrad;
  g_stash = &kstash;
  g_li = 0;
  memset(g_slot_out, 0, sizeof(g_slot_out));
  kout = pipe_synth(base, basen, kbase, iw, ih, kbuf, rmul, rounds, bn);
  return kout;
}

float *Cortex::synth(const float *_kinp) {
  assert(is_open);
  assert(is_prep);
  kcopy(_kinp, iwhc * bn, kinp);
  return synth();
}

void Cortex::stats() {
  float nrms = sqrt(ksumsq(kout, owhc * bn) / (float)(owhc * bn));

  float z = pow(1.0 - decay, (float)rounds);
  rms *= (1.0 - z);
  rms *= (1.0 - decay);
  rms += decay * nrms;
  rms *= 1.0 / (1.0 - z * (1.0 - decay));

  float nmax = kmaxabs(kout, owhc * bn);
  max *= (1.0 - z);
  max *= (1.0 - decay);
  max += decay * nmax;
  max *= 1.0 / (1.0 - z * (1.0 - decay));

  ++rounds;
  ++new_rounds;
}

void Cortex::target(const float *ktgt) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  assert(owhc > 0);
  ksubvec(ktgt, kout, owhc * bn, kout);
}

void Cortex::setloss(const float *ktgt) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  assert(owhc > 0);
  kcopy(ktgt, owhc * bn, kout);
}

float *Cortex::propback() {
  assert(is_open);
  assert(is_prep);
  g_kstyle = kstyle;
  g_kstylegrad = kstylegrad;
  g_stash = &kstash;
  g_li = (int)kstash.size();
  if (kstylegrad)
    kfill(kstylegrad, (unsigned int)sdim * bn, 0);
  memset(g_slot_grad, 0, sizeof(g_slot_grad));
  pipe_learn(base, basen, kbase, iw, ih, kbuf, 0, b1, b2, eps, clip, rounds, NULL, bn);
  return kinp;
}

float *Cortex::accumulate() {
  assert(is_open);
  assert(is_prep);
  g_kstyle = kstyle;
  g_kstylegrad = kstylegrad;
  g_stash = &kstash;
  g_li = (int)kstash.size();
  if (kstylegrad)
    kfill(kstylegrad, (unsigned int)sdim * bn, 0);
  stats();
  memset(g_slot_grad, 0, sizeof(g_slot_grad));
  pipe_learn(base, basen, kbase, iw, ih, kbuf, 0, b1, b2, eps, clip, rounds, kdw, bn);
  return kinp;
}

float *Cortex::learn(float mul) {
  assert(is_open);
  assert(is_prep);
  g_kstyle = kstyle;
  g_kstylegrad = kstylegrad;
  g_stash = &kstash;
  g_li = (int)kstash.size();
  if (kstylegrad)
    kfill(kstylegrad, (unsigned int)sdim * bn, 0);
  stats();
  memset(g_slot_grad, 0, sizeof(g_slot_grad));
  pipe_learn(base, basen, kbase, iw, ih, kbuf, nu * mul, b1, b2, eps, clip, rounds, kdw, bn);
  return kinp;
}

float *Cortex::learn(const float *_kout, float mul) {
  assert(is_open);
  assert(is_prep);
  assert(kout);
  kcopy(_kout, owhc * bn, kout);
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

// exponential moving average of another (identically structured) cortex's weights
// into this one: w = beta * w + (1 - beta) * src.w, on the device; call save() to persist
void Cortex::ema(const Cortex &src, float beta) {
  assert(is_prep && src.is_prep);
  assert(basen == src.basen);
  size_t off = 0;
  while (off < basen) {
    Segment hdr;
    memcpy(&hdr, base + off, sizeof(hdr));
    off += sizeof(hdr);
    uint32_t type; int len;
    _parse_header(hdr, &type, &len);
    if (len > 0) {
      int wn = len / 3;
      kema((float *)(kbase + off), (const float *)(src.kbase + off), wn, beta);
      off += (size_t)len * sizeof(float);
    }
  }
  assert(off == basen);
}

void Cortex::scram(float dev, bool fanin) {
  info(fmt("scrambling %s with dev=%g fanin=%d", fn.c_str(), dev, (int)fanin));

  assert(is_open);
  assert(!is_prep);

  unsigned int basei = 0;

  while (basei < basen) {
    Segment hdr;
    assert(basei + sizeof(hdr) <= basen);
    memcpy(&hdr, base + basei, sizeof(hdr));
    basei += sizeof(hdr);

    uint32_t type;
    int len, ic, oc, viw, vih, vow, voh, rad, freeze, relu, dim;
    float mul;
    _parse_header(hdr, &type, &len, &ic, &oc, &viw, &vih, &vow, &voh, &mul, &rad, &freeze, &relu, &dim);

    if (len == 0)
      continue;
    assert(len > 0);

    uint8_t *bwmv = base + basei;
    float *wmv = (float *)bwmv;
    assert(len % 3 == 0);

    // number of leading bias entries and fan-in for He-style init
    int nbias = 0, nin = 0;
    switch (type) {
    case TYPE_CONV: { int f = 2 * rad + 1; nbias = oc; nin = (dim == 1) ? ic * f : ic * f * f; break; }
    case TYPE_CON0: case TYPE_RCN0: nbias = oc; nin = ic; break;
    case TYPE_CON1: case TYPE_RCN1: nbias = oc; nin = (dim == 1) ? ic * 3 : ic * 9; break;
    case TYPE_CON2: case TYPE_RCN2: nbias = oc; nin = (dim == 1) ? ic * 5 : ic * 25; break;
    case TYPE_CON3: case TYPE_RCN3: nbias = oc; nin = (dim == 1) ? ic * 7 : ic * 49; break;
    case TYPE_LOC0: nbias = vow * voh * oc; nin = ic; break;
    case TYPE_LOC1: nbias = vow * voh * oc; nin = ic * 9; break;
    case TYPE_LOC2: nbias = vow * voh * oc; nin = ic * 25; break;
    case TYPE_NORM: case TYPE_BIAS: nbias = len / 3; nin = 0; break;
    case TYPE_STYL: nbias = 2 * ic; nin = rad; break;
    default: nbias = 0; nin = 0; break;
    }

    if (type == TYPE_MCON) {
      int wn = len / 3;
      long offab = oc + 9L * oc * ic, offA = offab + ic;
      bool eq = (relu & 2);
      float wdev = eq ? dev : dev * sqrt(2.0 / (9.0 * ic));
      float adev = eq ? dev : dev / sqrt((float)rad);
      for (int i = 0; i < wn; ++i) {
        float v;
        if (i < oc) v = 0.0;
        else if (i < offab) v = randgauss() * wdev;
        else if (i < offA) v = 1.0;
        else v = randgauss() * adev;
        wmv[i] = v; wmv[wn + i] = 0.0; wmv[2 * wn + i] = 0.0;
      }
      assert(basei + len * sizeof(float) <= basen);
      basei += len * sizeof(float);
      continue;
    }

    float wdev = dev;
    if (fanin)
      wdev = (nin > 0) ? dev * sqrt(2.0 / (float)nin) : 0.0;
    // equalized learning rate (relu bit 2 on con layers): weights stay N(0, dev), the
    // fan-in scale is applied at runtime through the segment's mul field
    if ((relu & 2) && nin > 0)
      wdev = dev;
    if (type == TYPE_STYL && !(relu & 2))
      wdev = dev / sqrt((float)nin);

    // segment layout: [w_0..w_wn-1][m_0..m_wn-1][v_0..v_wn-1]
    int wn = len / 3;
    for (int i = 0; i < wn; ++i) {
      bool isbias = i < nbias;
      wmv[i] = (fanin && isbias) ? 0.0 : randgauss() * wdev;
      wmv[wn + i] = 0.0;
      wmv[2 * wn + i] = 0.0;
    }

    assert(basei + len * sizeof(float) <= basen);
    basei += len * sizeof(float);
  }

  assert(basei == basen);
}

void Cortex::push(const std::string &stype, int nic, int noc, int niw, int nih, int now, int noh, float mul, int rad, int freeze, int relu, int dim) {
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
    else if (stype == "mbsd")	{ noc = (nic + 1); }
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
  case TYPE_STYL:
    assert(rad > 0);
    assert(oc == nic);
    len = size_styl(nic, rad);
    break;
  case TYPE_MCON:
    assert(rad > 0);
    len = size_mcon(nic, oc, rad);
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
  case TYPE_UPBL:
  case TYPE_DNAV:
  case TYPE_KEEP:
  case TYPE_RCAL:
  case TYPE_RADD:
  case TYPE_MBSD:
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
  case TYPE_CENT:
  case TYPE_RELU:
  case TYPE_LRLU:
  case TYPE_PNRM:
  case TYPE_CLMP:
  case TYPE_NERF:
  case TYPE_INRF:
  case TYPE_ABSV:
  case TYPE_GRND:
  case TYPE_LRND:
  case TYPE_DRND:
  case TYPE_PAD1:
  case TYPE_ADDG:
  case TYPE_IDEN:
  case TYPE_POPP:
  case TYPE_TLAB:
  case TYPE_FLAB:
  case TYPE_ZERO:
  case TYPE_MEAN:
  case TYPE_SMAX:
  case TYPE_SUMM:
  case TYPE_DIFF:
  case TYPE_BLUR:
  case TYPE_MEDI:
    len = 0;
    break;
  default:
    assert(0);
  }

  size_t new_basen = basen + sizeof(Segment) + len * sizeof(float);

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
