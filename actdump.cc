// Dump per-layer activation statistics of a ctx on random gaussian inputs.
// Usage: actdump net.ctx iw ih [bn]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <vector>
#include "cortex.hh"
#include "colonel.hh"
#include "random.hh"
using namespace makemore;

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s net.ctx iw ih [bn]\n", argv[0]); return 1; }
  int iw = atoi(argv[2]), ih = atoi(argv[3]), bn = argc > 4 ? atoi(argv[4]) : 4;
  setkdev(kndevs() > 1 ? 1 : 0); setkbs(256);
  Cortex *c = new Cortex(argv[1], O_RDONLY);
  c->prepare(iw, ih, bn);
  if (c->sdim > 0) c->kstyle = c->kinp;
  seedrand(1);
  kzero(c->kinp, c->iwhc * bn);
  kaddnoise(c->kinp, c->iwhc * bn, 1.0);
  c->synth();
  // walk segments to get per-layer output sizes; activations are laid out in kbuf
  std::vector<float> buf(c->kbufn / sizeof(float));
  dek((const float *)c->kbuf, buf.size(), buf.data());
  size_t off = 0, aoff = 0;
  int w = iw, h = ih, ic = c->ic;
  printf("%-6s %6s %12s %12s %12s\n", "layer", "shape", "rms", "xsample-std", "xpixel-std");
  auto stats = [&](const char *name, int ww, int hh, int cc) {
    size_t whc = (size_t)ww * hh * cc;
    double ss = 0; for (size_t i = 0; i < whc * bn; ++i) ss += (double)buf[aoff + i] * buf[aoff + i];
    double rms = sqrt(ss / (whc * bn));
    // std across samples (mean over elements) and across pixels (mean over channels and samples)
    double xs = 0; for (size_t i = 0; i < whc; ++i) { double m = 0, q = 0; for (int b = 0; b < bn; ++b) { double v = buf[aoff + b * whc + i]; m += v; q += v * v; } m /= bn; xs += sqrt(fmax(q / bn - m * m, 0)); } xs /= whc;
    double xp = 0; size_t np = (size_t)ww * hh; for (int b = 0; b < bn; ++b) for (int z = 0; z < cc; ++z) { double m = 0, q = 0; for (size_t p = 0; p < np; ++p) { double v = buf[aoff + b * whc + p * cc + z]; m += v; q += v * v; } m /= np; xp += sqrt(fmax(q / np - m * m, 0)); } xp /= (double)bn * cc;
    printf("%-6s %3dx%-3dx%-4d %12.4g %12.4g %12.4g\n", name, ww, hh, cc, rms, xs, xp);
    aoff += whc * bn;
  };
  stats("input", w, h, ic);
  while (off < c->basen) {
    Cortex::Segment hdr; memcpy(&hdr, c->base + off, sizeof(hdr)); off += sizeof(hdr);
    int len = ntohl(hdr.len); uint32_t ty = ntohl(hdr.type); int oc = ntohl(hdr.oc);
    char tn[5] = {char(ty>>24), char(ty>>16), char(ty>>8), char(ty), 0};
    std::string t = tn;
    int ow = w, oh = h;
    if (t == "ups1") { ow = w * 2; oh = h * 2; } else if (t == "ups2") { ow = w * 4; oh = h * 4; }
    else if (t == "dns1") { ow = w / 2; oh = h / 2; } else if (t == "dns2") { ow = w / 4; oh = h / 4; }
    else if (t == "upbl") { ow = w * 2; oh = h * 2; } else if (t == "dnav") { ow = w / 2; oh = h / 2; }
    off += (size_t)len * sizeof(float);
    stats(tn, ow, oh, oc);
    w = ow; h = oh; ic = oc;
  }
  return 0;
}
