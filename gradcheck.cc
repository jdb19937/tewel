// Finite-difference gradient check for tewel Cortex networks.
// Usage: gradcheck net.ctx iw ih [nweights] [ninputs] [batch]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arpa/inet.h>
#include <vector>
#include "cortex.hh"
#include "colonel.hh"
using namespace makemore;

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s net.ctx iw ih [nw] [nx]\n", argv[0]); return 1; }
  const char *fn = argv[1]; int iw = atoi(argv[2]), ih = atoi(argv[3]);
  int nw = argc > 4 ? atoi(argv[4]) : 24, nx = argc > 5 ? atoi(argv[5]) : 8, bn = argc > 6 ? atoi(argv[6]) : 1;
  setkdev(kndevs() > 1 ? 1 : 0); setkbs(256);
  Cortex *c = new Cortex(fn, O_RDWR);
  c->prepare(iw, ih, bn);
  if (c->sdim > 0) c->kstyle = c->kinp;   // standalone: the input doubles as the style
  int n = c->iwhc * bn, m = c->owhc * bn;   // whole batch; loss sums over it
  srand48(7);
  std::vector<float> x(n), t(m), y(m);
  for (int i = 0; i < n; ++i) x[i] = drand48();
  for (int i = 0; i < m; ++i) t[i] = drand48();
  float *kx, *kt; kmake(&kx, n); kmake(&kt, m);
  enk(x.data(), n, kx); enk(t.data(), m, kt);
  auto loss = [&]() {
    c->synth(kx); dek(c->kout, m, y.data());
    double L = 0; for (int i = 0; i < m; ++i) L += 0.5 * ((double)t[i] - (double)y[i]) * ((double)t[i] - (double)y[i]);
    return L;
  };
  double L0 = loss();
  // analytic gradient (ascent direction: target - output convention)
  c->synth(kx); c->target(kt); c->accumulate();
  std::vector<float> gw(c->dwn), gx(n);
  dek(c->kdw, c->dwn, gw.data()); dek(c->kinp, n, gx.data());
  if (c->sdim > 0) { std::vector<float> gs(n); dek(c->kstylegrad, n, gs.data()); for (int i = 0; i < n; ++i) gx[i] += gs[i]; }
  kfill(c->kdw, c->dwn, 0);
  // map global weight index -> host pointer and segment type
  std::vector<float*> wptr; std::vector<std::string> wtype;
  size_t off = 0;
  while (off < c->basen) {
    Cortex::Segment hdr; memcpy(&hdr, c->base + off, sizeof(hdr)); off += sizeof(hdr);
    int len = ntohl(hdr.len); uint32_t ty = ntohl(hdr.type);
    char tn[5] = {char(ty>>24), char(ty>>16), char(ty>>8), char(ty), 0};
    float *wmv = (float *)(c->base + off);
    for (int i = 0; i < len / 3; ++i) { wptr.push_back(&wmv[i]); wtype.push_back(tn); }
    off += (size_t)len * sizeof(float);
  }
  if (wptr.size() != c->dwn) { fprintf(stderr, "weight count mismatch %zu vs %lu\n", wptr.size(), c->dwn); return 1; }
  printf("net=%s in=%dx%dx%d out=%dx%dx%d batch=%d weights=%lu L0=%g\n", fn, iw, ih, c->ic, c->ow, c->oh, c->oc, bn, c->dwn, L0);
  double h = 5e-4; int bad = 0;
  printf("--- weights: idx type analytic(-dL/dw) finite-diff relerr\n");
  for (int s = 0; s < nw; ++s) {
    size_t k = (size_t)(drand48() * wptr.size()); if (s == 0) k = 0;
    float *p = wptr[k], w = *p;
    *p = w + h; c->load(); double Lp = loss();
    *p = w - h; c->load(); double Lm = loss();
    *p = w; c->load();
    double fd = -(Lp - Lm) / (2 * h), an = gw[k];
    double rel = fabs(fd - an) / (fabs(fd) + fabs(an) + 1e-12);
    if (rel > 2e-2) ++bad;
    printf("%8zu %s %14.6g %14.6g %8.2e%s\n", k, wtype[k].c_str(), an, fd, rel, rel > 2e-2 ? "  <-- MISMATCH" : "");
  }
  printf("--- inputs: idx analytic finite-diff relerr\n");
  for (int s = 0; s < nx; ++s) {
    int j = (int)(drand48() * n); float v = x[j];
    x[j] = v + h; enk(x.data(), n, kx); double Lp = loss();
    x[j] = v - h; enk(x.data(), n, kx); double Lm = loss();
    x[j] = v; enk(x.data(), n, kx);
    double fd = -(Lp - Lm) / (2 * h), an = gx[j];
    double rel = fabs(fd - an) / (fabs(fd) + fabs(an) + 1e-12);
    if (rel > 2e-2) ++bad;
    printf("%8d      %14.6g %14.6g %8.2e%s\n", j, an, fd, rel, rel > 2e-2 ? "  <-- MISMATCH" : "");
  }
  printf("RESULT: %d mismatches\n", bad);
  return bad ? 2 : 0;
}
