// Compare fp32-tile and tensor-core conv paths: forward outputs, weight and input gradients.
// Usage: prectest net.ctx iw ih bn
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "cortex.hh"
#include "colonel.hh"
#include "random.hh"
using namespace makemore;
static double relerr(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
  return sqrt(num / (den + 1e-30));
}
int main(int argc, char **argv) {
  int iw = atoi(argv[2]), ih = atoi(argv[3]), bn = atoi(argv[4]);
  setkdev(1); setkbs(256);
  Cortex *c = new Cortex(argv[1], O_RDWR);
  c->prepare(iw, ih, bn);
  if (c->sdim > 0) c->kstyle = c->kinp;
  int n = c->iwhc * bn, m = c->owhc * bn;
  float *kx, *kt; kmake(&kx, n); kmake(&kt, m);
  kzero(kx, n); kaddnoise(kx, n, 1.0); kzero(kt, m); kaddnoise(kt, m, 1.0);
  std::vector<float> out[2], gw[2], gx[2];
  for (int pass = 0; pass < 2; ++pass) {
    setkprec(pass == 0 ? 32 : 16);
    c->synth(kx); out[pass].resize(m); dek(c->kout, m, out[pass].data());
    c->target(kt); c->accumulate();
    gw[pass].resize(c->dwn); dek(c->kdw, c->dwn, gw[pass].data()); kfill(c->kdw, c->dwn, 0);
    gx[pass].resize(n); dek(c->kinp, n, gx[pass].data());
  }
  printf("%s bn=%d: relerr forward %.2e  weight-grad %.2e  input-grad %.2e\n", argv[1], bn, relerr(out[0], out[1]), relerr(gw[0], gw[1]), relerr(gx[0], gx[1]));
  return 0;
}
