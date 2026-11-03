// Deterministic sample-pick check: seed the RNG, pick n u8 samples with flips, print checksums.
// Usage: picktest file.rgb n
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "kleption.hh"
#include "colonel.hh"
#include "random.hh"
using namespace makemore;
int main(int argc, char **argv) {
  int n = atoi(argv[2]);
  setkdev(1); setkbs(256); seedrand(12345);
  Kleption k(argv[1], 256, 256, 3, (Kleption::Flags)(Kleption::FLAG_REPEAT | Kleption::FLAG_FLIP), Kleption::TRAV_RAND, Kleption::KIND_U8, 256, 256, 3);
  float *kd; kmake(&kd, 256 * 256 * 3);
  std::vector<float> h(256 * 256 * 3);
  for (int i = 0; i < n; ++i) {
    k.pick(kd); dek(kd, 256 * 256 * 3, h.data());
    double s = 0, sx = 0; for (size_t j = 0; j < h.size(); ++j) { s += h[j]; sx += h[j] * (double)(j % 997); }
    printf("%d %.6f %.3f %.6f %.6f %.6f\n", i, s, sx, h[0], h[12345], h[196607]);
  }
  return 0;
}
