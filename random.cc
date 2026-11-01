#define __MAKEMORE_RANDOM_CC__ 1

#include <algorithm>
#include <random>

#include "random.hh"
#include "colonel.hh"

namespace makemore {

static std::random_device rd;
static std::mt19937 rg(rd());
static std::normal_distribution<float> rn(0, 1);
static std::uniform_int_distribution<uint32_t> ru(0, 1UL<<31);

float randgauss() {
  return rn(rg);
}

float randrange(float a, float b) {
  std::uniform_real_distribution<float> uniform(a, b);
  return uniform(rg);
}

uint32_t randuint() {
  return ru(rg);
}

void seedrand(uint64_t seed) {
  rg.seed(seed);
}

void seedrand() {
  rg.seed(rd());
}

void kaddnoise(float *kdat, unsigned int n, float dev) {
  if (dev == 0.0)
    return;

  float *noise = new float[n];
  for (unsigned int j = 0; j < n; ++j)
    noise[j] = randgauss() * dev;
  float *knoise;
  kmake(&knoise, n);
  enk(noise, n, knoise);
  delete[] noise;

  kaddvec(kdat, knoise, n, kdat);
  kfree(knoise);
}

}
