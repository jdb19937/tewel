#define __MAKEMORE_RANDOM_CC__ 1

#include <algorithm>
#include <random>

#include "random.hh"
#include "colonel.hh"

namespace makemore {

static std::random_device rd;
static std::mt19937 rg(rd());
static std::normal_distribution<double> rn(0, 1);
static std::uniform_int_distribution<uint32_t> ru(0, 1UL<<31);

double randgauss() {
  return rn(rg);
}

double randrange(double a, double b) {
  std::uniform_real_distribution<double> uniform(a, b);
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

void kaddnoise(double *kdat, unsigned int n, double dev) {
  if (dev == 0.0)
    return;

  double *noise = new double[n];
  for (unsigned int j = 0; j < n; ++j)
    noise[j] = randgauss() * dev;
  double *knoise;
  kmake(&knoise, n);
  enk(noise, n, knoise);
  delete[] noise;

  kaddvec(kdat, knoise, n, kdat);
  kfree(knoise);
}

}
