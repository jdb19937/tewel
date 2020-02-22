#define __MAKEMORE_RANDOM_CC__ 1

#include <algorithm>
#include <random>

#include "random.hh"

namespace makemore {

static std::random_device rd;
static std::mt19937 rg(rd());
static std::normal_distribution<double> rn(0, 1);
static std::uniform_int_distribution<uint32_t> ru(0, 1UL<<31);

double randgauss() {
  return rn(rg);
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

}
