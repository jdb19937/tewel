#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__ 1

#include <stdint.h>

#include <string>
#include <vector>

namespace makemore {

extern float randgauss();
extern float randrange(float, float);
extern uint32_t randuint();
extern void seedrand(uint64_t);
extern void seedrand();

extern void kaddnoise(float *kdat, unsigned int n, float dev);

}

#endif
