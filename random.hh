#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__ 1

#include <stdint.h>

#include <string>
#include <vector>

namespace makemore {

extern double randgauss();
extern uint32_t randuint();
extern void seedrand(uint64_t);
extern void seedrand();

}

#endif
