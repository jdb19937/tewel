#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__ 1

#include <stdint.h>

#include <string>
#include <vector>

namespace makemore {

extern double randgauss();
extern uint32_t randuint();
extern void shuffle(std::vector<std::string> *);

}

#endif
