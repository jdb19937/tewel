#ifndef __MAKEMORE_CHAIN_HH__
#define __MAKEMORE_CHAIN_HH__

#include <vector>

#include "cortex.hh"

namespace makemore {

struct Chain {
  Chain();
  ~Chain();

  Cortex *head;
  Cortex *tail;

  std::vector<Cortex*> ctxv;
  void push(Cortex *ctx);

  double *kinp();
  double *kout();

  void synth();
  void learn(double mul);
  void target(const double *);
};

}

#endif
