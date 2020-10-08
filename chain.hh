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
  bool prepared;
  double rmul;

  std::vector<Cortex*> ctxv;
  void prepare(int iw, int ih);
  void push(const std::string &fn, int mode);

  double *kinp();
  double *kout();

  double *synth(int stop = 0);
  void learn(double mul);
  void target(const double *);

  void load();
  void save();
};

}

#endif
