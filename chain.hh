#ifndef __MAKEMORE_CHAIN_HH__
#define __MAKEMORE_CHAIN_HH__

#include <vector>

#include "paracortex.hh"

namespace makemore {

struct Chain {
  Chain();
  ~Chain();

  Paracortex *head;
  Paracortex *tail;
  bool prepared;
  double rmul;

  Paracortex *brak;
  void enbrak();
  void debrak();

  std::vector<Paracortex*> ctxv;
  void prepare(int iw, int ih);
  void push(const std::string &fn, int mode);

  double *kinp();
  double *kout();

  double *synth(int stop = 0);
  void learn(double mul);
  void target(const double *);

  void load();
  void save();

  unsigned long rounds();
  double rms();
};

}

#endif
