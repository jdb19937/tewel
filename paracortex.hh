#ifndef __MAKEMORE_PARACORTEX_HH__
#define __MAKEMORE_PARACORTEX_HH__ 1

#include <vector>

#include "cortex.hh"

namespace makemore {

struct Paracortex {
  Paracortex();
  ~Paracortex();

  bool prepared;
  double rmul;

  double *_kinp, *_kout;

  double *kinp();
  double *kout();

  int iw, ih, ic, iwhc;
  int ow, oh, oc, owhc;

  std::vector<Cortex*> ctxv;
  void prepare(int iw, int ih);
  void push(const std::string &fn, int mode);

  double *synth(double *kinp = NULL);
  void learn(double mul);
  void target(const double *);

  void load();
  void save();

  unsigned long rounds();
  double rms();
  std::string fn();
};

}

#endif
