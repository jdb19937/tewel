#ifndef __MAKEMORE_PARACORTEX_HH__
#define __MAKEMORE_PARACORTEX_HH__ 1

#include <vector>

#include "cortex.hh"

namespace makemore {

struct Paracortex {
  Paracortex();
  ~Paracortex();

  bool prepared;
  float rmul;

  float *_kinp, *_kout;

  float *kinp();
  float *kout();

  int bn;                 // batch size
  int iw, ih, ic, iwhc;   // iwhc/owhc include the batch: bn * w * h * c
  int ow, oh, oc, owhc;

  std::vector<Cortex*> ctxv;
  void prepare(int iw, int ih, int bn = 1);
  void push(const std::string &fn, int mode);

  float *synth(float *kinp = NULL);
  void learn(float mul);
  void accumulate();
  void propback();   // backprop only: no weight update, nothing accumulated
  void target(const float *);

  void load();
  void save();

  unsigned long rounds();
  float rms();
  std::string fn();
};

}

#endif
