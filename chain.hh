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
  int bn;
  float rmul;

  Paracortex *brak;
  void enbrak();
  void debrak();

  std::vector<Paracortex*> ctxv;
  // tailw/tailh > 0: the tail stage is prepared at these input dims and synth() feeds it an
  // external input (tailin) instead of the previous stage's output; style still comes from the
  // previous stage, and no gradient flows from the tail's input back into the chain
  bool tail_override;
  void prepare(int iw, int ih, int bn = 1, int tailw = 0, int tailh = 0);
  void push(const std::string &fn, int mode);

  float *kinp();
  float *kout();

  std::vector<float*> klat;

  float *synth(int stop = 0, float latnoise = 0.0, const float *tailin = NULL);
  void learn(float mul, float latreg = 0.0);
  void accumulate(float latreg = 0.0);
  void _backprop(float mul, float latreg, bool apply);
  void target(const float *);

  void load();
  void save();

  unsigned long rounds();
  float rms();
};

}

#endif
