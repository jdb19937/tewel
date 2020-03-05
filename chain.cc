#define __MAKEMORE_CHAIN_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include "cortex.hh"
#include "chain.hh"

namespace makemore {

Chain::Chain() {
  head = NULL;
  tail = NULL;
}

Chain::~Chain() {

}

void Chain::push(Cortex *ctx) {
  if (!ctxv.size()) {
    assert(!head);
    head = ctx;
  }

  ctxv.push_back(ctx);
  tail = ctx;
}


double *Chain::kinp() {
  assert(head);
  return head->kinp;
}

double *Chain::kout() {
  assert(tail);
  return tail->kout;
}

void Chain::synth() {
  int n = ctxv.size();
  assert(n > 0);

  ctxv[0]->synth();
  for (int i = 1; i < n; ++i)
    ctxv[i]->synth(ctxv[i - 1]->kout);
}

void Chain::target(const double *ktgt) {
  assert(tail);
  tail->target(ktgt);
}

void Chain::learn(double mul) {
  int n = ctxv.size();
  assert(n > 0);

  ctxv[n - 1]->learn(mul);
  for (int i = n - 2; i > 0; --i)
    ctxv[i]->learn(ctxv[i + 1]->kinp, mul);
}

}
