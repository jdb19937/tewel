#define __MAKEMORE_CHAIN_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include "cortex.hh"
#include "youtil.hh"
#include "chain.hh"
#include "colonel.hh"

namespace makemore {

Chain::Chain() {
  head = NULL;
  tail = NULL;
  prepared = false;
  rmul = 1.0;

  brak = NULL;
}

Chain::~Chain() {
  for (auto ctx : ctxv)
    delete ctx;
}

void Chain::enbrak() {
  assert(!brak);
  brak = new Paracortex;
  brak->rmul = rmul;
}

void Chain::debrak() {
  assert(brak);
  assert(brak->ctxv.size());

  if (!ctxv.size()) {
    assert(!head);
    head = brak;
  }
  ctxv.push_back(brak);
  tail = brak;

  brak = NULL;
}

void Chain::push(const std::string &fn, int mode) {
  if (fn == "[") {
    info("enbrak");
    enbrak();
    return;
  }

  if (fn == "]") {
    info("debrak");
    debrak();
    return;
  }

  info("adding " + fn + " to chain");

  if (brak) {
    brak->push(fn, mode);
  } else {
    Paracortex *ctx = new Paracortex;
    ctx->rmul = rmul;
    ctx->push(fn, mode);

    if (!ctxv.size()) {
      assert(!head);
      head = ctx;
    }
    ctxv.push_back(ctx);
    tail = ctx;
  }
}


double *Chain::kinp() {
  assert(head);
  return head->kinp();
}

double *Chain::kout() {
  assert(tail);
  return tail->kout();
}

void Chain::prepare(int iw, int ih) {
  assert(!prepared);
  assert(!brak);

  int n = ctxv.size();
  assert(n > 0);

  ctxv[0]->prepare(iw, ih);
  for (int i = 1; i < n; ++i) {
    ctxv[i]->prepare(ctxv[i - 1]->ow, ctxv[i - 1]->oh);
    assert(ctxv[i]->ic == ctxv[i - 1]->oc);
  }

  prepared = true;
}

double *Chain::synth(int stop) {
  assert(prepared);

  int n = ctxv.size();
  assert(n > 0);

  if (stop <= 0)
    stop = n + stop;

  ctxv[0]->synth();
  for (int i = 1; i < stop; ++i)
    ctxv[i]->synth(ctxv[i - 1]->kout());

  return ctxv[stop - 1]->kout();
}

void Chain::target(const double *ktgt) {
  assert(tail);
  tail->target(ktgt);
}

void Chain::learn(double mul) {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  ctxv[n - 1]->learn(mul);
  for (int i = n - 2; i >= 0; --i) {
    kcopy(ctxv[i + 1]->kinp(), ctxv[i]->owhc, ctxv[i]->kout());
    ctxv[i]->learn(mul);
  }
}

void Chain::load() {
  for (auto ctx : ctxv)
    ctx->load();
}

void Chain::save() {
  for (auto ctx : ctxv)
    ctx->save();
}

unsigned long Chain::rounds() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  return ctxv[n - 1]->rounds();
}

double Chain::rms() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  return ctxv[n - 1]->rms();
}

}
