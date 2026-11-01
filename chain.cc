#define __MAKEMORE_CHAIN_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include "cortex.hh"
#include "youtil.hh"
#include "chain.hh"
#include "random.hh"
#include "colonel.hh"

namespace makemore {

Chain::Chain() {
  tail_override = false;
  head = NULL;
  tail = NULL;
  prepared = false;
  bn = 1;
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


float *Chain::kinp() {
  assert(head);
  return head->kinp();
}

float *Chain::kout() {
  assert(tail);
  return tail->kout();
}

void Chain::prepare(int iw, int ih, int _bn, int tailw, int tailh) {
  assert(!prepared);
  assert(!brak);
  bn = _bn;

  int n = ctxv.size();
  assert(n > 0);

  tail_override = (tailw > 0);
  if (tail_override)
    assert(n > 1);

  ctxv[0]->prepare(iw, ih, bn);
  klat.resize(n, NULL);
  for (int i = 1; i < n; ++i) {
    if (tail_override && i == n - 1) {
      ctxv[i]->prepare(tailw, tailh, bn);
    } else {
      ctxv[i]->prepare(ctxv[i - 1]->ow, ctxv[i - 1]->oh, bn);
      assert(ctxv[i]->ic == ctxv[i - 1]->oc);
    }
    kmake(&klat[i], ctxv[i]->iwhc);
  }

  prepared = true;
}

float *Chain::synth(int stop, float latnoise, const float *tailin) {
  assert(prepared);

  int n = ctxv.size();
  assert(n > 0);

  if (stop <= 0)
    stop = n + stop;

  // a stage with styl layers takes its style from the previous stage's output
  // (or from its own input when it is the head); the pointer is only valid
  // once that output buffer exists, i.e. right before the stage runs
  {
    Cortex *c = ctxv[0]->ctxv[0];
    if (c->sdim > 0) {
      assert(ctxv[0]->ctxv.size() == 1);
      assert(c->sdim * c->bn == ctxv[0]->iwhc);
      c->kstyle = ctxv[0]->kinp();
    }
  }
  ctxv[0]->synth();
  for (int i = 1; i < stop; ++i) {
    float *k = ctxv[i - 1]->kout();
    if (latnoise > 0)
      kaddnoise(k, ctxv[i - 1]->owhc, latnoise);
    Cortex *c = ctxv[i]->ctxv[0];
    if (c->sdim > 0) {
      assert(ctxv[i]->ctxv.size() == 1);
      assert(c->sdim * c->bn == ctxv[i - 1]->owhc);
      c->kstyle = k;
    }
    if (tail_override && i == n - 1) {
      assert(tailin);
      ctxv[i]->synth((float *)tailin);
    } else {
      ctxv[i]->synth(k);
    }
  }

  return ctxv[stop - 1]->kout();
}

void Chain::target(const float *ktgt) {
  assert(tail);
  tail->target(ktgt);
}

// latreg > 0 adds an L2 penalty on every intermediate (latent) activation:
// gradient (ascent convention) is -latreg * z.
void Chain::_backprop(float mul, float latreg, bool apply) {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (latreg > 0)
    for (int i = 1; i < n; ++i)
      kcopy(ctxv[i]->kinp(), ctxv[i]->iwhc, klat[i]);

  if (apply) ctxv[n - 1]->learn(mul); else ctxv[n - 1]->accumulate();
  for (int i = n - 2; i >= 0; --i) {
    if (tail_override && i == n - 2)
      kzero(ctxv[i]->kout(), ctxv[i]->owhc);   // the tail's input gradient belongs to the condition, not to us
    else
      kcopy(ctxv[i + 1]->kinp(), ctxv[i]->owhc, ctxv[i]->kout());
    if (ctxv[i + 1]->ctxv[0]->sdim > 0)
      kaddvec(ctxv[i]->kout(), ctxv[i + 1]->ctxv[0]->kstylegrad, ctxv[i]->owhc, ctxv[i]->kout());
    if (latreg > 0) {
      kmul(klat[i + 1], -latreg, ctxv[i]->owhc, klat[i + 1]);
      kaddvec(ctxv[i]->kout(), klat[i + 1], ctxv[i]->owhc, ctxv[i]->kout());
    }
    if (apply) ctxv[i]->learn(mul); else ctxv[i]->accumulate();
  }
}

void Chain::learn(float mul, float latreg) {
  _backprop(mul, latreg, true);
}

void Chain::accumulate(float latreg) {
  _backprop(0.0, latreg, false);
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

float Chain::rms() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  return ctxv[n - 1]->rms();
}

}
