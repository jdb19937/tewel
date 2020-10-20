#define __MAKEMORE_PARACORTEX_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <vector>

#include "cortex.hh"
#include "youtil.hh"
#include "paracortex.hh"
#include "colonel.hh"

namespace makemore {

Paracortex::Paracortex() {
  prepared = false;
  rmul = 1.0;
  _kinp = NULL;
  _kout = NULL;

  iw = ih = ic = iwhc = 0;
  ow = oh = oc = owhc = 0;
}

Paracortex::~Paracortex() {
  for (auto ctx : ctxv)
    delete ctx;

  if (_kinp)
    kfree(_kinp);
  if (_kout)
    kfree(_kout);
}

void Paracortex::push(const std::string &fn, int mode) {
  Cortex *ctx = new Cortex(fn, mode);

  info("adding " + fn + " to paracortex");

  ctx->rmul = rmul;
  ctxv.push_back(ctx);
  ic += ctx->ic;
}


void Paracortex::prepare(int _iw, int _ih) {
  assert(!prepared);
  int n = ctxv.size();
  assert(n > 0);

  iw = _iw;
  ih = _ih;

  ctxv[0]->prepare(iw, ih);
  ic = ctxv[0]->ic;

  ow = ctxv[0]->ow;
  oh = ctxv[0]->oh;
  oc = ctxv[0]->oc;

  if (n == 1) {
    iwhc = iw * ih * ic;
    owhc = ow * oh * oc;
    prepared = true;
    return;
  }

  for (int i = 1; i < n; ++i) {
    ctxv[i]->prepare(iw, ih);
    ic += ctxv[i]->ic;

    assert(ctxv[i]->ow == ow);
    assert(ctxv[i]->oh == oh);
    oc += ctxv[i]->oc;
  }

  iwhc = iw * ih * ic;
  owhc = ow * oh * oc;

  kmake(&_kinp, iwhc);
  kmake(&_kout, owhc);

  prepared = true;
}

double *Paracortex::kinp() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (n == 1) {
    return ctxv[0]->kinp;
  }

  assert(_kinp);
  return _kinp;
}

double *Paracortex::kout() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (n == 1) {
    return ctxv[0]->kout;
  }

  assert(_kout);
  return _kout;
}

double *Paracortex::synth(double *kinp) {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (n == 1) {
    ctxv[0]->synth(kinp);
    return ctxv[0]->kout;
  }

  if (kinp)
    kcopy(kinp, iwhc, _kinp);

  int off = 0;

  for (int i = 0; i < n; ++i) {
    ksplice(_kinp, iw * ih, ic, off, ctxv[i]->ic, ctxv[i]->kinp, ctxv[i]->ic, 0);
    off += ctxv[i]->ic;

    ctxv[i]->synth();
  }
  assert(off == ic);

  off = 0;
  for (int i = 0; i < n; ++i) {
    ksplice(ctxv[i]->kout, ow * oh, ctxv[i]->oc, 0, ctxv[i]->oc, _kout, oc, off);
    off += ctxv[i]->oc;
  }
  assert(off == oc);

  return _kout;
}

void Paracortex::target(const double *ktgt) {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (n == 1) {
    ctxv[0]->target(ktgt);
    return;
  }

  ksubvec(ktgt, _kout, owhc, _kout);

  int off = 0;
  for (int i = 0; i < n; ++i) {
    ksplice(_kout, ow * oh, oc, off, ctxv[i]->oc, ctxv[i]->kout, ctxv[i]->oc, 0);
    off += ctxv[i]->oc;
  }
  assert(off == oc);
}

void Paracortex::learn(double mul) {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  if (n == 1) {
    ctxv[0]->learn(mul);
    return;
  }

  int off = 0;
  for (int i = 0; i < n; ++i) {
    ksplice(_kout, ow * oh, oc, off, ctxv[i]->oc, ctxv[i]->kout, ctxv[i]->oc, 0);
    off += ctxv[i]->oc;

    ctxv[0]->learn(mul);
  }
  assert(off == oc);

  off = 0;
  for (int i = 0; i < n; ++i) {
    ksplice(ctxv[i]->kinp, iw * ih, ctxv[i]->ic, 0, ctxv[i]->ic, _kinp, ic, off);
    off += ctxv[i]->ic;
  }
  assert(off == ic);
}

void Paracortex::load() {
  for (auto ctx : ctxv)
    ctx->load();
}

void Paracortex::save() {
  for (auto ctx : ctxv)
    ctx->save();
}

unsigned long Paracortex::rounds() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  return ctxv[0]->rounds;
}

std::string Paracortex::fn() {
  assert(prepared);
  int n = ctxv.size();
  assert(n > 0);

  return ctxv[0]->fn;
}


double Paracortex::rms() {
  assert(prepared);

  double z = 0;
  for (auto ctx : ctxv)
    z += ctx->rms * ctx->rms * ctx->oc;
  z /= (double)oc;
  z = sqrt(z);

  return z;
}



}
