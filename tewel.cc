#define __MAKEMORE_TEWEL_CC__ 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#include <math.h>

#include "colonel.hh"
#include "cortex.hh"
#include "paracortex.hh"
#include "youtil.hh"
#include "random.hh"
#include "rando.hh"
#include "kleption.hh"
#include "cmdline.hh"
#include "display.hh"
#include "camera.hh"
#include "picpipes.hh"
#include "server.hh"
#include "egserver.hh"
#include "tileserver.hh"
#include "chain.hh"

using namespace makemore;

namespace makemore{
  extern int verbose;
}

static void usage() {
  fprintf(stderr,
    "Usage:\n"
    "        tewel help\n"
    "                print this message\n"
    "\n"
    "        tewel make new.ctx spec=new.txt ...\n"
    "                create a new generator\n"
    "        tewel reset any.ctx\n"
    "                randomize generator state\n"
    "        tewel edit any.ctx [key=val ...]\n"
    "                display or edit header contents\n"
    "        tewel spec any.ctx > any.txt\n"
    "                display layer headers\n"
    "\n"
    "        tewel synth gen.ctx src=... out=...\n"
    "                run generator\n"
    "\n"
    "        tewel learnauto gen.ctx src=...\n"
    "                train generator on sample paired with self\n"
    "        tewel learnfunc gen.ctx src=... tgt=...\n"
    "                train generator on paired samples\n"
    "        tewel learnhans gen.ctx src=... tgt=... dis=dis.ctx\n"
    "                train generator on paired samples versus discriminator\n"
    "        tewel learngan gen.ctx tgt=... dis=dis.ctx [ema=gen-ema.ctx]\n"
    "                train generator from gaussian latents versus discriminator\n"
    "                options: replay=N replayfrac=f (discriminator replays past fakes),\n"
    "                disema=file disemabeta=b (generator plays against an EMA discriminator)\n"
  );
}

static void uerror(const std::string &str) {
  fprintf(stderr, "Error: %s\n", str.c_str());
  usage();
  exit(1);
}

static double calceta(double dt, int rep, int reps, int _round, int rounds, int repint) {
  double eta = -1;
  if (reps >= 0) {
    eta = (reps - rep) * dt;
    if (rounds >= 0)
      eta = std::min(eta, (rounds - _round) * dt / (double)repint);
  } else if (rounds >= 0) {
    eta = (rounds - _round) * dt / (double)repint;
  }
  return eta;
}


void learnfunc(
  Kleption *src,
  Kleption *tgt,
  Chain *chn,
  int repint,
  double mul,
  long reps,
  double stoploss,
  long stoprounds,
  bool lossl1,
  double lossdelta,
  double latnoise,
  double latreg
) {
  Paracortex *gen = chn->tail;
  int bn = chn->bn;
  int sin = chn->head->iwhc / bn;   // per-sample sizes
  int sout = gen->owhc / bn;

  if (chn->rounds() >= stoprounds && stoprounds >= 0) {
    warning("genrounds reached stoprounds value, all done");
    return;
  }

  float *ktgt;
  kmake(&ktgt, gen->owhc);
  float *kerr;
  kmake(&kerr, gen->owhc);

  double t0 = now();
  int rep = 0;
  long it = 0;
  double pixrms = 0;

  while (reps < 0 || rep < reps) {
    bool report = ((it + 1) % repint == 0);

    for (int b = 0; b < bn; ++b)
      Kleption::pick_pair(src, chn->kinp() + b * sin, tgt, ktgt + b * sout);
    chn->synth(0, latnoise);

    if (report) {
      ksubvec(ktgt, chn->kout(), gen->owhc, kerr);
      pixrms = sqrt(ksumsq(kerr, gen->owhc) / (double)gen->owhc);
    }

    chn->target(ktgt);
    if (lossl1)
      kcharb(chn->kout(), gen->owhc, lossdelta);
    chn->learn(mul, latreg);

    ++it;
    if (report) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g pixrms=%g\n",
         gen->fn().c_str(), gen->rounds(),
         dt, calceta(dt, rep, reps, gen->rounds(), stoprounds, repint),
        gen->rms(), pixrms
      );

      if (pixrms < stoploss) {
        warning("pixrms reached stoploss value, all done");
        break;
      }

      if (chn->rounds() >= stoprounds && stoprounds >= 0) {
        warning("genrounds reached stoprounds value, all done");
        break;
      }

      ++rep;
    }
  }

  kfree(ktgt);
  kfree(kerr);
}


// Conditional (optionally tethered) GAN training, batched.
//
// Per iteration: `disreps` discriminator updates, each on a batch of
// (fake, real) pairs, then one generator update on a batch of samples.
// The generator gradient is the discriminator's gradient (target "real")
// plus `tether` times the pixel-loss gradient.
// Discriminator targets: real = 0, fake = 1 (least-squares GAN).
void learnhans(
  Kleption *src,
  Kleption *tgt,
  Chain *chn,
  Cortex *dis,
  Cortex *cnd,
  int repint,
  double mul,
  long reps,
  long disreps,
  long stoprounds,
  double genmul0, double dismul0, double nzmul,
  double tether,
  bool lossl1,
  double lossdelta,
  double latnoise,
  double latreg
) {
  Paracortex *gen = chn->tail;
  int bn = chn->bn;
  int sin = chn->head->iwhc / bn;
  int sout = gen->owhc / bn;

  if (chn->rounds() >= stoprounds && stoprounds >= 0) {
    warning("genrounds reached stoprounds value, all done");
    return;
  }

  assert(dis->bn == bn);
  if (cnd) {
    assert(cnd->bn == bn);
    assert(cnd->iw == chn->head->iw);
    assert(cnd->ih == chn->head->ih);
    assert(cnd->ic == chn->head->ic);
    assert(cnd->ow == gen->ow);
    assert(cnd->oh == gen->oh);
  }

  assert(dis->iw == gen->ow);
  assert(dis->ih == gen->oh);
  assert(dis->ic == (cnd ? cnd->oc : 0) + gen->oc);

  int disin = dis->iwhc * bn, disout = dis->owhc * bn;
  int pix = bn * gen->ow * gen->oh;   // pixels across the batch

  float *ktmp;
  kmake(&ktmp, disin);
  float *ktgt;
  kmake(&ktgt, gen->owhc);
  float *kgen;
  kmake(&kgen, gen->owhc);
  float *kerr;
  kmake(&kerr, gen->owhc);

  float *kreal;
  kmake(&kreal, disout);
  kfill(kreal, disout, 0.0);

  float *kfake;
  kmake(&kfake, disout);
  kfill(kfake, disout, 1.0);

  double t0 = now();
  int rep = 0;
  long it = 0;
  double pixrms = 0, advrms = 0;

  // pick a batch, run cnd and gen, fill dis->kinp = [gen out | cnd out],
  // keep a clean copy in ktmp and the gen output in kgen
  auto feed = [&](double nz) {
    for (int b = 0; b < bn; ++b)
      Kleption::pick_pair(src, chn->kinp() + b * sin, tgt, ktgt + b * sout);
    if (cnd)
      ksplice(cnd->synth(chn->kinp()), pix, dis->ic - gen->oc, 0, dis->ic - gen->oc, dis->kinp, dis->ic, gen->oc);
    chn->synth(0, latnoise);
    kcopy(gen->kout(), gen->owhc, kgen);
    ksplice(gen->kout(), pix, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    kcopy(dis->kinp, disin, ktmp);
    if (nz > 0)
      kaddnoise(dis->kinp, disin, nz);
  };

  while (reps < 0 || rep < reps) {
    bool report = ((it + 1) % repint == 0);

    double nz = 0.0;
    if (nzmul > 0 && dis->rms < 0.5 && dis->rms > 0)
      nz = (0.5 / dis->rms - 1.0) * nzmul;

    // discriminator phase
    for (long r = 0; r < disreps; ++r) {
      feed(nz);
      dis->synth();
      dis->target(kfake);
      dis->accumulate();

      kcopy(ktmp, disin, dis->kinp);
      ksplice(ktgt, pix, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
      if (nz > 0)
        kaddnoise(dis->kinp, disin, nz);
      dis->synth();
      dis->target(kreal);
      dis->learn(dismul0 * mul);
    }

    // generator phase
    feed(nz);
    dis->synth();
    dis->target(kreal);
    dis->propback();
    ksplice(dis->kinp, pix, dis->ic, 0, gen->oc, gen->kout(), gen->oc, 0);

    if (report)
      advrms = sqrt(ksumsq(gen->kout(), gen->owhc) / (double)gen->owhc);

    ksubvec(ktgt, kgen, gen->owhc, kerr);
    if (report)
      pixrms = sqrt(ksumsq(kerr, gen->owhc) / (double)gen->owhc);

    if (tether > 0) {
      if (lossl1)
        kcharb(kerr, gen->owhc, lossdelta);
      kmul(kerr, tether, gen->owhc, kerr);
      kaddvec(gen->kout(), kerr, gen->owhc, gen->kout());
    }

    chn->learn(genmul0 * mul, latreg);

    ++it;
    if (report) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();
      dis->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g disrms=%g advrms=%g pixrms=%g\n",
         gen->fn().c_str(), gen->rounds(),
         dt, calceta(dt, rep, reps, gen->rounds(), stoprounds, repint),
        gen->rms(), dis->rms, advrms, pixrms
      );

      if (chn->rounds() >= stoprounds && stoprounds >= 0) {
        warning("genrounds reached stoprounds value, all done");
        break;
      }

      ++rep;
    }
  }

  kfree(ktmp);
  kfree(ktgt);
  kfree(kgen);
  kfree(kerr);
  kfree(kreal);
  kfree(kfake);
}

// Unconditional GAN: gaussian latent -> generator chain -> image, versus a
// discriminator on real images (tgt). Options follow StyleGAN2's recipe where
// the chain abstraction allows: non-saturating logistic loss (loss=ns) or
// least squares (loss=ls), lazy R1 gradient penalty on real images computed
// by finite differences (r1=gamma, r1int=interval), and an exponential
// moving average copy of the generator (ema=file, emabeta=beta).
void learngan(
  Kleption *tgt,
  Chain *chn,
  Cortex *dis,
  Cortex *ema,
  int repint,
  double mul,
  long reps,
  long disreps,
  long stoprounds,
  double genmul, double dismul,
  bool nsloss,
  double r1, int r1int, double r1eps,
  double emabeta,
  Cortex *disema, double disemabeta,   // optional EMA of the discriminator; the generator plays against it
  int replay, double replayfrac,       // optional ring buffer of past fake batches shown again to the discriminator
  Kleption *cnd, int cc                // optional condition paired with tgt: the generator's tail input, spliced into dis input
) {
  Paracortex *gen = chn->tail;
  int bn = chn->bn;
  int zn = chn->head->iwhc;          // whole batch of latents
  int sout = gen->owhc / bn;
  int goc = gen->oc;
  int pn = bn * gen->ow * gen->oh;   // pixels in a batch
  int cellc = cc * gen->ow * gen->oh;   // condition floats per sample
  if (!cnd) cc = 0;

  if (chn->rounds() >= stoprounds && stoprounds >= 0) {
    warning("genrounds reached stoprounds value, all done");
    return;
  }

  assert(dis->bn == bn);
  assert(dis->iw == gen->ow);
  assert(dis->ih == gen->oh);
  assert(dis->ic == gen->oc + cc);
  int disin = dis->iwhc * bn, disout = dis->owhc * bn;
  assert(disin == gen->owhc + bn * cellc);

  float *ktgt;  kmake(&ktgt, gen->owhc);   // real images
  float *krin;  kmake(&krin, disin);        // real dis input (image + condition)
  float *kfin;  kmake(&kfin, disin);        // fake dis input
  float *kcnd = NULL, *kcndr = NULL;        // conditions for the fake and real batches
  if (cc) { kmake(&kcnd, bn * cellc); kmake(&kcndr, bn * cellc); }
  float *kxp;   kmake(&kxp, disin);
  float *kg;    kmake(&kg, disin);
  float *kreal; kmake(&kreal, disout); kfill(kreal, disout, 0.0);
  float *kfake; kmake(&kfake, disout); kfill(kfake, disout, 1.0);

  // replay ring of past fake batches
  float *kring = NULL; int ringn = 0, ringi = 0;
  if (replay > 0)
    kmake(&kring, (long)replay * disin);
  Cortex *gdis = disema ? disema : dis;   // discriminator the generator is trained against

  double t0 = now();
  int rep = 0;
  long it = 0;
  double dreal = 0, dfake = 0, gnorm = 0;

  // dis input = [image | condition] per pixel
  auto splice_in = [&](const float *kimg, const float *kc, float *kdst) {
    ksplice(kimg, pn, goc, 0, goc, kdst, dis->ic, 0);
    if (cc) ksplice(kc, pn, cc, 0, cc, kdst, dis->ic, goc);
  };
  auto gen_fake = [&]() {
    kzero(chn->kinp(), zn);
    kaddnoise(chn->kinp(), zn, 1.0);
    if (cc)
      for (int b = 0; b < bn; ++b)
        cnd->pick(kcnd + b * cellc);
    chn->synth(0, 0.0, kcnd);
    splice_in(gen->kout(), kcnd, kfin);
    kcopy(kfin, disin, dis->kinp);
  };
  auto ktmp_fake = [&]() { return kfin; };

  while (reps < 0 || rep < reps) {
    bool report = ((it + 1) % repint == 0);
    bool dor1 = (r1 > 0 && r1int > 0 && it % r1int == 0);

    // discriminator phase
    for (long r = 0; r < disreps; ++r) {
      gen_fake();
      dis->synth();
      dfake = ksum(dis->kout, disout) / disout;
      if (nsloss) klogis(dis->kout, disout, -1); else dis->target(kfake);
      dis->accumulate();

      if (kring) {
        if (ringn > 0) {
          // one stored batch, weighted by replayfrac
          int j = randuint() % ringn;
          dis->synth(kring + (long)j * disin);
          if (nsloss) klogis(dis->kout, disout, -1); else dis->target(kfake);
          if (replayfrac != 1.0) kmul(dis->kout, (float)replayfrac, disout, dis->kout);
          dis->accumulate();
        }
        kcopy(ktmp_fake(), disin, kring + (long)ringi * disin);
        ringi = (ringi + 1) % replay;
        if (ringn < replay) ++ringn;
      }

      for (int b = 0; b < bn; ++b) {
        if (cc) Kleption::pick_pair(tgt, ktgt + b * sout, cnd, kcndr + b * cellc);
        else tgt->pick(ktgt + b * sout);
      }
      splice_in(ktgt, kcndr, krin);
      kcopy(krin, disin, dis->kinp);
      dis->synth();
      dreal = ksum(dis->kout, disout) / disout;
      if (nsloss) klogis(dis->kout, disout, +1); else dis->target(kreal);

      if (dor1 && r + 1 == disreps) {
        dis->accumulate();

        // g = d(sum D)/dx at the real batch
        dis->synth(krin);
        kfill(dis->kout, disout, 1.0);
        dis->propback();
        gnorm = sqrt(ksumsq(dis->kinp, disin));
        double eps = r1eps * sqrt((double)disin);
        if (gnorm > 0) {
          // x' = x + eps * g / |g|
          kmul(dis->kinp, (float)(eps / gnorm), disin, kg);
          kaddvec(krin, kg, disin, kxp);
          dis->synth(kxp);
          double s1 = ksum(dis->kout, disout);
          dis->synth(krin);
          double s0 = ksum(dis->kout, disout);
          double delta = (s1 - s0) / eps;                 // ~ |g|
          double c = r1 * r1int * delta / eps;            // lazy R1: scaled by interval
          // penalty R = r1/2 * delta^2; descend: kout = +c at x, -c at x'
          kfill(dis->kout, disout, (float)c);
          dis->accumulate();
          dis->synth(kxp);
          kfill(dis->kout, disout, (float)(-c));
        }
        dis->learn(dismul * mul);
      } else {
        dis->learn(dismul * mul);
      }
    }

    // generator phase (against the EMA discriminator when one is given)
    gen_fake();
    if (gdis != dis) kcopy(kfin, disin, gdis->kinp);
    gdis->synth();
    if (nsloss) klogis(gdis->kout, disout, +1); else gdis->target(kreal);
    gdis->propback();
    ksplice(gdis->kinp, pn, dis->ic, 0, goc, gen->kout(), goc, 0);   // image part of the input gradient
    chn->learn(genmul * mul);

    if (ema)
      ema->ema(*gen->ctxv[0], (float)emabeta);
    if (disema)
      disema->ema(*dis, (float)disemabeta);

    ++it;
    if (report) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();
      dis->save();
      if (ema)
        ema->save();
      if (disema)
        disema->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g dreal=%g dfake=%g gnorm=%g disrms=%g\n",
         gen->fn().c_str(), gen->rounds(),
         dt, calceta(dt, rep, reps, gen->rounds(), stoprounds, repint),
        dreal, dfake, gnorm, dis->rms
      );

      if (chn->rounds() >= stoprounds && stoprounds >= 0) {
        warning("genrounds reached stoprounds value, all done");
        break;
      }

      ++rep;
    }
  }

  kfree(ktgt);
  kfree(krin);
  kfree(kfin);
  if (kcnd) kfree(kcnd);
  if (kcndr) kfree(kcndr);
  kfree(kxp);
  kfree(kg);
  kfree(kreal);
  kfree(kfake);
  if (kring)
    kfree(kring);
}

void learn(
  Kleption *src,
  Kleption *tgt,
  Chain *chn,
  Cortex *dis,
  Cortex *cnd,
  Cortex *enc,
  Cortex *inf,
  int repint,
  double mul,
  long reps,
  long stoprounds
) {
  if (chn->rounds() >= stoprounds && stoprounds >= 0) {
    warning("genrounds reached stoprounds value, all done");
    return;
  }

  assert(enc->ow == chn->head->iw);
  assert(enc->oh == chn->head->ih);

  assert(inf->iw == enc->iw);
  assert(inf->ih == enc->ih);
  assert(inf->ic == enc->ic);
  assert(inf->ow == chn->head->iw);
  assert(inf->oh == chn->head->ih);

  assert(enc->oc + inf->oc == chn->head->ic);

  assert(cnd->iw == inf->ow);
  assert(cnd->ih == inf->oh);
  assert(cnd->ic == inf->oc);
  assert(cnd->ow == chn->tail->ow);
  assert(cnd->oh == chn->tail->oh);

  assert(dis->iw == chn->tail->ow);
  assert(dis->ih == chn->tail->oh);
  assert(dis->ic == cnd->oc + chn->tail->oc);

  float *ktmp;
  kmake(&ktmp, dis->iwhc);

  float *kinp;
  kmake(&kinp, enc->iwhc);

  float *ktgt;
  kmake(&ktgt, chn->tail->owhc);

  float *kreal;
  kmake(&kreal, dis->owhc);
  kfill(kreal, dis->owhc, 0.5);

  float *kfake;
  kmake(&kfake, dis->owhc);
  kfill(kfake, dis->owhc, -0.5);

  float *kloss;
  kmake(&kloss, dis->owhc);

  float *kimpr;
  kmake(&kimpr, dis->owhc);

  float *kstab;
  kmake(&kstab, dis->owhc);
  kfill(kstab, dis->owhc, 0.0);

  double t0 = now();
  int rep = 0;
  long i = 0;

  while (reps < 0 || rep < reps) {
    Kleption::pick_pair(src, enc->kinp, tgt, ktgt);
    enc->synth();

    kcopy(enc->kinp, enc->iwhc, inf->kinp);
    inf->synth();
      
    kcopy(inf->kout, cnd->iwhc, cnd->kinp);
    cnd->synth();

    ksplice(enc->kout, enc->ow * enc->oh, enc->oc, 0, enc->oc, chn->kinp(), chn->head->ic, 0);
    ksplice(inf->kout, inf->ow * inf->oh, inf->oc, 0, inf->oc, chn->kinp(), chn->head->ic, enc->oc);
    chn->synth();

    ksplice(chn->kout(), chn->tail->ow * chn->tail->oh, chn->tail->oc, 0, chn->tail->oc, dis->kinp, dis->ic, 0);
    ksplice(cnd->kout, cnd->ow * cnd->oh, cnd->oc, 0, cnd->oc, dis->kinp, dis->ic, chn->tail->oc);
    kcopy(dis->kinp, dis->iwhc, ktmp);
    dis->synth();


    dis->pushbuf();
    dis->target(kreal);
    dis->propback();
    ksplice(dis->kinp, dis->iw * dis->ih, dis->ic, 0, chn->tail->oc, chn->kout(), chn->tail->oc, 0);
    chn->learn(mul);

    ksplice(chn->kinp(), chn->head->iw * chn->head->ih, chn->head->ic, 0, enc->oc, enc->kout, enc->oc, 0);
    enc->learn(mul);


    dis->popbuf();
    dis->target(kfake);
    dis->accumulate();

    kcopy(ktmp, dis->iwhc, dis->kinp);
    ksplice(ktgt, chn->tail->ow * chn->tail->oh, chn->tail->oc, 0, chn->tail->oc, dis->kinp, dis->ic, 0);
    dis->synth();
    dis->target(kreal);
    dis->learn(mul);

    ++i;

    if (i % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();
      dis->save();
      enc->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g encrms=%g genrms=%g disrms=%g\n",
         chn->tail->fn().c_str(), chn->rounds(),
         dt, calceta(dt, rep, reps, chn->rounds(), stoprounds, repint),
        enc->rms, chn->rms(), dis->rms
      );

      if (chn->rounds() >= stoprounds && stoprounds >= 0) {
        warning("genrounds reached stoprounds value, all done");
        break;
      }

      ++rep;
    }
  }

  kfree(ktmp);
  kfree(ktgt);
  kfree(kinp);
}

static void cpumatxpose(double *a, int dim) {
  double *b = new double[dim * dim];

  for (int y = 0; y < dim; ++y)
    for (int x = 0; x < dim; ++x)
      b[y + x * dim] = a[x + y * dim];

  memcpy(a, b, dim * dim * sizeof(double));
  delete[] b;
}

static void cpumatmat(const double *a, int aw, int ah, const double *b, int bw, int bh, double *c) {
  int ch = ah;
  int cw = bw;

  assert(bh == aw);

  for (int cy = 0; cy < ch; ++cy) {
    if (cy % 16 == 0)
      info(fmt("multiplying matrix progress=%g", (double)cy / (double)ch));
    for (int cx = 0; cx < cw; ++cx) {
      c[cx + cw * cy] = 0;
      for (int i = 0; i < aw; ++i) {
        const int ax = i;
        const int ay = cy;
        const int bx = cx;
        const int by = i;
        c[cx + cw * cy] += a[ax + aw * ay] * b[bx + bw * by];
      }
    }
  }
  info(fmt("multiplying matrix progress=%g", 1));
}

static void cpumatvec(const double *a, int aw, int ah, const double *b, double *c) {
  cpumatmat(a, aw, ah, b, 1, aw, c);
}

static void cpuchol(double *m, unsigned int dim) {
  unsigned int i, j, k; 

  for (k = 0; k < dim; k++) {
    if (m[k * dim + k] < 0)
      m[k * dim + k] = 0;

    m[k * dim + k] = sqrt(m[k * dim + k]);
    
    if (m[k * dim + k] > 0) {
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] /= m[k * dim + k];
    } else {
      m[k * dim + k] = 1.0;
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] = 0;
    }
             
    for (i = k + 1; i < dim; i++)
      for (j = i; j < dim; j++)
        m[i * dim + j] -= m[k * dim + i] * m[k * dim + j];

  }

  for (i = 0; i < dim; i++)
    for (j = 0; j < i; j++)
      m[i * dim + j] = 0.0;
}

void matinv(const double *m, double *x, unsigned int n) {
  memset(x, 0, n * n * sizeof(double));

  for (unsigned int k = 0; k < n; ++k) {
    x[k * n + k] = 1.0 / m[k * n + k];

    for (unsigned int i = k + 1; i < n; ++i) {
      x[k * n + i] = 0;
      double a = 0, b = 0;
      for (unsigned int j = k; j < i; ++j) {
        double a = -m[j * n + i];
        double b = x[k * n + j];
        double c = m[i * n + i];
        x[k * n + i] += a * b / c;
      }
    }
  }
}

void normalize(
  Rando *rvg,
  Cortex *enc,
  Cortex *gen
) {
  int oc = enc->oc;
  assert(rvg->dim == oc);

  double *mean = new double[oc];
  double *chol = new double[oc * oc];
  double *unchol = new double[oc * oc];
  memcpy(mean, rvg->mean, oc * sizeof(double));
  memcpy(chol, rvg->chol, oc * oc * sizeof(double));
  matinv(chol, unchol, oc);
  cpumatxpose(chol, oc);
  cpumatxpose(unchol, oc);

  float *fem, *feb;
  int ew, eh;
  enc->_get_head_op(&feb, &fem, &ew, &eh);
  assert(ew > 0);
  assert(eh == oc);

  float *fgm, *fgb;
  int gw, gh;
  gen->_get_tail_op(&fgb, &fgm, &gw, &gh);
  assert(gw == oc);
  assert(gh > 0);

  if (eh != gw)
    error("matrices don't match, gen needs to start with con0 to normalize");

  // work in double on the host, store back as float
  double *em = new double[ew * eh], *eb = new double[eh];
  double *gm = new double[gw * gh], *gb = new double[gh];
  for (int i = 0; i < ew * eh; ++i) em[i] = fem[i];
  for (int i = 0; i < eh; ++i) eb[i] = feb[i];
  for (int i = 0; i < gw * gh; ++i) gm[i] = fgm[i];
  for (int i = 0; i < gh; ++i) gb[i] = fgb[i];

  for (int z = 0; z < oc; ++z)
    eb[z] -= mean[z];

  for (int y = 0; y < gh; ++y) {
    double q = 0;
    for (int z = 0; z < oc; ++z)
      q += gm[z + y * gw] * mean[z];
    gb[y] += q;
  }

  double *tem = new double[ew * eh];
  double *teb = new double[eh];
  double *tgm = new double[gw * gh];

  cpumatmat(unchol, oc, oc, em, ew, eh, tem);
  memcpy(em, tem, sizeof(double) * ew * eh);
  cpumatvec(unchol, oc, oc, eb, teb);
  memcpy(eb, teb, sizeof(double) * eh);
  cpumatmat(gm, gw, gh, chol, oc, oc, tgm);
  memcpy(gm, tgm, sizeof(double) * gw * gh);

  for (int i = 0; i < ew * eh; ++i) fem[i] = (float)em[i];
  for (int i = 0; i < eh; ++i) feb[i] = (float)eb[i];
  for (int i = 0; i < gw * gh; ++i) fgm[i] = (float)gm[i];
  for (int i = 0; i < gh; ++i) fgb[i] = (float)gb[i];

  gen->_put_tail_op(fgb, fgm, gw, gh);
  enc->_put_head_op(feb, fem, ew, eh);

  delete[] mean;
  delete[] chol;
  delete[] unchol;
  delete[] teb;
  delete[] tem;
  delete[] tgm;
  delete[] em;
  delete[] gm;
  delete[] eb;
  delete[] gb;
  delete[] fem;
  delete[] fgm;
  delete[] feb;
  delete[] fgb;
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  ++argv;
  --argc;

  if (argc < 2) {
    usage();
    return 1;
  }

  std::string cmd = argv[0];
  ++argv;
  --argc;

  if (cmd == "help" || cmd == "h") {
    usage();
    return 0;
  }

  std::vector<std::string> ctx;
  while (argc > 0 && **argv != '-' && !strchr(*argv, '=')) {
    ctx.push_back(*argv);
    ++argv;
    --argc;
  }

  if (ctx.empty()) {
    usage();
    return 1;
  }

  Cmdline arg(argc, argv, "0");
  verbose = arg.get("verbose", "0");
  info(fmt("verbose=%d", verbose));

  Kleption::set_picreader_cmd(arg.get("picreader", "/opt/makemore/share/tewel/picreader.pl"));
  Kleption::set_picwriter_cmd(arg.get("picwriter", "/opt/makemore/share/tewel/picwriter.pl"));
  Kleption::set_vidreader_cmd(arg.get("vidreader", "/opt/makemore/share/tewel/vidreader.pl"));
  Kleption::set_vidwriter_cmd(arg.get("vidwriter", "/opt/makemore/share/tewel/vidwriter.pl"));

  if (arg.present("randseed"))
    seedrand(strtoul(arg.get("randseed")));
  else
    seedrand();
  uint32_t firstrand = randuint();
  info(fmt("first random uint %u", firstrand));

  int lowmem = arg.get("lowmem", "0");
  int cuda = arg.get("cuda", "-1");
  int kbs = arg.get("kbs", "256");
  int prec = arg.get("prec", "32");
  if (prec != 16 && prec != 32)
    error("prec must be 16 or 32");
  setkprec(prec);

  if (cmd == "server") {
    double reload = arg.get("reload", "0.0");
    bool pngout = (int)arg.get("pngout", "0");

    int pw = 0, ph = 0;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &pw, &ph))
        error("dim must be like 256 or like 256x256");
    }

    int kids = arg.get("kids", "1");

    int port = arg.get("port", "4444");
    info(fmt("starting server on port %d", port));

    EGServer *server = new EGServer(ctx, pw, ph, cuda, kbs, reload, pngout);

    server->open();
    server->bind((uint16_t)port);
    server->listen(1);
    server->start(kids);
    server->wait();
    info("all done");

    delete server;
    return 0;
  }

  if (cmd == "tileserver") {
    int kids = arg.get("kids", "1");

    int port = arg.get("port", "9999");
    info(fmt("starting server on port %d", port));

    if (!arg.present("dir"))
      error("argument dir required");
    std::string dir = arg.get("dir");

    if (!arg.present("root"))
      error("argument root required");
    std::string root = arg.get("root");

    TileServer *server = new TileServer(root, dir, ctx, cuda, kbs);

    server->open();
    server->bind((uint16_t)port);
    server->listen(1);
    server->start(kids);
    server->wait();
    info("all done");

    delete server;
    return 0;
  }

  setkdev(cuda >= 0 ? cuda : kndevs() > 1 ? 1 : 0);
  setkbs(kbs);

  setkdev(cuda >= 0 ? cuda : kndevs() > 1 ? 1 : 0);
  setkbs(kbs);

  if (cmd == "make") {
    if (ctx.size() != 1)
      error("make requires exactly one ctx file arg");

    std::string spec = arg.get("spec");
    if (startswith(spec, "//"))
      spec = "/opt/makemore/share/tewel/" + std::string(spec.c_str() + 2);

    int clobber = arg.get("clobber", "0");

    if (!arg.unused.empty())
      error("unrecognized options");

    Cortex::create(ctx[0], (bool)clobber);

    Cortex *out = new Cortex(ctx[0]);

    FILE *specfp = fopen(spec.c_str(), "r");
    if (!specfp)
      error(str("can't open ") + spec + ": " + strerror(errno));

    std::string specline;
    std::string init = "gauss";
    int eqlr = 0;
    bool got_header = false;
    while (read_line(specfp, &specline)) {
      strmap specopt;

      specopt["decay"] = "1e-2";
      specopt["nu"] = "1e-4";
      specopt["b1"] = "1e-1";
      specopt["b2"] = "1e-3";
      specopt["eps"] = "1e-8";
      specopt["clip"] = "0";
      specopt["rdev"] = "0.1";
      specopt["init"] = "gauss";
      specopt["eqlr"] = "0";

      int keys = parsekv(specline, &specopt);
      if (keys < 0)
        error(str("can't parse kv in header of ") + spec);
      if (keys == 0)
        continue;

      if (specopt["v"] != "2")
        error(str("missing magic v=2 in ") + spec);

      out->head->decay = strtod(specopt["decay"]);
      out->head->nu = strtod(specopt["nu"]);
      out->head->b1 = strtod(specopt["b1"]);
      out->head->b2 = strtod(specopt["b2"]);
      out->head->eps = strtod(specopt["eps"]);
      out->head->clip = strtod(specopt["clip"]);
      out->head->rdev = strtod(specopt["rdev"]);
      init = specopt["init"];
      eqlr = strtoi(specopt["eqlr"]);

      got_header = true;
      break;
    }
    if (!got_header)
      error("no header line in spec");

    int poc = 0;
    while (read_line(specfp, &specline)) {
      strmap specopt;

      specopt["iw"] = "0";
      specopt["ih"] = "0";
      specopt["ic"] = "0";
      specopt["ow"] = "0";
      specopt["oh"] = "0";
      specopt["oc"] = "0";
      specopt["mul"] = "";
      specopt["freeze"] = "0";
      specopt["rad"] = "0";
      specopt["relu"] = "0";
      specopt["dim"] = "2";
      specopt["sdim"] = "0";
      specopt["slot"] = "";

      int keys = parsekv(specline, &specopt);
      if (keys < 0)
        error(str("can't parse kv in line of ") + spec);
      if (keys == 0)
        continue;

      std::string type = specopt["type"];
      if (specopt["mul"] == "")
        specopt["mul"] = (type == "lrlu") ? "0.2" : "1.0";
      int eqlr_nin = 0;
      if (eqlr && type.length() == 4 && (type.substr(0, 3) == "con" || type.substr(0, 3) == "rcn") && isdigit(type[3])) {
        int cic = strtoi(specopt["ic"]); if (cic == 0) cic = poc;
        int f = 2 * (type[3] - '0') + 1;
        eqlr_nin = (strtoi(specopt["dim"]) == 1) ? cic * f : cic * f * f;
      }
      int iw = strtoi(specopt["iw"]);
      int ih = strtoi(specopt["ih"]);
      int ic = strtoi(specopt["ic"]);
      int ow = strtoi(specopt["ow"]);
      int oh = strtoi(specopt["oh"]);
      int oc = strtoi(specopt["oc"]);
      double mul = strtod(specopt["mul"]);
      int freeze = strtoi(specopt["freeze"]);
      int rad = strtoi(specopt["rad"]);
      int relu = strtoi(specopt["relu"]);
      if (eqlr_nin > 0) {
        mul *= sqrt(2.0 / (double)eqlr_nin);
        relu |= 2;
      }
      if (type == "styl") {
        rad = strtoi(specopt["sdim"]);
        if (rad <= 0)
          error("styl requires sdim");
        if (eqlr) {
          mul *= sqrt(1.0 / (double)rad);
          relu |= 2;
        }
      }
      if (type == "mcon") {
        rad = strtoi(specopt["sdim"]);
        if (rad <= 0)
          error("mcon requires sdim");
        if (eqlr) {
          mul *= sqrt(2.0 / (9.0 * (double)(ic ? ic : poc)));
          relu |= 2;
        }
      }
      if (type == "keep" || type == "rcal" || type == "radd") {
        if (specopt["slot"] == "")
          error(type + " requires slot");
        rad = strtoi(specopt["slot"]);
        if (rad < 0 || rad >= 16)
          error("slot must be 0..15");
      }
      int dim = strtoi(specopt["dim"]);

      if (ic == 0)
        ic = poc;
      if (ic == 0)
        error("input channels required");
      if (poc && ic != poc)
        error("input channels don't match output channels of previous layer");

      out->push(type, ic, oc, iw, ih, ow, oh, mul, rad, freeze, relu, dim);

      poc = out->oc;
    }
    fclose(specfp);

    out->scram(out->head->rdev, init == "he");
    delete out;

    return 0;
  }

  if (cmd == "identity") {
    if (ctx.size() != 1)
      error("identity requires exactly one ctx file arg");

    int clobber = arg.get("clobber", "0");

    if (!arg.present("dim"))
      error("option dim required");
    int dim = arg.get("dim");

    if (!arg.unused.empty())
      error("unrecognized options");

    Cortex::create(ctx[0], (bool)clobber);

    Cortex *out = new Cortex(ctx[0]);

    out->head->decay = 0;
    out->head->nu = 0;
    out->head->b1 = 0;
    out->head->b2 = 0;
    out->head->eps = 0;
    out->head->clip = 0;
    out->head->rdev = 0;

    out->push("con0", dim, dim);

    float *m, *b;
    int w, h;
    out->_get_head_op(&b, &m, &w, &h);
    assert(w == dim);
    assert(h == dim);

    memset(b, 0, sizeof(float) * dim);

    for (int y = 0; y < dim; ++y)
      for (int x = 0; x < dim; ++x)
        m[x + dim * y] = (x == y) ? 1.0 : 0.0;

    out->_put_head_op(b, m, w, h);

    delete[] m;
    delete[] b;
    delete out;

    return 0;
  }

  if (cmd == "edit") {
    if (ctx.size() != 1)
      error("edit requires exactly one ctx file arg");

    Cortex *gen = new Cortex(ctx[0]);

    if (arg.present("decay"))
      gen->head->decay = arg.get("decay");
    if (arg.present("nu"))
      gen->head->nu = arg.get("nu");
    if (arg.present("b1"))
      gen->head->b1 = arg.get("b1");
    if (arg.present("b2"))
      gen->head->b2 = arg.get("b2");
    if (arg.present("eps"))
      gen->head->eps = arg.get("eps");
    if (arg.present("clip"))
      gen->head->clip = arg.get("clip");
    if (arg.present("rdev"))
      gen->head->rdev = arg.get("rdev");

    if (!arg.unused.empty())
      warning("unrecognized options");

    delete gen;
    gen = new Cortex(ctx[0]);

    printf(
      "rounds=%lu rms=%g max=%g decay=%g nu=%g b1=%g b2=%g eps=%g clip=%g rdev=%g\n",
      gen->rounds, gen->rms, gen->max, gen->decay,
      gen->nu, gen->b1, gen->b2, gen->eps, gen->clip, gen->head->rdev
    );

    return 0;
  }

  if (cmd == "unfreeze") {
    if (ctx.size() != 1)
      error("unfreeze requires exactly one ctx file arg");

    int last = arg.get("last");

    if (!arg.unused.empty())
      warning("unrecognized options");

    Cortex *gen = new Cortex(ctx[0], O_RDWR);

    gen->unfreeze(last);

    gen->save();

    delete gen;
    return 0;
  }

  if (cmd == "spec") {
    if (ctx.size() != 1)
      error("spec requires exactly one ctx file arg");
    if (!arg.unused.empty())
      warning("unrecognized options");
    Cortex *gen = new Cortex(ctx[0], O_RDONLY);

    printf(
      "v=2 decay=%g nu=%g b1=%g b2=%g eps=%g clip=%g rdev=%g\n",
      gen->decay, gen->nu, gen->b1, gen->b2, gen->eps, gen->clip, gen->head->rdev
    );

    gen->dump(stdout);

    delete gen;
    return 0;
  }

  if (cmd == "bindump") {
    if (ctx.size() != 1)
      error("bindump requires exactly one ctx file arg");
    unsigned int a = 0, b = (1 << 31);
    if (arg.present("cut"))
      if (!parserange(arg.get("cut"), &a, &b))
        error("cut must be number or number range");

    if (!arg.unused.empty())
      warning("unrecognized options");

    Cortex *gen = new Cortex(ctx[0], O_RDONLY);
    gen->bindump(stdout, a, b);
    delete gen;
    return 0;
  }

  if (cmd == "reset") {
    if (ctx.size() != 1)
      error("reset requires exactly one ctx file arg");
    Cortex *gen = new Cortex(ctx[0]);

    gen->head->rounds = 0;
    gen->head->rms = 0;
    gen->head->max = 0;

    if (arg.present("rdev"))
      gen->head->rdev = arg.get("rdev");

    if (!arg.unused.empty())
      warning("unrecognized options");

    double rdev = gen->head->rdev;
    gen->scram(rdev, std::string(arg.get("init", "gauss")) == "he");

    delete gen;
    return 0;
  }

  if (cmd == "glyphs") {
    // tewel glyphs map.ctx gen.ctx ref=ref96.u8 zdat=z.dat out=sheets.rgb [refdim=32x48x1]
    // decodes the 96 reference glyph cells with one style per latent (f64le rows of the mapping
    // net's input size) and tiles each font into a 12x8 sheet, written as raw u8 rgb
    if (ctx.size() != 2)
      error("glyphs needs map.ctx gen.ctx");
    int cw, chh, cc;
    if (!parsedim(arg.get("refdim", "32x48x1"), &cw, &chh, &cc))
      error("bad refdim");
    std::string reffn = arg.get("ref"), zfn = arg.get("zdat"), outfn = arg.get("out");
    Cortex *map = new Cortex(ctx[0], O_RDONLY);
    map->prepare(1, 1, 1);
    Cortex *gen = new Cortex(ctx[1], O_RDONLY);
    gen->prepare(cw, chh, 96);
    if (gen->sdim != map->oc)
      error("gen style dim doesn't match map output");
    if (gen->ic != cc)
      error("gen input channels don't match refdim");
    int sdim = gen->sdim, zn = map->iwhc, goc = gen->oc;
    long refn = 96L * chh * cw * cc;
    std::vector<uint8_t> refb(refn);
    { FILE *fp = fopen(reffn.c_str(), "r"); if (!fp || fread(refb.data(), 1, refn, fp) != (size_t)refn) error("can't read ref"); fclose(fp); }
    std::vector<float> reff(refn);
    for (long i = 0; i < refn; ++i) reff[i] = refb[i] / 255.0f;
    float *kref; kmake(&kref, refn); enk(reff.data(), refn, kref);
    std::vector<double> zd;
    { FILE *fp = fopen(zfn.c_str(), "r"); if (!fp) error("can't read zdat"); double v; while (fread(&v, 8, 1, fp) == 1) zd.push_back(v); fclose(fp); }
    long n = zd.size() / zn;
    float *kz; kmake(&kz, zn);
    float *kw; kmake(&kw, 96L * sdim);
    std::vector<float> zf(zn), outf(96L * chh * cw * goc);
    int W = 12 * cw, H = 8 * chh;
    std::vector<uint8_t> sheet((size_t)W * H * 3);
    FILE *fo = fopen(outfn.c_str(), "w"); if (!fo) error("can't write out");
    for (long i = 0; i < n; ++i) {
      for (int j = 0; j < zn; ++j) zf[j] = (float)zd[i * zn + j];
      enk(zf.data(), zn, kz);
      Cortex::set_noise_seed(Cortex::hash_input(zf.data(), zn));
      float *w = map->synth(kz);
      for (int b = 0; b < 96; ++b) kcopy(w, sdim, kw + (long)b * sdim);
      gen->kstyle = kw;
      float *o = gen->synth(kref);
      dek(o, outf.size(), outf.data());
      for (int b = 0; b < 96; ++b) {
        int col = b % 12, row = b / 12;
        for (int y = 0; y < chh; ++y)
          for (int x = 0; x < cw; ++x) {
            float v = outf[(((long)b * chh + y) * cw + x) * goc];
            int u = (int)lrintf(v * 255.0f); if (u < 0) u = 0; if (u > 255) u = 255;
            uint8_t *p = &sheet[(((size_t)(row * chh + y)) * W + col * cw + x) * 3];
            p[0] = p[1] = p[2] = (uint8_t)u;
          }
      }
      fwrite(sheet.data(), 1, sheet.size(), fo);
    }
    fclose(fo);
    return 0;
  }

  if (cmd == "synth") {
    Chain *chn = new Chain;

    if (arg.present("rmul"))
      chn->rmul = strtod(arg.get("rmul"));

    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDONLY);

    int limit = arg.get("limit", "-1");
    int reload = arg.get("reload", "0");
    double delay = arg.get("delay", "0.0");

    Kleption::Trav srctrav = Kleption::get_trav(arg.get("srctrav", "scan"));
    if (srctrav == Kleption::TRAV_NONE)
      error("srctrav must be scan or rand or refs");

    Kleption::Flags srcflags = 0;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;

    int repeat = arg.get("repeat", "0");
    if (repeat)
      srcflags |= Kleption::FLAG_REPEAT;

    double trim = arg.get("trim", "0.0");
    std::string crop = arg.get("crop", "center");
    if (crop == "center")
      srcflags |= Kleption::FLAG_CENTER;
    else if (crop != "random")
      error("crop must be random or center");

    int pw = 0, ph = 0;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &pw, &ph))
        error("dim must be like 256 or like 256x256");
    }
    int pc = chn->head->ic;

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    std::string srcrefs = arg.get("srcrefs", "");
    const char *refsfn = NULL;
    if (srcrefs != "") {
      if (srctrav != Kleption::TRAV_REFS)
        error("srcrefs given without srctrav=refs");
      refsfn = srcrefs.c_str();
    }

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    double evolve = arg.get("evolve", "0.0");
    double rvgmul = arg.get("rvgmul", "1.0");

    std::string srcfn = arg.get("src");

    Kleption *src = new Kleption(
      srcfn,
      pw, ph, pc,
      srcflags, srctrav, srckind,
      sw, sh, sc,
      refsfn, evolve, rvgmul, trim
    );

    
   

    src->load();
    assert(src->pw > 0);
    assert(src->ph > 0);

    chn->prepare(src->pw, src->ph);
    int ic = chn->head->ic;
    assert(ic == src->pc);
    
    Kleption::Kind outkind = Kleption::get_kind(arg.get("outkind", ""));
    if (outkind == Kleption::KIND_UNK)
      error("unknown outkind");
    if (outkind != Kleption::KIND_SDL)
      arg.get("out");

    Kleption::Flags outflags = Kleption::FLAG_WRITER;
    if (lowmem)
      outflags |= Kleption::FLAG_LOWMEM;

    int append = strtoi(arg.get("append", "0"));
    if (append)
      outflags |= Kleption::FLAG_APPEND;

    Kleption *out = new Kleption(
      arg.get("out", ""),
      chn->tail->ow, chn->tail->oh, chn->tail->oc,
      outflags, Kleption::TRAV_SCAN, outkind
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    int i = 0;
    while (1) {
      if (limit >= 0 && i >= limit)
        break;

      std::string id;
      if (!src->pick(chn->kinp(), &id))
        break;
      chn->synth();

      if (!out->place(id, chn->kout()))
        break;

      if (delay > 0)
        usleep(delay * 1000000.0);
      if (reload)
        chn->load();
      ++i;
    }


    delete chn;
    delete src;
    delete out;
    return 0;
  }

  if (cmd == "normalize") {
    if (ctx.size() != 2)
      error("normalize requires exactly two ctx file args");
    std::string encfn = ctx[0];
    std::string genfn = ctx[1];

    if (!arg.present("rvg"))
      error("normalize requires rvg arg");
    std::string rvgfn = arg.get("rvg");

    int nerf = arg.get("nerf", "1");
    int clobber = arg.get("clobber", "0");

    if (!arg.unused.empty())
      error("unrecognized options");

    Rando *rvg = new Rando;
    rvg->load(rvgfn);
    int dim = rvg->dim;

    Cortex::create(encfn, (bool)clobber);
    Cortex::create(genfn, (bool)clobber);

    Cortex *enc = new Cortex(encfn, O_RDWR);
    enc->push("con0", dim, dim);
    if (nerf)
      enc->push("nerf", dim, dim);

    {
      float *m, *b;
      int w, h;
      enc->_get_head_op(&b, &m, &w, &h);
      assert(w == dim);
      assert(h == dim);

      memset(b, 0, sizeof(float) * dim);

      for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x)
          m[x + dim * y] = (x == y) ? 1.0 : 0.0;

      enc->_put_head_op(b, m, w, h);

      delete[] m;
      delete[] b;
    }

    delete enc;
    enc = new Cortex(encfn, O_RDWR);
    enc->prepare(1, 1);

    Cortex *gen = new Cortex(genfn, O_RDWR);
    if (nerf)
      gen->push("inrf", dim, dim);
    gen->push("con0", dim, dim);

    {
      float *m, *b;
      int w, h;
      gen->_get_tail_op(&b, &m, &w, &h);
      assert(w == dim);
      assert(h == dim);

      memset(b, 0, sizeof(float) * dim);

      for (int y = 0; y < dim; ++y)
        for (int x = 0; x < dim; ++x)
          m[x + dim * y] = (x == y) ? 1.0 : 0.0;

      gen->_put_tail_op(b, m, w, h);

      delete[] m;
      delete[] b;
    }

    delete gen;
    gen = new Cortex(genfn, O_RDWR);
    gen->prepare(1, 1);

    normalize(rvg, enc, gen);

    delete gen;
    delete enc;
    delete rvg;
    return 0;
  }


  std::map<std::string, std::string> diropt;

  diropt["repint"] = "100";
  diropt["reps"] = "-1";
  diropt["mul"] = "1.0";
  diropt["lossreg"] = "1";
  diropt["srcdim"] = "0x0x0";
  diropt["tgtdim"] = "0x0x0";
  diropt["stydim"] = "0x0x0";
  diropt["srckind"] = "";
  diropt["tgtkind"] = "";
  diropt["stykind"] = "";
  
  if (cmd == "learnfunc") {
    Chain *chn = new Chain;
    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDWR);

    int iw, ih;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &iw, &ih))
        error("dim must be like 256 or like 256x256");
    } else if (diropt.count("dim")) {
      std::string dim = diropt["dim"];
      if (!parsedim2(dim, &iw, &ih))
        error("dim must be like 256 or like 256x256");
    }

    int repint = arg.get("repint", diropt["repint"]);
    double mul = arg.get("mul", diropt["mul"]);
    int batch = arg.get("batch", "1");

    int ic;
    chn->prepare(iw, ih, batch);
    ic = chn->head->ic;

    std::string srcdim = arg.get("srcdim", diropt["srcdim"]);
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", diropt["srckind"]));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (diropt.count("src")) {
      srcfn = diropt["src"];
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption::Flags srcflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      srcflags, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string tgtdim = arg.get("tgtdim", diropt["tgtdim"]);
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", diropt["tgtkind"]));

    std::string tgtfn;
    if (arg.present("tgt")) {
      tgtfn = (std::string)arg.get("tgt");
    } else if (diropt.count("tgt")) {
      tgtfn = diropt["tgt"];
    } else {
      arg.get("tgt");
      assert(0);
    }

    Kleption::Flags tgtflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      tgtflags |= Kleption::FLAG_LOWMEM;

    Kleption *tgt = new Kleption(
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));
    double stoploss = arg.get("stoploss", "0.0");
    long stoprounds = strtol(arg.get("stoprounds", "-1"));
    std::string loss = arg.get("loss", "l2");
    if (loss != "l1" && loss != "l2")
      error("loss must be l1 or l2");
    double lossdelta = arg.get("lossdelta", "0.01");
    double latnoise = arg.get("latnoise", "0.0");
    double latreg = arg.get("latreg", "0.0");

    if (!arg.unused.empty())
      error("unrecognized options");

    learnfunc(src, tgt, chn, repint, mul, reps, stoploss, stoprounds, loss == "l1", lossdelta, latnoise, latreg);

    delete src;
    delete tgt;
    delete chn;
    return 0;
  }

  if (cmd == "learnhans") {
    Chain *chn = new Chain;
    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDWR);

    int iw, ih;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &iw, &ih))
        error("dim must be like 256 or like 256x256");
    } else if (diropt.count("dim")) {
      std::string dim = diropt["dim"];
      if (!parsedim2(dim, &iw, &ih))
        error("dim must be like 256 or like 256x256");
    }

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", diropt["repint"]);
    double mul = arg.get("mul", diropt["mul"]);
    int batch = arg.get("batch", "1");

    int ic;
    chn->prepare(iw, ih, batch);
    ic = chn->head->ic;

    Cortex *dis = NULL;
    if (arg.present("dis")) {
      dis = new Cortex(arg.get("dis"));
    } else {
      error("dis argument required");
    }

    dis->prepare(chn->tail->ow, chn->tail->oh, batch);

    Cortex *cnd = NULL;
    if (arg.present("cnd")) {
      cnd = new Cortex(arg.get("cnd"));

      cnd->prepare(chn->head->iw, chn->head->ih, batch);

      if (cnd->ow != dis->iw)
        error("cnd ow doesn't match dis iw");
      if (cnd->oh != dis->ih)
        error("cnd oh doesn't match dis ih");

      if (dis->ic != chn->tail->oc + cnd->oc)
        error("gen oc + cnd oc doesn't match dis ic");
    } else {
      if (dis->iw != chn->tail->ow)
        error("gen ow doesn't match dis iw");
      if (dis->ih != chn->tail->oh)
        error("gen oh doesn't match dis ih");

      if (dis->ic != chn->tail->oc)
        error("gen oc + gen ic doesn't match dis ic");
    }

    std::string srcdim = arg.get("srcdim", diropt["srcdim"]);
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", diropt["srckind"]));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (diropt.count("src")) {
      srcfn = diropt["src"];
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption::Flags srcflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      srcflags, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string tgtdim = arg.get("tgtdim", diropt["tgtdim"]);
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", diropt["tgtkind"]));

    std::string tgtfn;
    if (arg.present("tgt")) {
      tgtfn = (std::string)arg.get("tgt");
    } else if (diropt.count("tgt")) {
      tgtfn = diropt["tgt"];
    } else {
      arg.get("tgt");
      assert(0);
    }

    Kleption::Flags tgtflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      tgtflags |= Kleption::FLAG_LOWMEM;

    Kleption *tgt = new Kleption(
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));
    long stoprounds = strtol(arg.get("stoprounds", "-1"));
    long disreps = strtol(arg.get("disreps", "1"));
    double genmul = strtod(arg.get("genmul", "1.0"));
    double dismul = strtod(arg.get("dismul", "1.0"));
    double noise = strtod(arg.get("noise", "0.0"));
    double tether = arg.get("tether", "0.0");
    std::string loss = arg.get("loss", "l2");
    if (loss != "l1" && loss != "l2")
      error("loss must be l1 or l2");
    double lossdelta = arg.get("lossdelta", "0.01");
    double latnoise = arg.get("latnoise", "0.0");
    double latreg = arg.get("latreg", "0.0");

    if (!arg.unused.empty())
      error("unrecognized options");

    info(fmt("dim=%dx%d reps=%ld batch=%d disreps=%ld tether=%g", iw, ih, reps, batch, disreps, tether));

    learnhans(src, tgt, chn, dis, cnd, repint, mul, reps, disreps, stoprounds, genmul, dismul, noise, tether, loss == "l1", lossdelta, latnoise, latreg);

    delete src;
    delete tgt;
    delete chn;
    delete dis;
    return 0;
  }

  if (cmd == "learngan") {
    Chain *chn = new Chain;
    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDWR);

    int iw = 1, ih = 1;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &iw, &ih))
        error("dim must be like 1 or like 1x1");
    }

    int repint = arg.get("repint", diropt["repint"]);
    double mul = arg.get("mul", diropt["mul"]);
    int batch = arg.get("batch", "1");

    // optional condition paired with the targets (same ids): the generator's tail input,
    // and extra channels of the discriminator input
    Kleption *cnd = NULL;
    int cw = 0, chh = 0, cc = 0;
    if (arg.present("cnd")) {
      if (!parsedim(arg.get("cnddim", ""), &cw, &chh, &cc) || cw <= 0 || chh <= 0 || cc <= 0)
        error("cnd requires cnddim like 32x48x1");
      Kleption::Kind cndkind = Kleption::get_kind(arg.get("cndkind", "u8"));
      Kleption::Flags cndflags = Kleption::FLAG_REPEAT;
      if (lowmem)
        cndflags |= Kleption::FLAG_LOWMEM;
      cnd = new Kleption((std::string)arg.get("cnd"), cw, chh, cc, cndflags, Kleption::TRAV_RAND, cndkind, cw, chh, cc);
    }

    chn->prepare(iw, ih, batch, cw, chh);

    if (!arg.present("dis"))
      error("dis argument required");
    Cortex *dis = new Cortex(arg.get("dis"));
    dis->prepare(chn->tail->ow, chn->tail->oh, batch);

    if (dis->iw != chn->tail->ow || dis->ih != chn->tail->oh)
      error("gen output dims don't match dis input dims");
    if (dis->ic != chn->tail->oc + cc)
      error("gen oc (plus cnd channels) doesn't match dis ic");
    if (cnd && (cw != chn->tail->ow || chh != chn->tail->oh))
      error("cnddim doesn't match gen output dims");

    Cortex *ema = NULL;
    if (arg.present("ema")) {
      if (chn->tail->ctxv.size() != 1)
        error("ema requires a single-ctx generator");
      std::string emafn = arg.get("ema");
      if (!fexists(emafn)) {
        info("creating " + emafn + " as a copy of the generator");
        std::string genfn = chn->tail->fn();
        FILE *fi = fopen(genfn.c_str(), "r"); if (!fi) error("can't open " + genfn);
        FILE *fo = fopen(emafn.c_str(), "w"); if (!fo) error("can't create " + emafn);
        char buf[65536]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fi)) > 0) fwrite(buf, 1, n, fo);
        fclose(fi); fclose(fo);
      }
      ema = new Cortex(emafn);
      ema->prepare(chn->tail->ctxv[0]->iw, chn->tail->ctxv[0]->ih, batch);
    }

    std::string tgtdim = arg.get("tgtdim", diropt["tgtdim"]);
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");
    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", diropt["tgtkind"]));
    if (!arg.present("tgt"))
      error("tgt argument required");
    std::string tgtfn = (std::string)arg.get("tgt");
    Kleption::Flags tgtflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      tgtflags |= Kleption::FLAG_LOWMEM;
    if ((int)arg.get("flip", "0"))
      tgtflags |= Kleption::FLAG_FLIP;
    Kleption *tgt = new Kleption(
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));
    long stoprounds = strtol(arg.get("stoprounds", "-1"));
    long disreps = strtol(arg.get("disreps", "1"));
    double genmul = strtod(arg.get("genmul", "1.0"));
    double dismul = strtod(arg.get("dismul", "1.0"));
    std::string loss = arg.get("loss", "ns");
    if (loss != "ns" && loss != "ls")
      error("loss must be ns or ls");
    double r1 = arg.get("r1", "0.0");
    int r1int = arg.get("r1int", "16");
    double r1eps = arg.get("r1eps", "1e-3");
    double emabeta = arg.get("emabeta", "0.999");
    int replay = arg.get("replay", "0");
    double replayfrac = arg.get("replayfrac", "0.5");
    double disemabeta = arg.get("disemabeta", "0.99");

    Cortex *disema = NULL;
    if (arg.present("disema")) {
      std::string fn2 = arg.get("disema");
      if (!fexists(fn2)) {
        info("creating " + fn2 + " as a copy of the discriminator");
        std::string dfn = arg.get("dis");
        FILE *fi = fopen(dfn.c_str(), "r"); if (!fi) error("can't open " + dfn);
        FILE *fo = fopen(fn2.c_str(), "w"); if (!fo) error("can't create " + fn2);
        char buf[65536]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fi)) > 0) fwrite(buf, 1, n, fo);
        fclose(fi); fclose(fo);
      }
      disema = new Cortex(fn2);
      disema->prepare(chn->tail->ow, chn->tail->oh, batch);
    }

    if (!arg.unused.empty())
      error("unrecognized options");

    info(fmt("learngan batch=%d disreps=%ld loss=%s r1=%g r1int=%d replay=%d disema=%d cnd=%d", batch, disreps, loss.c_str(), r1, r1int, replay, disema ? 1 : 0, cc));

    learngan(tgt, chn, dis, ema, repint, mul, reps, disreps, stoprounds, genmul, dismul, loss == "ns", r1, r1int, r1eps, emabeta, disema, disemabeta, replay, replayfrac, cnd, cc);

    delete tgt;
    delete chn;
    delete dis;
    if (ema) delete ema;
    if (disema) delete disema;
    return 0;
  }

  if (cmd == "learn") {
    int iw, ih;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &iw, &ih))
        error("dim must be like 256 or like 256x256");
    } else if (diropt.count("dim")) {
      std::string dim = diropt["dim"];
      if (!parsedim2(dim, &iw, &ih))
        error("dim must be like 256 or like 256x256");
    }

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", diropt["repint"]);
    double mul = arg.get("mul", diropt["mul"]);
    bool lossreg = strtoi(arg.get("lossreg", "0"));
    double losspin = strtod(arg.get("losspin", "0.0"));

    if (!arg.present("enc"))
      error("enc argument required");
    Cortex *enc = new Cortex(arg.get("enc"));
    int ic = enc->ic;
    enc->prepare(iw, ih);

    if (!arg.present("inf"))
      error("inf argument required");
    Cortex *inf = new Cortex(arg.get("inf"));

    inf->prepare(iw, ih);

    Chain *chn = new Chain;
    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDWR);
    chn->prepare(enc->ow, enc->oh);

    if (!arg.present("dis"))
      error("dis argument required");
    Cortex *dis = new Cortex(arg.get("dis"));
    dis->prepare(chn->tail->ow, chn->tail->oh);

    if (!arg.present("cnd"))
      error("cnd argument required");
    Cortex *cnd = new Cortex(arg.get("cnd"));
    cnd->prepare(inf->ow, inf->oh);

    if (cnd->ow != dis->iw)
      error("cnd ow doesn't match dis iw");
    if (cnd->oh != dis->ih)
      error("cnd oh doesn't match dis ih");
    if (dis->ic != chn->tail->oc + cnd->oc)
      error("gen oc + cnd oc doesn't match dis ic");



    std::string srcdim = arg.get("srcdim", diropt["srcdim"]);
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", diropt["srckind"]));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (diropt.count("src")) {
      srcfn = diropt["src"];
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption::Flags srcflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      srcflags, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );



    std::string tgtdim = arg.get("tgtdim", diropt["tgtdim"]);
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", diropt["tgtkind"]));

    std::string tgtfn;
    if (arg.present("tgt")) {
      tgtfn = (std::string)arg.get("tgt");
    } else if (diropt.count("tgt")) {
      tgtfn = diropt["tgt"];
    } else {
      arg.get("tgt");
      assert(0);
    }

    Kleption::Flags tgtflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      tgtflags |= Kleption::FLAG_LOWMEM;

    Kleption *tgt = new Kleption(
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );



    long reps = strtol(arg.get("reps", diropt["reps"]));
    long stoprounds = strtol(arg.get("stoprounds", "-1"));

    if (!arg.unused.empty())
      error("unrecognized options");

    info(fmt("dim=%dx%d reps=%ld", iw, ih, reps));

    learn(src, tgt, chn, dis, cnd, enc, inf, repint, mul, reps, stoprounds);

    delete src;
    delete tgt;
    delete chn;
    delete dis;
    return 0;
  }

  uerror("unknown command " + cmd);
  assert(0);
}
