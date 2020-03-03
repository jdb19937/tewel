#define __MAKEMORE_TEWEL_CC__ 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#include <math.h>

#include "colonel.hh"
#include "cortex.hh"
#include "youtil.hh"
#include "random.hh"
#include "kleption.hh"
#include "cmdline.hh"
#include "display.hh"
#include "camera.hh"
#include "picpipes.hh"

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
    "        tewel learnstyl gen.ctx src=... sty=... dis=dis.ctx\n"
    "                train generator on paired samples versus discriminator\n"
    "        tewel learnhans gen.ctx src=... tgt=... dis=dis.ctx\n"
    "                train generator on paired samples versus discriminator\n"
  );
}

static void uerror(const std::string &str) {
  fprintf(stderr, "Error: %s\n", str.c_str());
  usage();
  exit(1);
}

void learnauto(
  Kleption *src,
  Cortex *gen,
  Cortex *enc,
  int repint,
  double mul,
  long reps
) {
  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);
    assert(enc->iw == gen->ow);
    assert(enc->ih == gen->oh);
    assert(enc->ic == gen->oc);
  } else {
    assert(gen->iw == gen->ow);
    assert(gen->ih == gen->oh);
    assert(gen->ic == gen->oc);
  }

  double t0 = now();
  int rep = 0;
  while (reps < 0 || rep < reps) {
    if (enc) {
      assert(src->pick(enc->kinp));
      enc->synth();
      gen->synth(enc->kout);
      gen->target(enc->kinp);
      gen->learn(mul);
      enc->learn(gen->kinp, mul);
    } else {
      assert(src->pick(gen->kinp));
      gen->synth();
      gen->target(gen->kinp);
      gen->learn(mul);
    }

    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g %sgenrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt / (double)repint, reps < 0 ? 0 : (reps - rep) * dt,
        enc ? fmt("encrms=%g ", enc->rms).c_str() : "",
        gen->rms
      );

      ++rep;
    }
  }
}

void learnfunc(
  Kleption *src,
  Kleption *tgt,
  Cortex *gen,
  Cortex *enc,
  int repint,
  double mul,
  long reps
) {
  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);
  }

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  double t0 = now();
  int rep = 0;

  while (reps < 0 || rep < reps) {
    if (enc) {
      Kleption::pick_pair(src, enc->kinp, tgt, ktgt);
      enc->synth();
      gen->synth(enc->kout);
    } else {
      Kleption::pick_pair(src, gen->kinp, tgt, ktgt);
      gen->synth();
    }

    gen->target(ktgt);
    gen->learn(mul);

    if (enc) {
      enc->learn(gen->kinp, mul);
    } 

    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g %sgenrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt / (double)repint, reps < 0 ? 0 : (reps - rep) * dt,
        enc ? fmt("encrms=%g ", enc->rms).c_str() : "",
        gen->rms
      );

      ++rep;
    }
  }

  kfree(ktgt);
}

void learnstyl(
  Kleption *src,
  Kleption *sty,
  Cortex *gen,
  Cortex *dis,
  Cortex *enc,
  int repint,
  double mul,
  bool lossreg,
  long reps
) {
  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);
  }

  assert(gen->ow == dis->iw);
  assert(gen->oh == dis->ih);
  assert(gen->oc == dis->ic);

  double *ktmp;
  kmake(&ktmp, gen->owhc);

  double *kreal;
  kmake(&kreal, dis->owhc);
  kfill(kreal, dis->owhc, 0.0);

  double *kfake;
  kmake(&kfake, dis->owhc);
  kfill(kfake, dis->owhc, 1.0);

  double t0 = now();
  int rep = 0;

  while (reps < 0 || rep < reps) {
    double genmul = mul;
    double dismul = mul;
    if (lossreg) {
      genmul *= (dis->rms > 0.5 ? 0.0 : 2.0 * (0.5 - dis->rms));
      dismul *= (dis->rms > 0.5 ? 1.0 : 2.0 * dis->rms);
    }
 
    if (enc) {
      assert(src->pick(enc->kinp));
      enc->synth();
      gen->synth(enc->kout);
    } else {
      assert(src->pick(gen->kinp));
      gen->synth();
    }

    kcopy(gen->kout, gen->owhc, ktmp);

    dis->synth(ktmp);
    dis->target(kreal);
    gen->learn(dis->propback(), genmul);

    if (enc) {
      enc->learn(gen->kinp, genmul);
    } 

    dis->synth(ktmp);
    dis->target(kfake);
    dis->learn(dismul);

    assert(sty->pick(dis->kinp));
    dis->synth();
    dis->target(kreal);
    dis->learn(dismul);

    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();
      dis->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g %sgenrms=%g disrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt / (double)repint, reps < 0 ? 0 : (reps - rep) * dt,
        enc ? fmt("encrms=%g ", enc->rms).c_str() : "",
        gen->rms, dis->rms
      );

      ++rep;
    }
  }

  kfree(ktmp);
}

void learnhans(
  Kleption *src,
  Kleption *tgt,
  Cortex *gen,
  Cortex *dis,
  Cortex *enc,
  int repint,
  double mul,
  bool lossreg,
  double noise,
  long reps
) {

  int iw, ih, ic, iwhc;
  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);
    iw = enc->iw;
    ih = enc->ih;
    ic = enc->ic;
  } else {
    iw = gen->iw;
    ih = gen->ih;
    ic = gen->ic;
  }
  iwhc = iw * ih * ic;

  assert(iw == gen->ow);
  assert(ih == gen->oh);
  assert(dis->iw == gen->ow);
  assert(dis->ih == gen->oh);
  assert(dis->ic == ic + gen->oc);

  double *ktmp;
  kmake(&ktmp, gen->owhc);

  double *kinp;
  kmake(&kinp, iwhc);
  double *kinpn;
  kmake(&kinpn, iwhc);

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  double *kreal;
  kmake(&kreal, dis->owhc);
  kfill(kreal, dis->owhc, 0.0);

  double *kfake;
  kmake(&kfake, dis->owhc);
  kfill(kfake, dis->owhc, 1.0);

  double t0 = now();
  int rep = 0;

  while (reps < 0 || rep < reps) {
    double genmul = mul;
    double dismul = mul;
    if (lossreg) {
      genmul *= (dis->rms > 0.5 ? 0.0 : 2.0 * (0.5 - dis->rms));
      dismul *= (dis->rms > 0.5 ? 1.0 : 2.0 * dis->rms);
    }
 
    Kleption::pick_pair(src, kinp, tgt, ktgt);
    if (enc) {
      enc->synth(kinp);
      gen->synth(enc->kout);
    } else {
      gen->synth(kinp);
    }

    kcopy(gen->kout, gen->owhc, ktmp);

    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    kcopy(kinp, iwhc, kinpn);
    kaddnoise(kinpn, iwhc, noise);
    ksplice(kinpn, iw * ih, ic, 0, ic, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kreal);
    dis->propback();

    ksplice(dis->kinp, gen->ow * gen->oh, dis->ic, 0, gen->oc, gen->kout, gen->oc, 0);

    gen->learn(genmul);
    if (enc) {
      enc->learn(gen->kinp, genmul);
    } 


    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    kcopy(kinp, iwhc, kinpn);
    kaddnoise(kinpn, iwhc, noise);
    ksplice(kinpn, iw * ih, ic, 0, ic, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kfake);
    dis->learn(dismul);




    ksplice(ktgt, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    kcopy(kinp, iwhc, kinpn);
    kaddnoise(kinpn, iwhc, noise);
    ksplice(kinpn, iw * ih, ic, 0, ic, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kreal);
    dis->learn(dismul);




    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();
      dis->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g %sgenrms=%g disrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt / (double)repint, reps < 0 ? 0 : (reps - rep) * dt,
        enc ? fmt("encrms=%g ", enc->rms).c_str() : "",
        gen->rms, dis->rms
      );

      ++rep;
    }
  }

  kfree(ktmp);
  kfree(ktgt);
  kfree(kinp);
}

static void cpumatmat(const double *a, const double *b, int aw, int ahbw, int bh, double *c) {
  int ah = ahbw;
  int bw = ahbw;
  int ch = bh;
  int cw = aw;

  for (int cy = 0; cy < ch; ++cy) {
    if (cy % 16 == 0)
      info(fmt("multiplying matrix progress=%g", (double)cy / (double)ch));
    for (int cx = 0; cx < cw; ++cx) {
      c[cx + cw * cy] = 0;
      for (int i = 0; i < ahbw; ++i) {
        c[cx + cw * cy] += a[cx + aw * i] * b[i + bw * cy];
      }
    }
  }
  info(fmt("multiplying matrix progress=%g", 1));
}

static void cpumatvec(const double *a, const double *b, int aw, int ahbw, double *c) {
  int ah = ahbw;
  int bw = ahbw;
  int cw = aw;

  for (int cx = 0; cx < cw; ++cx) {
    c[cx] = 0;
    for (int i = 0; i < ahbw; ++i) {
      c[cx] += a[cx + aw * i] * b[i];
    }
  }
}

static void cpuchol(double *m, unsigned int dim) {
  unsigned int i, j, k; 

  for (k = 0; k < dim; k++) {
#if 0
    assert(m[k * dim + k] > 0);
#endif
#if 1
    if (m[k * dim + k] < 0)
      m[k * dim + k] = 0;
#endif


    m[k * dim + k] = sqrt(m[k * dim + k]);
    
#if 1
    if (m[k * dim + k] > 0) {
#endif
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] /= m[k * dim + k];
#if 1
    } else {
      m[k * dim + k] = 1.0;
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] = 0;
    }
#endif
             
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
  Kleption *src,
  Cortex *gen,
  Cortex *enc,
  int limit
) {
  int i = 0;
  int owh = enc->ow * enc->oh;
  int oc = enc->oc;
  int owhc = owh * oc;

  double *tmp = new double[owhc];
  double *sum = new double[oc];
  double *mean = new double[oc];
  double *summ = new double[oc * oc];
  double *cov = new double[oc * oc];
  double *chol = new double[oc * oc];
  double *unchol = new double[oc * oc];

  memset(sum, 0, oc * sizeof(double));
  memset(summ, 0, oc * oc * sizeof(double));

  while (1) {
    if (limit >= 0 && i >= limit)
      break;

    std::string id;
    if (!src->pick(enc->kinp, &id))
      break;
    enc->synth();
    dek(enc->kout, enc->owhc, tmp);

    info("encoding " + id);

    for (unsigned int xy = 0; xy < owh; ++xy) {
      for (unsigned int z = 0; z < oc; ++z) {
        sum[z] += tmp[z + oc * xy];
      }
      for (unsigned int z0 = 0; z0 < oc; ++z0) {
        for (unsigned int z1 = 0; z1 < oc; ++z1) {
          summ[z0 + z1 * oc] += tmp[z0 + oc * xy] * tmp[z1 + oc * xy];
        }
      }
    }
    ++i;
  }
  if (i < 2)
    error("need more samples for normalize");
  long n = owh * i;

  for (int z = 0; z < oc; ++z) {
    mean[z] = sum[z] / (double)n;
  }
  for (unsigned int z0 = 0; z0 < oc; ++z0) {
    for (unsigned int z1 = 0; z1 < oc; ++z1) {
      cov[z0 + z1 * oc] = summ[z0 + z1 * oc] / (double)n - mean[z0] * mean[z1];
    }
  }

  memcpy(chol, cov, oc * oc * sizeof(double));
  cpuchol(chol, oc);
  matinv(chol, unchol, oc);

  double *em, *eb;
  int ew, eh;
  enc->_get_tail_op(&eb, &em, &ew, &eh);
  assert(ew > 0);
  assert(eh == oc);

  double *gm, *gb;
  int gw, gh;
  gen->_get_head_op(&gb, &gm, &gw, &gh);
  assert(gw == oc);
  assert(gh > 0);

  if (eh != gw)
    error("matrices don't match, gen needs to start with con0 to normalize");

  for (int z = 0; z < oc; ++z)
    eb[z] -= mean[z];
  for (int y = 0; y < gh; ++y) {
    double q = 0;
    for (int z = 0; z < oc; ++z)
      q += gm[z + y * gw] * mean[z];
    gb[y] += q;
  }


  double *tem = new double[ew * eh];
  cpumatmat(em, unchol, ew, oc, oc, tem);
  memcpy(em, tem, sizeof(double) * ew * eh);

  double *tgm = new double[gw * gh];
  cpumatmat(chol, gm, oc, oc, gh, tgm);
  memcpy(gm, tgm, sizeof(double) * gw * gh);

  gen->_put_head_op(gb, gm, gw, gh);
  enc->_put_tail_op(eb, em, ew, eh);

  delete[] tmp;
  delete[] sum;
  delete[] mean;
  delete[] summ;
  delete[] cov;
  delete[] chol;
  delete[] unchol;

  delete[] tem;
  delete[] tgm;
  delete[] em;
  delete[] gm;
  delete[] eb;
  delete[] gb;
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

  std::string ctx = argv[0];
  ++argv;
  --argc;

  Cmdline arg(argc, argv, "0");
  verbose = arg.get("verbose", "0");
  info(fmt("verbose=%d", verbose));

  if (startswith(ctx, "//"))
    ctx = "/opt/makemore/share/tewel/" + std::string(ctx.c_str() + 2);

  bool ctx_is_dir = is_dir(ctx);

   
  setkdev(arg.get("cuda", kndevs() > 1 ? "1" : "0"));
  setkbs(arg.get("kbs", "256"));

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

  if (cmd == "make") {
    if (ctx_is_dir)
      error("dir exists with name of file");

    std::string spec = arg.get("spec");
    if (startswith(spec, "//"))
      spec = "/opt/makemore/share/tewel/" + std::string(spec.c_str() + 2);

    int clobber = arg.get("clobber", "0");

    if (!arg.unused.empty())
      error("unrecognized options");

    Cortex::create(ctx, (bool)clobber);

    Cortex *out = new Cortex(ctx);

    FILE *specfp = fopen(spec.c_str(), "r");
    if (!specfp)
      error(str("can't open ") + spec + ": " + strerror(errno));

    std::string specline;
    if (read_line(specfp, &specline)) {
      strmap specopt;

      specopt["decay"] = "1e-2";
      specopt["nu"] = "1e-4";
      specopt["b1"] = "1e-1";
      specopt["b2"] = "1e-3";
      specopt["eps"] = "1e-8";
      specopt["rdev"] = "0.1";

      if (!parsekv(specline, &specopt))
        error(str("can't parse kv in header of ") + spec);

      if (specopt["v"] != "2")
        error(str("missing magic v=2 in ") + spec);

      out->head->decay = strtod(specopt["decay"]);
      out->head->nu = strtod(specopt["nu"]);
      out->head->b1 = strtod(specopt["b1"]);
      out->head->b2 = strtod(specopt["b2"]);
      out->head->eps = strtod(specopt["eps"]);
      out->head->rdev = strtod(specopt["rdev"]);
    } else {
      error("no header line in spec");
    }

    int poc = 0;
    while (read_line(specfp, &specline)) {
      int specargc;
      char **specargv;
      parseargstrad(specline, &specargc, &specargv);

      Cmdline specargs(specargc, specargv, "");
      std::string type = specargs.get("type");
      int ic = specargs.get("ic", "0");
      int oc = specargs.get("oc", "0");
      int iw = specargs.get("iw", "0");
      int ih = specargs.get("ih", "0");
      int ow = specargs.get("ow", "0");
      int oh = specargs.get("oh", "0");

      freeargstrad(specargc, specargv);
      if (!specargs.unused.empty())
        error("bad options in spec layer");

      if (ic == 0)
        ic = poc;
      if (ic == 0)
        error("input channels required");
      if (poc && ic != poc)
        error("input channels don't match output channels of previous layer");

      out->push(type, ic, oc, iw, ih, ow, oh);

      poc = out->oc;
    }
    fclose(specfp);

    out->scram(out->head->rdev);
    delete out;

    return 0;
  }

  if (cmd == "edit") {
    if (ctx_is_dir)
      error("dir exists with name of file");

    Cortex *gen = new Cortex(ctx);

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
    if (arg.present("rdev"))
      gen->head->rdev = arg.get("rdev");

    if (!arg.unused.empty())
      warning("unrecognized options");

    delete gen;
    gen = new Cortex(ctx);

    printf(
      "rounds=%lu rms=%g max=%g decay=%g nu=%g b1=%g b2=%g eps=%g rdev=%g\n",
      gen->rounds, gen->rms, gen->max, gen->decay,
      gen->nu, gen->b1, gen->b2, gen->eps, gen->head->rdev
    );

    return 0;
  }

  if (cmd == "spec") {
    if (ctx_is_dir)
      error("dir exists with name of file");
    if (!arg.unused.empty())
      warning("unrecognized options");
    Cortex *gen = new Cortex(ctx, O_RDONLY);

    printf(
      "v=2 decay=%g nu=%g b1=%g b2=%g eps=%g rdev=%g\n",
      gen->decay, gen->nu, gen->b1, gen->b2, gen->eps, gen->head->rdev
    );

    gen->dump(stdout);

    delete gen;
    return 0;
  }

  if (cmd == "bindump") {
    if (ctx_is_dir)
      error("dir exists with name of file");
    unsigned int a = 0, b = (1 << 31);
    if (arg.present("cut"))
      if (!parserange(arg.get("cut"), &a, &b))
        error("cut must be number or number range");

    if (!arg.unused.empty())
      warning("unrecognized options");

    Cortex *gen = new Cortex(ctx, O_RDONLY);
    gen->bindump(stdout, a, b);
    delete gen;
    return 0;
  }

  if (cmd == "reset") {
    if (ctx_is_dir)
      error("dir exists with name of file");
    Cortex *gen = new Cortex(ctx);

    gen->head->rounds = 0;
    gen->head->rms = 0;
    gen->head->max = 0;

    if (arg.present("rdev"))
      gen->head->rdev = arg.get("rdev");

    if (!arg.unused.empty())
      warning("unrecognized options");

    double rdev = gen->head->rdev;
    gen->scram(rdev);

    delete gen;
    return 0;
  }

  if (cmd == "synth") {
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    Cortex *gen = new Cortex(genfn, O_RDONLY);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
    }

    int limit = arg.get("limit", "-1");
    int reload = arg.get("reload", "0");
    double delay = arg.get("delay", "0.0");

    Kleption::Trav srctrav = Kleption::get_trav(arg.get("srctrav", "scan"));
    if (srctrav == Kleption::TRAV_NONE)
      error("srctrav must be scan or rand or refs");

    Kleption::Flags srcflags = 0;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

    int repeat = arg.get("repeat", "0");
    if (repeat)
      srcflags |= Kleption::FLAG_REPEAT;

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
    int pc = enc ? enc->ic : gen->ic;

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

    std::string srcfn = arg.get("src");

    Kleption *src = new Kleption(
      srcfn,
      pw, ph, pc,
      srcflags, srctrav, srckind,
      sw, sh, sc,
      refsfn
    );

    
   

    src->load();
    assert(src->pw > 0);
    assert(src->ph > 0);

    int ic;
    if (enc) {
      assert(enc->ic == src->pc);
      enc->prepare(src->pw, src->ph);
      gen->prepare(enc->ow, enc->oh);
      if (enc->oc != gen->ic)
        error("encoder oc doesn't match generator ic");
      ic = enc->ic;
    } else {
      assert(gen->ic == src->pc);
      gen->prepare(src->pw, src->ph);
      ic = gen->ic;
    }

    Kleption::Kind outkind = Kleption::get_kind(arg.get("outkind", ""));
    if (outkind == Kleption::KIND_UNK)
      error("unknown outkind");
    if (outkind != Kleption::KIND_SDL)
      arg.get("out");

    Kleption::Flags outflags = Kleption::FLAG_WRITER;
    if (lowmem)
      outflags |= Kleption::FLAG_LOWMEM;

    Kleption *out = new Kleption(
      arg.get("out", ""),
      gen->ow, gen->oh, gen->oc,
      outflags, Kleption::TRAV_SCAN, outkind
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    int i = 0;
    while (1) {
      if (limit >= 0 && i >= limit)
        break;

      std::string id;
      if (enc) {
        if (!src->pick(enc->kinp, &id))
          break;
        enc->synth();
        kcopy(enc->kout, enc->owhc, gen->kinp);
      } else {
        if (!src->pick(gen->kinp, &id))
          break;
      }
      gen->synth();

      if (!out->place(id, gen->kout))
        break;

      if (delay > 0)
        usleep(delay * 1000000.0);
      if (reload) {
        gen->load();
        if (enc)
          enc->load();
      }
      ++i;
    }


    delete gen;
    if (enc)
      delete enc;
    delete src;
    delete out;
    return 0;
  }

  if (cmd == "normalize") {
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    Cortex *gen = new Cortex(genfn);

    Cortex *enc;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
    } else {
      error("normalize requires encoder");
    }

    int limit = arg.get("limit", "-1");

    Kleption::Trav srctrav = Kleption::get_trav(arg.get("srctrav", "scan"));
    if (srctrav == Kleption::TRAV_NONE)
      error("srctrav must be scan or rand or refs");

    Kleption::Flags srcflags = 0;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

    int repeat = arg.get("repeat", "0");
    if (repeat)
      srcflags |= Kleption::FLAG_REPEAT;

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
    int pc = enc->ic;

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

    std::string srcfn = arg.get("src");

    Kleption *src = new Kleption(
      srcfn,
      pw, ph, pc,
      srcflags, srctrav, srckind,
      sw, sh, sc,
      refsfn
    );

    src->load();
    assert(src->pw > 0);
    assert(src->ph > 0);

    int ic = enc->ic;
    assert(enc->ic == src->pc);
    enc->prepare(src->pw, src->ph);
    gen->prepare(enc->ow, enc->oh);
    if (enc->oc != gen->ic)
      error("encoder oc doesn't match generator ic");

    if (!arg.unused.empty())
      error("unrecognized options");

    normalize(src, gen, enc, limit);

    delete gen;
    delete enc;
    delete src;
    return 0;
  }


  std::map<std::string, std::string> diropt;

  diropt["repint"] = "64";
  diropt["reps"] = "-1";
  diropt["mul"] = "1.0";
  diropt["lossreg"] = "1";
  diropt["noise"] = "0.5";
  diropt["srcdim"] = "0x0x0";
  diropt["tgtdim"] = "0x0x0";
  diropt["stydim"] = "0x0x0";
  diropt["srckind"] = "";
  diropt["tgtkind"] = "";
  diropt["stykind"] = "";
  
  if (ctx_is_dir) {
    if (!fexists(ctx + "/gen.ctx"))
      error(fmt("%s/gen.ctx: %s", ctx.c_str(), strerror(errno)));
    diropt["gen"] = ctx + "/gen.ctx";

    if (fexists(ctx + "/enc.ctx"))
      diropt["enc"] = ctx + "/enc.ctx";
    if (fexists(ctx + "/dis.ctx"))
      diropt["dis"] = ctx + "/dis.ctx";

    std::string diroptfn = ctx + "/opt.txt";
    if (fexists(diroptfn)) {
      info(fmt("reading default options from %s", diroptfn.c_str()));

      std::string line;

      FILE *optfp = fopen(diroptfn.c_str(), "r");
      if (!optfp)
        error(fmt("can't open %s", diroptfn.c_str()));
      if (!read_line(optfp, &line))
        error(fmt("can't read line from %s", diroptfn.c_str()));
      if (getc(optfp) != EOF)
        warning(fmt("extra data in %s following first line", diroptfn.c_str()));
      fclose(optfp);

      if (!parsekv(line, &diropt))
        error(fmt("can't parse options in %s", diroptfn.c_str()));
      info(kvstr(diropt));

      if (diropt.count("src") && !startswith(diropt["src"], "/"))
        diropt["src"] = ctx + "/" + diropt["src"];
      if (diropt.count("tgt") && !startswith(diropt["tgt"], "/"))
        diropt["tgt"] = ctx + "/" + diropt["tgt"];
      if (diropt.count("sty") && !startswith(diropt["sty"], "/"))
        diropt["sty"] = ctx + "/" + diropt["sty"];
    }
  }


  if (cmd == "learnauto") {
    std::string genfn = ctx_is_dir ? diropt["gen"] : ctx;

    int iw, ih;
    if (arg.present("dim")) {
      if (!parsedim2(arg.get("dim"), &iw, &ih))
        error("dim must be like 256 or like 256x256");
    } else if (diropt.count("dim")) {
      if (!parsedim2(diropt["dim"], &iw, &ih))
        error("dim must be like 256 or like 256x256");
    }

    int repint = arg.get("repint", diropt["repint"]);
    double mul = arg.get("mul", diropt["mul"]);

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (diropt.count("enc")) {
      enc = new Cortex(diropt["enc"]);
    }

    int ic;
    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
      ic = enc->ic;
    } else {
      gen->prepare(iw, ih);
      ic = gen->ic;
    }

    if (iw != gen->ow)
      error("input and output width don't match");
    if (ih != gen->oh)
      error("input and output height don't match");
    if (ic != gen->oc)
      error("input and output channels don't match");

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
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      srcflags, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    learnauto(src, gen, enc, repint, mul, reps);

    delete src;
    delete gen;
    if (enc)
      delete enc;
    return 0;
  }


  if (cmd == "learnfunc") {
    std::string genfn = ctx_is_dir ? diropt["gen"] : ctx;

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

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (diropt.count("enc")) {
      enc = new Cortex(diropt["enc"]);
    }

    int ic;
    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
      ic = enc->ic;
    } else {
      gen->prepare(iw, ih);
      ic = gen->ic;
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
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

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
      tgtfn, gen->ow, gen->oh, gen->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    learnfunc(src, tgt, gen, enc, repint, mul, reps);

    delete src;
    delete tgt;
    delete gen;
    if (enc)
      delete enc;
    return 0;
  }

  if (cmd == "learnstyl") {
    std::string genfn = ctx_is_dir ? diropt["gen"] : ctx;

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
    int lossreg = arg.get("lossreg", diropt["lossreg"]);

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (diropt.count("enc")) {
      enc = new Cortex(diropt["enc"]);
    }

    int ic;
    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
      ic = enc->ic;
    } else {
      gen->prepare(iw, ih);
      ic = gen->ic;
    }

    Cortex *dis = NULL;
    if (arg.present("dis")) {
      dis = new Cortex(arg.get("dis"));
    } else if (diropt.count("dis")) {
      dis = new Cortex(diropt["dis"]);
    } else {
      error("dis argument required");
    }

    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc)
      error("gen oc doesn't match dis ic");

    std::string srcdim = arg.get("srcdim", diropt["srcdim"]);
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", diropt["srckind"]));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string) arg.get("src");
    } else if (diropt.count("src")) {
      srcfn = diropt["src"];
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption::Flags srcflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      srcflags |= Kleption::FLAG_LOWMEM;
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      srcflags, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string stydim = arg.get("stydim", diropt["stydim"]);
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(stydim, &tw, &th, &tc))
      error("bad stydim format");

    Kleption::Kind stykind = Kleption::get_kind(arg.get("stykind", diropt["stykind"]));

    std::string styfn;
    if (arg.present("sty")) {
      styfn = (std::string)arg.get("sty");
    } else if (diropt.count("sty")) {
      styfn = diropt["sty"];
    } else {
      arg.get("sty");
      assert(0);
    }

    Kleption::Flags styflags = Kleption::FLAG_REPEAT;
    if (lowmem)
      styflags |= Kleption::FLAG_LOWMEM;

    Kleption *sty = new Kleption(
      styfn, gen->ow, gen->oh, gen->oc,
      styflags, Kleption::TRAV_RAND, stykind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    learnstyl(src, sty, gen, dis, enc, repint, mul, lossreg, reps);

    delete src;
    delete sty;

    delete gen;
    if (enc)
      delete enc;
    delete dis;
    return 0;
  }



  if (cmd == "learnhans") {
    std::string genfn = ctx_is_dir ? diropt["gen"] : ctx;

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
    int lossreg = arg.get("lossreg", diropt["lossreg"]);
    double noise = arg.get("noise", diropt["noise"]);

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (diropt.count("enc")) {
      enc = new Cortex(diropt["enc"]);
    }

    int ic;
    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
      ic = enc->ic;
    } else {
      gen->prepare(iw, ih);
      ic = gen->ic;
    }

    if (iw != gen->ow)
      error("iw differs from ow");
    if (ih != gen->oh)
      error("iw differs from oh");

    Cortex *dis = NULL;
    if (arg.present("dis")) {
      dis = new Cortex(arg.get("dis"));
    } else if (diropt.count("dis")) {
      dis = new Cortex(diropt["dis"]);
    } else {
      error("dis argument required");
    }

    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc + ic)
      error("gen oc+ic doesn't match dis ic");

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
    int addgeo = arg.get("addgeo", "0");
    if (addgeo)
      srcflags |= Kleption::FLAG_ADDGEO;

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
      tgtfn, gen->ow, gen->oh, gen->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    info(fmt("dim=%dx%d reps=%ld", iw, ih, reps));

    learnhans(src, tgt, gen, dis, enc, repint, mul, lossreg, noise, reps);

    delete src;
    delete tgt;

    delete gen;
    if (enc)
      delete enc;
    delete dis;
    return 0;
  }

  uerror("unknown command " + cmd);
  assert(0);
}
