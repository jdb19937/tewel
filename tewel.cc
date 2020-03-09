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
#include "rando.hh"
#include "kleption.hh"
#include "cmdline.hh"
#include "display.hh"
#include "camera.hh"
#include "picpipes.hh"
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
  Chain *chn,
  int repint,
  double mul,
  long reps
) {
  assert(chn->head->iw == chn->tail->ow);
  assert(chn->head->ih == chn->tail->oh);
  assert(chn->head->ic == chn->tail->oc);

  Cortex *gen = chn->tail;

  double t0 = now();
  int rep = 0;
  while (reps < 0 || rep < reps) {
    assert(src->pick(chn->kinp()));
    chn->synth();
    chn->target(chn->kinp());
    chn->learn(mul);

    if (chn->tail->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt, reps < 0 ? 0 : (reps - rep) * dt,
        gen->rms
      );

      ++rep;
    }
  }
}

void learnfunc(
  Kleption *src,
  Kleption *tgt,
  Chain *chn,
  int repint,
  double mul,
  long reps
) {
  Cortex *gen = chn->tail;

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  double t0 = now();
  int rep = 0;

  while (reps < 0 || rep < reps) {
    Kleption::pick_pair(src, chn->kinp(), tgt, ktgt);
    chn->synth();
    chn->target(ktgt);
    chn->learn(mul);

    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt, reps < 0 ? 0 : (reps - rep) * dt,
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
  Chain *chn,
  Cortex *dis,
  int repint,
  double mul,
  bool lossreg,
  long reps
) {
  Cortex *gen = chn->tail;

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
 
    assert(src->pick(chn->kinp()));
    chn->synth();

    kcopy(gen->kout, gen->owhc, ktmp);

    dis->synth(ktmp);
    dis->target(kreal);

    kcopy(dis->propback(), gen->owhc, gen->kout); 
    chn->learn(genmul);

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

      chn->save();
      dis->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g disrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt, reps < 0 ? 0 : (reps - rep) * dt,
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
  Chain *chn,
  Cortex *dis,
  Cortex *cnd,
  int repint,
  double mul,
  bool lossreg,
  long reps
) {
  Cortex *gen = chn->tail;

  int iw, ih, ic, iwhc;
  iw = chn->head->iw;
  ih = chn->head->ih;
  ic = chn->head->ic;
  iwhc = iw * ih * ic;

  if (cnd) {
    assert(cnd->iw == iw);
    assert(cnd->ih == ih);
    assert(cnd->ic == ic);
  }

  assert((cnd ? cnd->ow : iw) == gen->ow);
  assert((cnd ? cnd->oh : ih) == gen->oh);
  assert(dis->iw == gen->ow);
  assert(dis->ih == gen->oh);

  assert(dis->ic == (cnd ? cnd->oc : ic) + gen->oc);

  double *ktmp;
  kmake(&ktmp, gen->owhc);

  double *kinp;
  kmake(&kinp, iwhc);

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
    kcopy(kinp, iwhc, chn->kinp());
    chn->synth();

    kcopy(gen->kout, gen->owhc, ktmp);

    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    ksplice(cnd ? cnd->synth(kinp) : kinp, dis->iw * dis->ih, dis->ic - gen->oc, 0, dis->ic - gen->oc, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kreal);
    dis->propback();

    ksplice(dis->kinp, gen->ow * gen->oh, dis->ic, 0, gen->oc, gen->kout, gen->oc, 0);

    chn->learn(genmul);

    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    ksplice(cnd ? cnd->synth(kinp) : kinp, dis->iw * dis->ih, dis->ic - gen->oc, 0, dis->ic - gen->oc, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kfake);
    dis->learn(dismul);
    if (cnd) {
      ksplice(dis->kinp, dis->iw * dis->oh, dis->ic, gen->oc, cnd->oc, cnd->kout, cnd->oc, 0);
      cnd->learn(dismul);
    }



    ksplice(ktgt, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    ksplice(cnd ? cnd->synth(kinp) : kinp, dis->iw * dis->ih, dis->ic - gen->oc, 0, dis->ic - gen->oc, dis->kinp, dis->ic, gen->oc);

    dis->synth();
    dis->target(kreal);
    dis->learn(dismul);
    if (cnd) {
      ksplice(dis->kinp, dis->iw * dis->oh, dis->ic, gen->oc, cnd->oc, cnd->kout, cnd->oc, 0);
      cnd->learn(dismul);
    }




    if (gen->rounds % repint == 0) {
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      chn->save();
      dis->save();

      if (cnd)
        cnd->save();

      printf(
        "gen=%s genrounds=%lu dt=%g eta=%g genrms=%g disrms=%g\n",
         gen->fn.c_str(), gen->rounds,
         dt, reps < 0 ? 0 : (reps - rep) * dt,
        gen->rms, dis->rms
      );

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
  Cortex *gen,
  int limit
) {
  int owh = enc->ow * enc->oh;
  int oc = enc->oc;
  int owhc = owh * oc;

  assert(rvg->dim == oc);

  double *mean = new double[oc];
  double *chol = new double[oc * oc];
  double *unchol = new double[oc * oc];

  memcpy(mean, rvg->mean, oc * sizeof(double));
  memcpy(chol, rvg->chol, oc * oc * sizeof(double));
  matinv(chol, unchol, oc);
  cpumatxpose(chol, oc);
  cpumatxpose(unchol, oc);

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
  double *teb = new double[eh];
  double *tgm = new double[gw * gh];

  cpumatmat(unchol, oc, oc, em, ew, eh, tem);
  memcpy(em, tem, sizeof(double) * ew * eh);

  cpumatvec(unchol, oc, oc, eb, teb);
  memcpy(eb, teb, sizeof(double) * eh);

  cpumatmat(gm, gw, gh, chol, oc, oc, tgm);
  memcpy(gm, tgm, sizeof(double) * gw * gh);

  gen->_put_head_op(gb, gm, gw, gh);
  enc->_put_tail_op(eb, em, ew, eh);

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
    bool got_header = false;
    while (read_line(specfp, &specline)) {
      strmap specopt;

      specopt["decay"] = "1e-2";
      specopt["nu"] = "1e-4";
      specopt["b1"] = "1e-1";
      specopt["b2"] = "1e-3";
      specopt["eps"] = "1e-8";
      specopt["rdev"] = "0.1";

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
      out->head->rdev = strtod(specopt["rdev"]);

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

      int keys = parsekv(specline, &specopt);
      if (keys < 0)
        error(str("can't parse kv in line of ") + spec);
      if (keys == 0)
        continue;

      std::string type = specopt["type"];
      int iw = strtoi(specopt["iw"]);
      int ih = strtoi(specopt["ih"]);
      int ic = strtoi(specopt["ic"]);
      int ow = strtoi(specopt["ow"]);
      int oh = strtoi(specopt["oh"]);
      int oc = strtoi(specopt["oc"]);


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
    out->head->rdev = 0;

    out->push("con0", dim, dim);

    double *m, *b;
    int w, h;
    out->_get_head_op(&b, &m, &w, &h);
    assert(w == dim);
    assert(h == dim);

    memset(b, 0, sizeof(double) * dim);

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
    if (arg.present("rdev"))
      gen->head->rdev = arg.get("rdev");

    if (!arg.unused.empty())
      warning("unrecognized options");

    delete gen;
    gen = new Cortex(ctx[0]);

    printf(
      "rounds=%lu rms=%g max=%g decay=%g nu=%g b1=%g b2=%g eps=%g rdev=%g\n",
      gen->rounds, gen->rms, gen->max, gen->decay,
      gen->nu, gen->b1, gen->b2, gen->eps, gen->head->rdev
    );

    return 0;
  }

  if (cmd == "spec") {
    if (ctx.size() != 1)
      error("spec requires exactly one ctx file arg");
    if (!arg.unused.empty())
      warning("unrecognized options");
    Cortex *gen = new Cortex(ctx[0], O_RDONLY);

    printf(
      "v=2 decay=%g nu=%g b1=%g b2=%g eps=%g rdev=%g\n",
      gen->decay, gen->nu, gen->b1, gen->b2, gen->eps, gen->head->rdev
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
    gen->scram(rdev);

    delete gen;
    return 0;
  }

  if (cmd == "synth") {
    Chain *chn = new Chain;
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

    double evolve = 0.0;
    if (arg.present("evolve"))
      evolve = arg.get("evolve");

    std::string srcfn = arg.get("src");

    Kleption *src = new Kleption(
      srcfn,
      pw, ph, pc,
      srcflags, srctrav, srckind,
      sw, sh, sc,
      refsfn, evolve
    );

    
   

    src->load();
    assert(src->pw > 0);
    assert(src->ph > 0);

    int ic = chn->head->ic;
    assert(ic == src->pc);
    chn->prepare(src->pw, src->ph);
    
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

    Cortex *enc = new Cortex(encfn);
    Cortex *gen = new Cortex(genfn);
    if (enc->oc != gen->ic)
      error("encoder oc doesn't match generator ic");

    if (!arg.present("rvg"))
      error("normalize requires rvg arg");
    Rando *rvg = new Rando;
    rvg->load(arg.get("rvg"));

    int limit = arg.get("limit", "-1");

    if (!arg.unused.empty())
      error("unrecognized options");

    normalize(rvg, enc, gen, limit);

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
  
  if (cmd == "learnauto") {
    Chain *chn = new Chain;
    for (auto ctxfn : ctx)
      chn->push(ctxfn, O_RDWR);

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

    int ic;
    chn->prepare(iw, ih);
    ic = chn->head->ic;

    if (iw != chn->tail->ow)
      error("input and output width don't match");
    if (ih != chn->tail->oh)
      error("input and output height don't match");
    if (ic != chn->tail->oc)
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

    learnauto(src, chn, repint, mul, reps);

    delete src;
    delete chn;
    return 0;
  }


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

    int ic;
    chn->prepare(iw, ih);
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
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    learnfunc(src, tgt, chn, repint, mul, reps);

    delete src;
    delete tgt;
    delete chn;
    return 0;
  }

  if (cmd == "learnstyl") {
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
    int lossreg = arg.get("lossreg", diropt["lossreg"]);

    int ic;
    chn->prepare(iw, ih);
    ic = chn->head->ic;

    Cortex *dis = NULL;
    if (arg.present("dis")) {
      dis = new Cortex(arg.get("dis"));
    } else {
      error("dis argument required");
    }

    dis->prepare(chn->tail->ow, chn->tail->oh);
    if (dis->ic != chn->tail->oc)
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
      styfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      styflags, Kleption::TRAV_RAND, stykind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    learnstyl(src, sty, chn, dis, repint, mul, lossreg, reps);

    delete src;
    delete sty;
    delete chn;
    delete dis;
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
    int lossreg = arg.get("lossreg", diropt["lossreg"]);

    int ic;
    chn->prepare(iw, ih);
    ic = chn->head->ic;

    Cortex *dis = NULL;
    if (arg.present("dis")) {
      dis = new Cortex(arg.get("dis"));
    } else {
      error("dis argument required");
    }

    dis->prepare(chn->tail->ow, chn->tail->oh);

    Cortex *cnd = NULL;
    if (arg.present("cnd")) {
      cnd = new Cortex(arg.get("cnd"));

      cnd->prepare(chn->head->iw, chn->head->ih);

      if (cnd->ow != dis->iw)
        error("cnd ow doesn't match dis iw");
      if (cnd->oh != dis->ih)
        error("cnd oh doesn't match dis ih");

      if (dis->ic != chn->tail->oc + cnd->oc)
        error("gen oc + cnd oc doesn't match dis ic");
    } else {
      if (iw != chn->tail->ow)
        error("gen ow doesn't match dis iw");
      if (ih != chn->tail->oh)
        error("gen oh doesn't match dis ih");

      if (dis->ic != chn->tail->oc + ic)
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
      tgtfn, chn->tail->ow, chn->tail->oh, chn->tail->oc,
      tgtflags, Kleption::TRAV_RAND, tgtkind,
      tw, th, tc
    );

    long reps = strtol(arg.get("reps", diropt["reps"]));

    if (!arg.unused.empty())
      error("unrecognized options");

    info(fmt("dim=%dx%d reps=%ld", iw, ih, reps));

    learnhans(src, tgt, chn, dis, cnd, repint, mul, lossreg, reps);

    delete src;
    delete tgt;
    delete chn;
    delete dis;
    return 0;
  }

  uerror("unknown command " + cmd);
  assert(0);
}
