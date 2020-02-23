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

using namespace makemore;

static void usage() {
  fprintf(stderr,
    "Usage:\n"
    "        tewel help\n"
    "                print this message\n"
    "        tewel new gen.mm\n"
    "                create a new generator\n"
    "        tewel dump gen.mm\n"
    "                display layer header contents\n"
    "        tewel scram gen.mm [stddev]\n"
    "                randomize generator state\n"
    "        tewel push gen.mm -t type [...]\n"
    "                append a layer to a generator\n"
    "        tewel ppmsynth gen.mm in.ppm > out.ppm\n"
    "                input ppm to a generator\n"
    "        tewel learnauto gen.mm -dat samples.dat ...\n"
    "                train generator on sample paired with self\n"
    "        tewel learnfunc gen.mm -dat samples.dat ...\n"
    "                train generator on paired samples\n"
    "        tewel learnhance gen.mm -dat samples.dat -dis dis.mm ...\n"
    "                train generator on paired samples versus discriminator\n"
  );
}

static void uerror(const std::string &str) {
  fprintf(stderr, "Error: %s\n", str.c_str());
  usage();
  exit(1);
}

static void parsecmd(int *argcp, char ***argvp, std::string *cmdp) {
  assert(*argcp > 0);
  const char *cmd = **argvp;
  while (*cmd == '-')
    ++cmd;
  ++*argvp;
  --*argcp;

  *cmdp = cmd;
}

static bool parseopt(
  int *argcp, char ***argvp,
  std::string *optp, std::string *valp
) {
  assert(*argcp >= 0);
  if (*argcp == 0)
    return false;

  const char *opt = **argvp;
  if (char *q = strchr(**argvp, '=')) {
    *q = 0;
    **argvp = q + 1;
  } else {
    if (*opt != '-') {
      opt = "";
    } else {
      --*argcp;
      ++*argvp;

      if (*argcp == 0) {
        warning("ignoring trailing option with no value");
        return false;
      }
    }
  }
  while (*opt == '-')
    ++opt;

  assert(*argcp > 0);
  const char *val = **argvp;
  --*argcp;
  ++*argvp;

  *optp = opt;
  *valp = val;

  return true;
}

struct Optval {
  std::string x;

  Optval() { }
  Optval(const std::string &_x) : x(_x) { }
  Optval(const char *_x) : x(_x) { }

  operator const std::string &() const
    { return x; }
  operator const char *() const
    { return x.c_str(); }
  operator int() const
    { return strtoi(x); }
  operator double() const
    { return strtod(x); }
};

struct Optmap {
  std::map<std::string, Optval> m;

  Optmap(int *argcp, char ***argvp) {
    std::string opt, val;
    while (parseopt(argcp, argvp, &opt, &val)) {
      if (opt == "")
        opt = "gen";
      put(opt, val);
    }
  }

  void put(const std::string &opt, const std::string &val) {
    if (m.count(opt)) 
      warning(std::string("ignoring repeated option -") + opt + "=" + val);
    else
      m[opt] = val;
  }

  const Optval &get(const std::string &opt) {
    if (m.count(opt))
      return m[opt];
    uerror("no value for required option -" + opt);
  }

  const Optval &get(const std::string &opt, const std::string &dval) {
    if (!m.count(opt))
      m[opt] = dval;
    return m[opt];
  }
};


void learnauto(
  Kleption *inp,
  Cortex *gen,
  Cortex *enc,
  int repint,
  double mul
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

  while (1) {
    if (enc) {
      inp->pick(enc->kinp);
      enc->_synth();
      kcopy(enc->kout, gen->iwhc, gen->kinp);

      gen->_synth();
      ksubvec(enc->kinp, gen->kout, gen->owhc, gen->kout);
      gen->_stats();
      gen->_learn(mul);

      kcopy(gen->kinp, gen->iwhc, enc->kout);
      enc->_stats();
      enc->_learn(mul);
    } else {
      inp->pick(gen->kinp);
      gen->_synth();
      ksubvec(gen->kinp, gen->kout, gen->owhc, gen->kout);
      gen->_stats();
      gen->_learn(mul);
    }

    if (gen->rounds % repint == 0) {
      if (enc) {
        enc->report();
        enc->save();
      }

      gen->report();
      gen->save();
    }
  }
}

void learnfunc(
  Kleption *inp,
  Kleption *tgt,
  Cortex *gen,
  Cortex *enc,
  int repint,
  double mul
) {
  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);
  }

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  while (1) {
    if (enc) {
      Kleption::pick_pair(inp, enc->kinp, tgt, ktgt);
      enc->_synth();
      kcopy(enc->kout, gen->iwhc, gen->kinp);
    } else {
      Kleption::pick_pair(inp, gen->kinp, tgt, ktgt);
    }

    gen->_synth();
    ksubvec(ktgt, gen->kout, gen->owhc, gen->kout);
    gen->_stats();
    gen->_learn(mul);

    if (enc) {
      kcopy(gen->kinp, gen->iwhc, enc->kout);
      enc->_stats();
      enc->_learn(mul);
    } 

    if (gen->rounds % repint == 0) {
      if (enc) {
        enc->report();
        enc->save();
      }

      gen->report();
      gen->save();
    }
  }

  kfree(ktgt);
}

void learnstyl(
  Kleption *inp,
  Kleption *tgt,
  Cortex *gen,
  Cortex *dis,
  Cortex *enc,
  int repint,
  double mul,
  bool lossreg
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

  while (1) {
    double genmul = mul;
    double dismul = mul;
    if (lossreg) {
      genmul *= (dis->rms > 0.5 ? 0.0 : 2.0 * (0.5 - dis->rms));
      dismul *= (dis->rms > 0.5 ? 1.0 : 2.0 * dis->rms);
    }
 
    if (enc) {
      inp->pick(enc->kinp);
      enc->_synth();
      kcopy(enc->kout, gen->iwhc, gen->kinp);
    } else {
      inp->pick(gen->kinp);
    }

    gen->_synth();
    kcopy(gen->kout, gen->owhc, ktmp);

    kcopy(ktmp, gen->owhc, dis->kinp);
    dis->_synth();
    ksubvec(kreal, dis->kout, dis->owhc, dis->kout);
    dis->_learn(0);
    kcopy(dis->kinp, dis->iwhc, gen->kout);

    gen->_stats();
    gen->_learn(genmul);

    if (enc) {
      kcopy(gen->kinp, gen->iwhc, enc->kout);
      enc->_stats();
      enc->_learn(genmul);
    } 

    kcopy(ktmp, gen->owhc, dis->kinp);
    dis->_synth();
    ksubvec(kfake, dis->kout, dis->owhc, dis->kout);
    dis->_stats();
    dis->_learn(dismul);

    tgt->pick(dis->kinp);
    dis->_synth();
    ksubvec(kreal, dis->kout, dis->owhc, dis->kout);
    dis->_stats();
    dis->_learn(dismul);

    if (gen->rounds % repint == 0) {
      if (enc) {
        enc->report();
        enc->save();
      }

      gen->report();
      gen->save();

      dis->report();
      dis->save();
    }
  }

  kfree(ktmp);
}

void learnhans(
  Kleption *inp,
  Kleption *tgt,
  Cortex *gen,
  Cortex *dis,
  Cortex *enc,
  int repint,
  double mul,
  bool lossreg,
  double noise
) {

  if (enc) {
    assert(enc->ow == gen->iw);
    assert(enc->oh == gen->ih);
    assert(enc->oc == gen->ic);

    assert(enc->iw == gen->ow);
    assert(enc->ih == gen->oh);
    assert(dis->ic == enc->ic + gen->oc);
  } else {
    assert(gen->iw == gen->ow);
    assert(gen->ih == gen->oh);
    assert(dis->ic == gen->ic + gen->oc);
  }
  assert(dis->iw == gen->ow);
  assert(dis->ih == gen->oh);

  double *ktmp;
  kmake(&ktmp, gen->owhc);

  double *kinp;
  kmake(&kinp, enc ? enc->iwhc : gen->iwhc);

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  double *kreal;
  kmake(&kreal, dis->owhc);
  kfill(kreal, dis->owhc, 0.0);

  double *kfake;
  kmake(&kfake, dis->owhc);
  kfill(kfake, dis->owhc, 1.0);

  while (1) {
    double genmul = mul;
    double dismul = mul;
    if (lossreg) {
      genmul *= (dis->rms > 0.5 ? 0.0 : 2.0 * (0.5 - dis->rms));
      dismul *= (dis->rms > 0.5 ? 1.0 : 2.0 * dis->rms);
    }
 
    Kleption::pick_pair(inp, kinp, tgt, ktgt);
    if (enc) {
      kcopy(kinp, enc->iwhc, enc->kinp);
      enc->_synth();
      kcopy(enc->kout, gen->iwhc, gen->kinp);
    } else {
      kcopy(kinp, gen->iwhc, gen->kinp);
    }

    gen->_synth();
    kcopy(gen->kout, gen->owhc, ktmp);

    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    if (enc) {
      ksplice(kinp, enc->iw * enc->ih, enc->ic, 0, enc->ic, dis->kinp, dis->ic, gen->oc);
    } else {
      ksplice(kinp, gen->iw * gen->ih, gen->ic, 0, gen->ic, dis->kinp, dis->ic, gen->oc);
    }
    // addnoise

    dis->_synth();
    ksubvec(kreal, dis->kout, dis->owhc, dis->kout);
    dis->_learn(0);
    kcopy(dis->kinp, dis->iwhc, gen->kout);

    gen->_stats();
    gen->_learn(genmul);
    if (enc) {
      kcopy(gen->kinp, gen->iwhc, enc->kout);
      enc->_stats();
      enc->_learn(genmul);
    } 

    kcopy(ktmp, gen->owhc, dis->kinp);
    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    if (enc) {
      ksplice(kinp, enc->iw * enc->ih, enc->ic, 0, enc->ic, dis->kinp, dis->ic, gen->oc);
    } else {
      ksplice(kinp, gen->iw * gen->ih, gen->ic, 0, gen->ic, dis->kinp, dis->ic, gen->oc);
    }

    // addnoise
    dis->_synth();
    ksubvec(kfake, dis->kout, dis->owhc, dis->kout);
    dis->_stats();
    dis->_learn(dismul);

    kcopy(ktgt, gen->owhc, dis->kinp);
    ksplice(ktmp, gen->ow * gen->oh, gen->oc, 0, gen->oc, dis->kinp, dis->ic, 0);
    if (enc) {
      ksplice(kinp, enc->iw * enc->ih, enc->ic, 0, enc->ic, dis->kinp, dis->ic, gen->oc);
    } else {
      ksplice(kinp, gen->iw * gen->ih, gen->ic, 0, gen->ic, dis->kinp, dis->ic, gen->oc);
    }
    // addnoise

    dis->_synth();
    ksubvec(kreal, dis->kout, dis->owhc, dis->kout);
    dis->_stats();
    dis->_learn(dismul);

    if (gen->rounds % repint == 0) {
      if (enc) {
        enc->report();
        enc->save();
      }

      gen->report();
      gen->save();

      dis->report();
      dis->save();
    }
  }

  kfree(ktmp);
  kfree(ktgt);
  kfree(kinp);
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  ++argv;
  --argc;

  if (argc < 1) {
    usage();
    return 1;
  }

  std::string cmd;
  parsecmd(&argc, &argv, &cmd);

  Optmap arg(&argc, &argv);
  assert(argc == 0);

  if (cmd == "help" || cmd == "h") {
    usage();
    return 0;
  }

  if (cmd == "new") {
    Cortex::create(arg.get("gen"));
    return 0;
  }

  if (cmd == "dump") {
    Cortex *gen = new Cortex(arg.get("gen"));
    gen->dump(stdout);
    delete gen;
    return 0;
  }

  if (cmd == "scram") {
    Cortex *gen = new Cortex(arg.get("gen"));
    double dev = arg.get("dev", "1.0");
    gen->scram(dev);
    delete gen;
    return 0;
  }

  if (cmd == "push") {
    std::string t = arg.get("t");
    int ic = arg.get("ic", "0");
    int oc = arg.get("oc", "0");

    if (ic < 0)
      uerror("input channels can't be negative");
    if (oc < 0)
      uerror("output channels can't be negative");

    Cortex *gen = new Cortex(arg.get("gen"));

    if (gen->oc > 0) {
      if (ic == 0)
        ic = gen->oc;
      if (ic != gen->oc)
        uerror("input channels don't match output channels of last layer");
    } else {
      if (ic == 0)
        uerror("number of input channels required (-ic)");
    }
    assert(ic > 0);

    if (oc <= 0) {
      int ric = ic;

      if      (t == "dns1")	{ oc = (ic << 2); }
      else if (t == "dns2")	{ oc = (ic << 4); }
      else if (t == "dns3")	{ oc = (ic << 6); }
      else if (t == "dns4")	{ oc = (ic << 8); }
      else if (t == "ups1")	{ oc = (ic >> 2); ric = (oc << 2); }
      else if (t == "ups2")	{ oc = (ic >> 4); ric = (oc << 4); }
      else if (t == "ups3")	{ oc = (ic >> 6); ric = (oc << 6); }
      else if (t == "ups4")	{ oc = (ic >> 8); ric = (oc << 8); }
      else if (t == "grnd")	{ oc = (ic + 1); }
      else if (t == "lrnd")	{ oc = (ic + 1); }
      else if (t == "pad1")	{ oc = (ic + 1); }
      else                      { oc = ic; }

      if (ric != ic)
        uerror("input channels can't divide to upscale");
    }
    assert(oc > 0);

    gen->push(t, ic, oc);
    delete gen;
    return 0;
  }

  if (cmd == "synth") {
    std::string img = arg.get("img", "/dev/stdin");

    uint8_t *rgb;
    unsigned int w, h;
    load_pic(img, &rgb, &w, &h);

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

    if (enc) {
      enc->prepare(w, h);
      gen->prepare(enc->ow, enc->oh);

      if (enc->oc != gen->ic)
        error("encoder oc doesn't match generator ic");
    } else {
      gen->prepare(w, h);
    }
    if (gen->oc != 3)
      error("generator must have 3 output channels");

    int rgbn = w * h * 3;
    double *drgb = new double[rgbn];
    endub(rgb, rgbn, drgb);

    if (enc) {
      enk(drgb, rgbn, enc->kinp);
      enc->_synth();
      kcopy(enc->kout, enc->owhc, gen->kinp);
    } else {
      enk(drgb, rgbn, gen->kinp);
    }
    delete[] drgb;

    gen->_synth();

    int orgbn = gen->ow * gen->oh * 3;
    double *dorgb = new double[orgbn];
    dek(gen->kout, orgbn, dorgb);
    uint8_t *orgb = new uint8_t[orgbn];
    dedub(dorgb, orgbn, orgb);
    delete[] dorgb;

    save_ppm(stdout, orgb, gen->ow, gen->oh);
    delete[] orgb;

    delete gen;
    if (enc)
      delete enc;
    return 0;
  }

  if (cmd == "learnauto") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int repint = arg.get("repint", "256");
    double mul = arg.get("mul", "1.0");

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

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

    Kleption *inp = new Kleption(arg.get("inp"), iw, ih, ic);

    learnauto(inp, gen, enc, repint, mul);

    delete inp;
    delete gen;
    if (enc)
      delete enc;
    return 0;
  }


  if (cmd == "learnfunc") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int repint = arg.get("repint", "256");
    double mul = arg.get("mul", "1.0");

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

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

    Kleption *inp = new Kleption(arg.get("inp"), iw, ih, ic);
    Kleption *tgt = new Kleption(arg.get("tgt"), gen->ow, gen->oh, gen->oc);

    learnfunc(inp, tgt, gen, enc, repint, mul);

    delete inp;
    delete tgt;
    delete gen;
    if (enc)
      delete enc;
    return 0;
  }

  if (cmd == "learnstyl") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", "256");
    double mul = arg.get("mul", "1.0");
    int lossreg = arg.get("lossreg", "1");

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

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

    Cortex *dis = new Cortex(arg.get("dis"));
    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc)
      error("gen oc doesn't match dis ic");

    Kleption *inp = new Kleption(arg.get("inp"), iw, ih, ic);
    Kleption *tgt = new Kleption(arg.get("tgt"), gen->ow, gen->oh, gen->oc);

    learnstyl(inp, tgt, gen, dis, enc, repint, mul, lossreg);

    delete inp;
    delete tgt;

    delete gen;
    if (enc)
      delete enc;
    delete dis;
    return 0;
  }



  if (cmd == "learnhans") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", "256");
    double mul = arg.get("mul", "1.0");
    int lossreg = arg.get("lossreg", "1");
    double noise = arg.get("noise", "0.33");

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

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

    Cortex *dis = new Cortex(arg.get("dis"));
    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc)
      error("gen oc doesn't match dis ic");

    Kleption *inp = new Kleption(arg.get("inp"), iw, ih, ic);
    Kleption *tgt = new Kleption(arg.get("tgt"), gen->ow, gen->oh, gen->oc);

    learnhans(inp, tgt, gen, dis, enc, repint, mul, lossreg, noise);

    delete inp;
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
