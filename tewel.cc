#define __MAKEMORE_TEWEL_CC__ 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#include <math.h>

#include "colonel.hh"
#include "pipeline.hh"
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


void learnfunc(
  Kleption *inp,
  Kleption *tgt,
  Pipeline *gen,
  Pipeline *enc,
  unsigned int ri,
  double mul
) {
  double *kinp;
  kmake(&kinp, enc ? enc->iwhc : gen->iwhc);

  double *ktgt;
  kmake(&ktgt, gen->owhc);

  while (1) {
    Kleption::pick_pair(inp, kinp, tgt, ktgt);

    if (enc) {
      double *fout = enc->_synth(kinp);

      double *kout = gen->_synth(fout);
      ksubvec(ktgt, kout, gen->owhc, kout);
      gen->_stats(kout);

      assert(gen->iwhc == enc->owhc);
      kcopy(gen->_learn(mul), gen->iwhc, fout);
      enc->_stats(fout);
      enc->_learn(mul);
    } else {
      double *kout = gen->_synth(kinp);
      ksubvec(ktgt, kout, gen->owhc, kout);
      gen->_stats(kout);
      gen->_learn(mul);
    }

    if (gen->rounds % ri == 0) {
      if (enc) {
        enc->report();
        enc->save();
      }

      gen->report();
      gen->save();
    }
  }

  kfree(kinp);
  kfree(ktgt);
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
    Pipeline *gen = new Pipeline;
    gen->create(arg.get("gen"));
    delete gen;
    return 0;
  }

  if (cmd == "dump") {
    Pipeline *gen = new Pipeline(arg.get("gen"));
    gen->dump(stdout);
    delete gen;
    return 0;
  }

  if (cmd == "scram") {
    Pipeline *gen = new Pipeline(arg.get("gen"));
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

    Pipeline *gen = new Pipeline(arg.get("gen"));

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
    load_img(img, &rgb, &w, &h);

    Pipeline *gen = new Pipeline(arg.get("gen"));
    gen->prepare(w, h);

    Pipeline *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Pipeline(encfn);

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
    double *kinp;
    kmake(&kinp, rgbn);
    enk(drgb, rgbn, kinp);
    delete[] drgb;

    const double *kout;
    if (enc) {
      kout = gen->_synth(enc->_synth(kinp));
    } else {
      kout = gen->_synth(kinp);
    }

    kfree(kinp);

    int orgbn = gen->ow * gen->oh * 3;
    double *dorgb = new double[orgbn];
    dek(kout, orgbn, dorgb);
    uint8_t *orgb = new uint8_t[orgbn];
    dedub(dorgb, orgbn, orgb);
    delete[] dorgb;

    save_ppm(stdout, orgb, gen->ow, gen->oh);
    delete[] orgb;

    delete gen;
    return 0;
  }

  if (cmd == "learnfunc") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int ri = arg.get("ri", "256");
    double mul = arg.get("mul", "1.0");

    Pipeline *gen = new Pipeline(arg.get("gen"));

    Pipeline *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Pipeline(encfn);

    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
    } else {
      gen->prepare(iw, ih);
    }

    Kleption *inp;
    if (enc) {
      inp = new Kleption(arg.get("inp"), enc->iw, enc->ih, enc->ic);
    } else {
      inp = new Kleption(arg.get("inp"), gen->iw, gen->ih, gen->ic);
    }

    Kleption *tgt = new Kleption(arg.get("tgt"), gen->ow, gen->oh, gen->oc);

    learnfunc(inp, tgt, gen, enc, ri, mul);

    delete inp;
    delete tgt;
    delete gen;
    return 0;
  }

  if (cmd == "learnstyl") {
    int iw = arg.get("iw");
    int ih = arg.get("ih");

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int ri = arg.get("ri", "256");
    double mul = arg.get("mul", "1.0");

    Pipeline *gen = new Pipeline(arg.get("gen"));

    Pipeline *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Pipeline(encfn);

    if (enc) {
      if (enc->oc != gen->ic)
        error("enc oc doesn't match gen ic");
      enc->prepare(iw, ih);
      gen->prepare(enc->ow, enc->oh);
    } else {
      gen->prepare(iw, ih);
    }

    Pipeline *dis = new Pipeline(arg.get("dis"));
    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc)
      error("gen oc doesn't match dis ic");

#if 0
    Kleption *inp = new Kleption(arg.get("inp"), gen->iw, gen->ih, gen->ic);
    Kleption *tgt = new Kleption(arg.get("tgt"), gen->ow, gen->oh, gen->oc);

    learnfunc(&inp, &tgt, &gen, ri, mul);

    delete inp;
    delete tgt;
#endif

    delete gen;
    delete dis;
    return 0;
  }

  uerror("unknown command " + cmd);
  assert(0);
}
