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
    std::string gen = arg.get("gen");
    Pipeline genpipe;
    genpipe.create(gen);
    return 0;
  }

  if (cmd == "dump") {
    std::string gen = arg.get("gen");
    Pipeline genpipe(gen);
    genpipe.dump(stdout);
    return 0;
  }

  if (cmd == "scram") {
    std::string gen = arg.get("gen");
    double dev = arg.get("dev", "1.0");
    Pipeline genpipe(gen);
    genpipe.scram(dev);
    return 0;
  }

  if (cmd == "push") {
    std::string gen = arg.get("gen");
    std::string t = arg.get("t");
    int ic = arg.get("ic", "0");
    int oc = arg.get("oc", "0");

    if (ic < 0)
      uerror("input channels can't be negative");
    if (oc < 0)
      uerror("output channels can't be negative");

    Pipeline genpipe(gen);
    if (genpipe.oc > 0) {
      if (ic == 0)
        ic = genpipe.oc;
      if (ic != genpipe.oc)
        uerror("input channels don't match output channels of last layer");
    } else {
      if (ic == 0)
        uerror("number of input channels required (-ic)");
    }
    assert(ic > 0);

    if (oc <= 0) {
      int ric = ic;

      if      (t == "dns1")	oc = (ic << 2);
      else if (t == "dns2")	oc = (ic << 4);
      else if (t == "dns3")	oc = (ic << 6);
      else if (t == "dns4")	oc = (ic << 8);
      else if (t == "ups1")	{ oc = (ic >> 2); ric = (oc << 2); }
      else if (t == "ups2")	{ oc = (ic >> 4); ric = (oc << 4); }
      else if (t == "ups3")	{ oc = (ic >> 6); ric = (oc << 6); }
      else if (t == "ups4")	{ oc = (ic >> 8); ric = (oc << 8); }
      else if (t == "grnd")	oc = (ic + 1);
      else if (t == "lrnd")	oc = (ic + 1);
      else if (t == "pad1")	oc = (ic + 1);
      else                      oc = ic;

      if (ric != ic)
        uerror("number of input channels can't divide to upscale");
    }
    assert(oc > 0);

    genpipe.push(t, ic, oc);
    return 0;
  }



  if (cmd == "synth") {
    std::string gen = arg.get("gen");
    std::string img = arg.get("img", "/dev/stdin");

    uint8_t *rgb;
    unsigned int w, h;
    load_img(img, &rgb, &w, &h);

    Pipeline genpipe(gen);
    genpipe.prepare(w, h);
    assert(genpipe.iw == w);
    assert(genpipe.ih == h);
    assert(genpipe.ic == 3);
    assert(genpipe.oc == 3);

    int rgbn = w * h * 3;
    double *drgb = new double[rgbn];
    endub(rgb, rgbn, drgb);
    double *kin;
    kmake(&kin, rgbn);
    enk(drgb, rgbn, kin);
    delete[] drgb;

    const double *kout = genpipe.synth(kin);

    kfree(kin);

    int orgbn = genpipe.ow * genpipe.oh * 3;
    double *dorgb = new double[orgbn];
    dek(kout, orgbn, dorgb);
    uint8_t *orgb = new uint8_t[orgbn];
    dedub(dorgb, orgbn, orgb);
    delete[] dorgb;

    save_ppm(stdout, orgb, genpipe.ow, genpipe.oh);
    delete[] orgb;
    return 0;
  }


  if (cmd == "learnauto") {
    std::string gen = arg.get("gen");
    std::string inp = arg.get("inp");
    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int nr = arg.get("nr", "256");
    double nu = arg.get("nu", "1e-4");
    double b1 = arg.get("b1", "1e-1");
    double b2 = arg.get("b1", "1e-3");
    double eps = arg.get("eps", "1e-8");

    Pipeline genpipe(gen);
    genpipe.prepare(iw, ih);

    int ic = genpipe.ic;
    assert(ic > 0);
    int iwhc = iw * ih * ic;

    Kleption klep(inp, iw, ih, ic);
    double *kin;
    kmake(&kin, iwhc);

    while (1) {
      klep.pick(kin);
      genpipe.learnauto(kin);

      if (genpipe.rounds % nr == 0) {
        genpipe.report();
        genpipe.save();
      }
    }

    kfree(kin);
    return 0;
  }



  if (cmd == "learnfunc") {
    std::string gen = arg.get("gen");
    std::string inp = arg.get("inp");
    std::string tgt = arg.get("tgt");
    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int nr = arg.get("nr", "256");
    double nu = arg.get("nu", "1e-4");
    double b1 = arg.get("b1", "1e-1");
    double b2 = arg.get("b1", "1e-3");
    double eps = arg.get("eps", "1e-8");

    Pipeline genpipe(gen);
    genpipe.prepare(iw, ih);

    int ic = genpipe.ic;
    assert(ic > 0);
    int iwhc = iw * ih * ic;

    int ow = genpipe.ow;
    int oh = genpipe.oh;
    int oc = genpipe.oc;
    int owhc = ow * oh * oc;

    Kleption inpklep(inp, iw, ih, ic);
    double *kin;
    kmake(&kin, iwhc);

    Kleption tgtklep(tgt, ow, oh, oc);
    double *ktgt;
    kmake(&ktgt, owhc);

    while (1) {
      genpipe.learnfunc(kin, ktgt);

      if (genpipe.rounds % nr == 0) {
        genpipe.report();
        genpipe.save();
      }
    }

    kfree(kin);
    kfree(ktgt);
    return 0;
  }



#if 0
  if (cmd == "learnhans") {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;

    std::string datfn, disfn;
    int iw = -1, ih = -1, ic = 3;
    int ow = -1, oh = -1, oc = -1;
    double nu = 0.0001, b1 = 0.1, b2 = 0.001, eps = 1e-8;
    int nr = 256;
    double nz = 0.5;

    std::string opt, val;
    while (parseopt(&argc, &argv, &opt, &val)) {
      if      (opt == "dat") datfn = val;
      if      (opt == "dis") disfn = val;
      else if (opt == "nr")  nr = strtoi(val);
      else if (opt == "iw")  iw = strtoi(val);
      else if (opt == "ih")  ih = strtoi(val);
      else if (opt == "ic")  ic = strtoi(val);
      else if (opt == "nu")  nu = strtod(val);
      else if (opt == "b1")  b1 = strtod(val);
      else if (opt == "b2")  b2 = strtod(val);
      else if (opt == "eps") eps = strtod(val);
      else                   uerror("unknown option");
    }

    if (datfn == "")
      uerror("required option -dat missing");
    if (disfn == "")
      uerror("required option -dis missing");
    if (iw < 0)
      uerror("required option -iw missing");
    if (ih < 0)
      uerror("required option -ih missing");
    if (ic < 0)
      uerror("required option -ic missing");
    if (ow < 0)
      ow = iw;
    if (oh < 0)
      oh = ih;
    if (oc < 0)
      oc = ic;
    int iwhc = iw * ih * ic;
    int owhc = ow * oh * oc;
    int iowhc = iwhc + owhc;

    Pipeline genpipe(fn);
    genpipe.prepare(iw, ih);

    Pipeline dispipe(disfn);
    dispipe.prepare(genpipe.ow, genpipe.oh);

    unsigned int sampn;
    double *kdat;
    enkdat(datfn.c_str(), iowhc, &sampn, &kdat);

    const double decay = 1.0 / (double)nr;
    double cerr = 0;
    int i = 0;

    while (1) {
      unsigned int sampi = randuint() % sampn;
      double *kin = kdat + iowhc * sampi;
      double *ktgt = kin + iwhc;

      assert(0);

      if (genpipe.rounds % nr == 0) {
        genpipe.report();
        genpipe.save();
      }
    }

    kfree(kdat);
    return 0;
  }
#endif


  uerror("unknown command");
}
