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

static int usage() {
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
  return 1;
}

static void warning(const char *str) {
  fprintf(stderr, "Warning: %s\n", str);
}

static void uerror(const char *str) {
  fprintf(stderr, "Error: %s\n", str);
  usage();
  exit(1);
}

static bool parseopt(
  int *argcp, char ***argvp,
  std::string *optp, std::string *valp
) {
  assert(*argcp >= 0);
  if (*argcp == 0)
    return false;

  char *opt = **argvp;
  char *q = strchr(opt, '=');
  if (*opt != '-') {
    if (!q)
      uerror("expected key=value option");
    *q = 0;
    **argvp = q + 1;
  } else {
    while (*opt == '-')
      ++opt;
    if (q) {
      *q = 0;
      **argvp = q + 1;
    } else {
      --*argcp;
      ++*argvp;
    }
  }

  if (*argcp == 0)
    uerror("expected -key value option");

  const char *val = **argvp;
  --*argcp;
  ++*argvp;

  *optp = opt;
  *valp = val;

  return true;
}


static void enkdat(const char *datfn, int iowhc, unsigned int *sampnp, double **kdatp) {
  FILE *datfp = fopen(datfn, "r");
  assert(datfp);
  struct stat stbuf;
  int ret = ::stat(datfn, &stbuf);
  assert(ret == 0);
  size_t datn = stbuf.st_size;
  assert(datn > 0);
  assert(datn % iowhc == 0);
  unsigned int sampn = datn / iowhc;
  assert(sampn > 0);
  uint8_t *bdat = new uint8_t[datn];
  size_t fret = ::fread(bdat, 1, datn, datfp);
  ::fclose(datfp);
  assert(fret == datn);
  double *ddat = new double[datn];
  endub(bdat, datn, ddat);
  delete[] bdat;
  double *kdat = NULL;
  kmake(&kdat, datn);
  assert(kdat);
  enk(ddat, datn, kdat);
  delete[] ddat;

  *sampnp = sampn;
  *kdatp = kdat;
}

#if 0
void foo() {
  Kleption kl0("/home/dan/nova.tagged", 32, 32, 3);
  Kleption kl1("/home/dan/nova.tagged", 32, 32, 3);

  double *kdat0;
  kmake(&kdat0, 32 * 32 * 3);
  double *kdat1;
  kmake(&kdat1, 32 * 32 * 3);

  std::string id = Kleption::pick_pair(&kl0, kdat0, &kl1, kdat1);
  fprintf(stderr, "%s\n", id.c_str());
}
#endif

int main(int argc, char **argv) {
  setbuf(stdout, NULL);

  if (argc < 2)
    return usage();
  ++argv;
  --argc;

  const char *cmd = *argv;
  while (*cmd == '-')
    ++cmd;
  ++argv;
  --argc;

  if (!strcmp(cmd, "help")) {
    usage();
    return 0;
  }

  if (!strcmp(cmd, "new")) {
    if (argc < 1)
      uerror("expected mmfile");
    if (argc > 1)
      warning("ignoring extra arguments");

    const char *fn = *argv;
    Pipeline pipe;
    pipe.create(fn);
    return 0;
  }

  if (!strcmp(cmd, "dump")) {
    if (argc < 1)
      uerror("expected mmfile");
    if (argc > 1)
      warning("ignoring extra arguments");
    const char *fn = *argv;

    Pipeline pipe(fn);
    pipe.dump(stdout);
    return 0;
  }

  if (!strcmp(cmd, "scram")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    --argc;
    ++argv;

    double dev = 1.0;
    if (argc > 0)
      dev = strtod(*argv, NULL);

    Pipeline pipe(fn);
    pipe.scram(dev);
    return 0;
  }

  if (!strcmp(cmd, "push")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;

    std::string stype = "";
    int ic = -1, oc = -1, d = -1, s = -1;

    while (argc > 0) {
      const char *opt = *argv;
      if (*opt != '-')
        uerror("expected option name prefixed by -");
      while (*opt == '-')
        ++opt;
      --argc;
      ++argv;

      if (argc == 0)
        uerror("expected option value");
      const char *val = *argv;
      --argc;
      ++argv;
      
      if (!strcmp(opt, "t")) {
        stype = val;
      } else if (!strcmp(opt, "ic")) {
        ic = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "oc")) {
        oc = (int)strtoul(val, NULL, 0);
      } else {
        uerror("unknown option");
      }
    }

    if (stype == "") {
      uerror("layer type required (-t)");
    }
    if (ic < 0) {
      uerror("number of input channels required (-ic)");
    }
    if (oc < 0) {
      oc = ic;
    }

    Pipeline pipe(fn);
    pipe.push(stype, ic, oc);
    return 0;
  }



  if (!strcmp(cmd, "ppmsynth")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;


    const char *ppmfn;
    if (argc < 1)
      ppmfn = "-";
    else {
      ppmfn = *argv;
      ++argv;
      --argc;
    }
    if (!strcmp(ppmfn, "-"))
      ppmfn = "/dev/stdin";

    uint8_t *rgb;
    unsigned int w, h;
    load_img(ppmfn, &rgb, &w, &h);

    Pipeline pipe(fn);
    pipe.prepare(w, h);
    assert(pipe.iw == w);
    assert(pipe.ih == h);
    assert(pipe.ic == 3);
    assert(pipe.oc == 3);

    int rgbn = w * h * 3;
    double *drgb = new double[rgbn];
    endub(rgb, rgbn, drgb);
    double *kin;
    kmake(&kin, rgbn);
    enk(drgb, rgbn, kin);
    delete[] drgb;

    const double *kout = pipe.synth(kin);

    kfree(kin);

    int orgbn = pipe.ow * pipe.oh * 3;
    double *dorgb = new double[orgbn];
    dek(kout, orgbn, dorgb);
    uint8_t *orgb = new uint8_t[orgbn];
    dedub(dorgb, orgbn, orgb);
    delete[] dorgb;

    save_ppm(stdout, orgb, pipe.ow, pipe.oh);
    delete[] orgb;
    return 0;
  }


  if (!strcmp(cmd, "learnauto")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;

    std::string datfn;
    int iw = -1, ih = -1, ic = 3;
    double nu = 0.0001, b1 = 0.1, b2 = 0.001, eps = 1e-8;
    int nr = 256;

    std::string opt, val;
    while (parseopt(&argc, &argv, &opt, &val)) {
//fprintf(stderr, "[%s=%s]\n", opt.c_str(), val.c_str());
      if      (opt == "dat") datfn = val;
      else if (opt == "nr")  nr = atoi(val);
      else if (opt == "iw")  iw = atoi(val);
      else if (opt == "ih")  ih = atoi(val);
      else if (opt == "ic")  ic = atoi(val);
      else if (opt == "nu")  nu = atod(val);
      else if (opt == "b1")  b1 = atod(val);
      else if (opt == "b2")  b2 = atod(val);
      else if (opt == "eps") eps = atod(val);
      else                   uerror("unknown option");
    }

    if (datfn == "")
      uerror("required option -dat missing");
    if (iw < 0)
      uerror("required option -iw missing");
    if (ih < 0)
      uerror("required option -ih missing");
    if (ic < 0)
      uerror("required option -ic missing");

    int iwhc = iw * ih * ic;

    Pipeline pipe(fn);
    pipe.prepare(iw, ih);

    Kleption klep(datfn, iw, ih, ic);
    double *kin;
    kmake(&kin, iwhc);

    while (1) {
      klep.pick(kin);
      pipe.learnauto(kin);

      if (pipe.rounds % nr == 0) {
        pipe.report();
        pipe.save();
      }
    }

    kfree(kin);
    return 0;
  }



  if (!strcmp(cmd, "learnfunc")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;

    const char *datfn = NULL;
    int iw = -1, ih = -1, ic = 3;
    int ow = -1, oh = -1, oc = -1;
    double nu = 0.0001, b1 = 0.1, b2 = 0.001, eps = 1e-8;
    int nr = 256;

    while (argc > 0) {
      const char *opt = *argv;
      if (*opt != '-')
        uerror("expected option name prefixed by -");
      while (*opt == '-')
        ++opt;
      --argc;
      ++argv;

      if (argc == 0)
        uerror("expected option value");
      const char *val = *argv;
      --argc;
      ++argv;
      
      if (!strcmp(opt, "dat")) {
        datfn = val;
      } else if (!strcmp(opt, "nr")) {
        nr = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "iw")) {
        iw = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ih")) {
        ih = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ic")) {
        ic = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ow")) {
        ow = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "oh")) {
        oh = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "oc")) {
        oc = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "nu")) {
        nu = strtod(val, NULL);
      } else if (!strcmp(opt, "b1")) {
        b1 = strtod(val, NULL);
      } else if (!strcmp(opt, "b2")) {
        b2 = strtod(val, NULL);
      } else if (!strcmp(opt, "eps")) {
        eps = strtod(val, NULL);
      } else {
        uerror("unknown option");
      }
    }

    if (!datfn)
      uerror("required option -dat missing");
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

    Pipeline pipe(fn);
    pipe.prepare(iw, ih);

    unsigned int sampn;
    double *kdat;
    enkdat(datfn, iowhc, &sampn, &kdat);

    const double decay = 1.0 / (double)nr;
    double cerr = 0;
    int i = 0;

    while (1) {
      unsigned int sampi = randuint() % sampn;
      double *kin = kdat + iowhc * sampi;
      double *ktgt = kin + iwhc;
      pipe.learnfunc(kin, ktgt);

      if (pipe.rounds % nr == 0) {
        pipe.report();
        pipe.save();
      }
    }

    kfree(kdat);
    return 0;
  }



  if (!strcmp(cmd, "learnhans")) {
    if (argc < 1)
      uerror("expected mmfile");
    const char *fn = *argv;
    ++argv;
    --argc;

    const char *datfn = NULL;
    const char *disfn = NULL;
    int iw = -1, ih = -1, ic = 3;
    int ow = -1, oh = -1, oc = -1;
    double nu = 0.0001, b1 = 0.1, b2 = 0.001, eps = 1e-8;
    int nr = 256;
    double nz = 0.5;

    while (argc > 0) {
      const char *opt = *argv;
      if (*opt != '-')
        uerror("expected option name prefixed by -");
      while (*opt == '-')
        ++opt;
      --argc;
      ++argv;

      if (argc == 0)
        uerror("expected option value");
      const char *val = *argv;
      --argc;
      ++argv;
      
      if (!strcmp(opt, "dat")) {
        datfn = val;
      } else if (!strcmp(opt, "dis")) {
        disfn = val;
      } else if (!strcmp(opt, "nz")) {
        nz = strtod(val, NULL);
      } else if (!strcmp(opt, "nr")) {
        nr = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "iw")) {
        iw = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ih")) {
        ih = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ic")) {
        ic = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "ow")) {
        ow = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "oh")) {
        oh = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "oc")) {
        oc = (int)strtoul(val, NULL, 0);
      } else if (!strcmp(opt, "nu")) {
        nu = strtod(val, NULL);
      } else if (!strcmp(opt, "b1")) {
        b1 = strtod(val, NULL);
      } else if (!strcmp(opt, "b2")) {
        b2 = strtod(val, NULL);
      } else if (!strcmp(opt, "eps")) {
        eps = strtod(val, NULL);
      } else {
        uerror("unknown option");
      }
    }

    if (!datfn)
      uerror("required option -dat missing");
    if (!disfn)
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
    enkdat(datfn, iowhc, &sampn, &kdat);

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


  uerror("unknown command");
}
