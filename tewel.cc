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
    "        tewel synth gen.mm in.ppm > out.ppm\n"
    "                input ppm to a generator\n"
    "        tewel learnauto gen.mm -src samples.dat ...\n"
    "                train generator on sample paired with self\n"
    "        tewel learnfunc gen.mm -src samples.dat ...\n"
    "                train generator on paired samples\n"
    "        tewel learnstyl gen.mm -src samples.dat -dis dis.mm ...\n"
    "                train generator on paired samples versus discriminator\n"
    "        tewel learnhans gen.mm -src samples.dat -dis dis.mm ...\n"
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
  Kleption *src,
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
  Kleption *src,
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

    assert(tgt->pick(dis->kinp));
    dis->synth();
    dis->target(kreal);
    dis->learn(dismul);

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
  Kleption *src,
  Kleption *tgt,
  Cortex *gen,
  Cortex *dis,
  Cortex *enc,
  int repint,
  double mul,
  bool lossreg,
  double noise
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

  while (1) {
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

  Cmdline arg(&argc, &argv, "gen");
  std::string cmd = arg.cmd;

  setkdev(arg.get("cuda", kndevs() > 1 ? "1" : "0"));
  setkbs(arg.get("kbs", "256"));

  Picreader::set_cmd(arg.get("picreader", "/opt/makemore/share/tewel/picreader.pl"));
  Picwriter::set_cmd(arg.get("picwriter", "/opt/makemore/share/tewel/picwriter.pl"));

  if (cmd == "help" || cmd == "h") {
    usage();
    return 0;
  }

  if (cmd == "new") {
    std::string genfn = arg.get("gen");
    if (!arg.unused.empty())
      error("unrecognized options");
    Cortex::create(genfn);
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
    if (!arg.unused.empty())
      error("unrecognized options");

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
      else if (t == "iden")	{ oc = ic; }
      else                      { oc = ic; }

      if (ric != ic)
        uerror("input channels can't divide to upscale");
    }
    assert(oc > 0);

    if (!arg.unused.empty())
      error("unrecognized options");

    gen->push(t, ic, oc);
    delete gen;
    return 0;
  }

  if (cmd == "synth") {
    int linear = arg.get("linear", "1");
    linear = linear ? 1 : 0;

    int repeat = arg.get("repeat", "0");
    repeat = repeat ? 1 : 0;

    int limit = arg.get("limit", "-1");

    std::string outcrop = arg.get("outcrop", "center");
    if (outcrop != "random" && outcrop != "center")
      error("outcrop must be random or center");

    double delay = arg.get("delay", "0.0");

    Kleption::Flags srcflags = Kleption::FLAGS_NONE;
    if (repeat)
      srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_REPEAT);
    if (linear)
      srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_LINEAR);
    if (outcrop == "center")
      srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_CENTER);

    std::string outdim = arg.get("outdim", "0x0x0");
    int pw = 0, ph = 0, pc = 3;
    if (!parsedim(outdim, &pw, &ph, &pc))
      error("bad outdim format");

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");
    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    Kleption *src = new Kleption(
      arg.get("src", "/dev/stdin"),
      pw, ph, pc,
      srcflags, srckind,
      sw, sh, sc
    );

    Cortex *gen = new Cortex(arg.get("gen"));

    Cortex *enc = NULL;
    std::string encfn = arg.get("enc", "");
    if (encfn != "")
      enc = new Cortex(encfn);

    src->load();
    assert(src->pw > 0);
    assert(src->ph > 0);
    assert(src->pc == 3);

    int ic;
    if (enc) {
      enc->prepare(src->pw, src->ph);
      if (enc->ic != 3)
        error("encoder must have ic=3");

      gen->prepare(enc->ow, enc->oh);
      if (enc->oc != gen->ic)
        error("encoder oc doesn't match generator ic");
      ic = enc->ic;
    } else {
      gen->prepare(src->pw, src->ph);
      if (gen->ic != 3)
        error("generator must have ic=3");
      ic = gen->ic;
    }

    Kleption::Kind outkind = Kleption::get_kind(arg.get("outkind", ""));
    Kleption *out = new Kleption(
      outkind == Kleption::KIND_SDL ? arg.get("out", "") : arg.get("out"),
      gen->ow, gen->oh, gen->oc, Kleption::FLAG_WRITER, outkind
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
      ++i;
    }


    delete gen;
    if (enc)
      delete enc;
    delete src;
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

    Kleption::Flags srcflags = Kleption::FLAGS_NONE;
    srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_REPEAT);

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));

    Kleption *src = new Kleption(
      arg.get("src"), iw, ih, ic,
      srcflags, srckind,
      sw, sh, sc
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    learnauto(src, gen, enc, repint, mul);

    delete src;
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

    Kleption::Flags srcflags = Kleption::FLAGS_NONE;
    srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_REPEAT);

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));

    Kleption *src = new Kleption(
      arg.get("src"), iw, ih, ic,
      srcflags, srckind,
      sw, sh, sc
    );

    Kleption::Flags tgtflags = Kleption::FLAGS_NONE;
    tgtflags = Kleption::add_flags(tgtflags, Kleption::FLAG_REPEAT);

    std::string tgtdim = arg.get("tgtdim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", ""));

    Kleption *tgt = new Kleption(
      arg.get("tgt"), gen->ow, gen->oh, gen->oc,
      tgtflags, tgtkind,
      tw, th, tc
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    learnfunc(src, tgt, gen, enc, repint, mul);

    delete src;
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

    Kleption::Flags srcflags = Kleption::FLAGS_NONE;
    srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_REPEAT);

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));

    Kleption *src = new Kleption(
      arg.get("src"), iw, ih, ic,
      srcflags, srckind,
      sw, sh, sc
    );

    Kleption::Flags tgtflags = Kleption::FLAGS_NONE;
    tgtflags = Kleption::add_flags(tgtflags, Kleption::FLAG_REPEAT);

    std::string tgtdim = arg.get("tgtdim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", ""));

    Kleption *tgt = new Kleption(
      arg.get("tgt"), gen->ow, gen->oh, gen->oc,
      tgtflags, tgtkind,
      tw, th, tc
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    learnstyl(src, tgt, gen, dis, enc, repint, mul, lossreg);

    delete src;
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
    if (dis->ic != gen->oc + gen->ic)
      error("gen oc+ic doesn't match dis ic");

    Kleption::Flags srcflags = Kleption::FLAGS_NONE;
    srcflags = Kleption::add_flags(srcflags, Kleption::FLAG_REPEAT);

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));

    Kleption *src = new Kleption(
      arg.get("src"), iw, ih, ic,
      srcflags, srckind,
      sw, sh, sc
    );

    Kleption::Flags tgtflags = Kleption::FLAGS_NONE;
    tgtflags = Kleption::add_flags(tgtflags, Kleption::FLAG_REPEAT);

    std::string tgtdim = arg.get("tgtdim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", ""));

    Kleption *tgt = new Kleption(
      arg.get("tgt"), gen->ow, gen->oh, gen->oc,
      tgtflags, tgtkind,
      tw, th, tc
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    learnhans(src, tgt, gen, dis, enc, repint, mul, lossreg, noise);

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
