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

  double t0 = now();
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
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();

      char buf[4096];
      sprintf(buf, "genrounds=%lu dt=%lf ", gen->rounds, dt);
      if (enc)
        sprintf(buf + strlen(buf), "encrms=%lf ", enc->rms);
      sprintf(buf + strlen(buf), "genrms=%lf", gen->rms);
      fprintf(stderr, "%s\n", buf);
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

  double t0 = now();

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
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();

      char buf[4096];
      sprintf(buf, "genrounds=%lu dt=%lf ", gen->rounds, dt);
      if (enc)
        sprintf(buf + strlen(buf), "encrms=%lf ", enc->rms);
      sprintf(buf + strlen(buf), "genrms=%lf", gen->rms);
      fprintf(stderr, "%s\n", buf);
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

  double t0 = now();

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

      char buf[4096];
      sprintf(buf, "genrounds=%lu dt=%lf ", gen->rounds, dt);
      if (enc)
        sprintf(buf + strlen(buf), "encrms=%lf ", enc->rms);
      sprintf(buf + strlen(buf), "genrms=%lf disrms=%lf", gen->rms, dis->rms);
      fprintf(stderr, "%s\n", buf);
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

  double t0 = now();

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
      double t1 = now();
      double dt = t1 - t0;
      t0 = t1;

      if (enc)
        enc->save();
      gen->save();
      dis->save();

      char buf[4096];
      sprintf(buf, "genrounds=%lu dt=%lf ", gen->rounds, dt);
      if (enc)
        sprintf(buf + strlen(buf), "encrms=%lf ", enc->rms);
      sprintf(buf + strlen(buf), "genrms=%lf disrms=%lf", gen->rms, dis->rms);
      fprintf(stderr, "%s\n", buf);
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

  bool ctx_is_dir = is_dir(ctx);

  Cmdline arg(argc, argv, "0");

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

  if (cmd == "make") {
    if (ctx_is_dir)
      error("dir exists with name of file");

    std::string spec = arg.get("spec", "/dev/stdin");
    int clobber = arg.get("clobber", "0");

    if (!arg.unused.empty())
      error("unrecognized options");

    Cortex::create(ctx, (bool)clobber);

    Cortex *out = new Cortex(ctx);

    FILE *specfp = fopen(spec.c_str(), "r");
    if (!specfp)
      error(std::string("can't open ") + spec + ": " + strerror(errno));

    std::string specline;
    if (read_line(specfp, &specline)) {
      int specargc;
      char **specargv;
      parseargstrad(specline, &specargc, &specargv);

      Cmdline specargs(specargc, specargv, "");
      int v = specargs.get("v", "0");
      if (v != 2)
        error("missing magic v=2");
      out->head->decay = specargs.get("decay", "1e-2");
      out->head->nu = specargs.get("nu", "1e-4");
      out->head->b1 = specargs.get("b1", "1e-1");
      out->head->b2 = specargs.get("b2", "1e-3");
      out->head->eps = specargs.get("eps", "1e-8");
      out->head->rdev = specargs.get("rdev", "0.25");

      freeargstrad(specargc, specargv);
      if (!specargs.unused.empty())
        error("unrecognized options");
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

      freeargstrad(specargc, specargv);
      if (!specargs.unused.empty())
        error("unrecognized options");

      if (ic == 0)
        ic = poc;
      if (ic == 0)
        error("input channels required");
      if (poc && ic != poc)
        error("input channels don't match output channels of previous layer");

      out->push(type, ic, oc);

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

    if (!arg.unused.empty())
      warning("unrecognized options");

    delete gen;
    gen = new Cortex(ctx);

    printf(
      "rounds=%lu rms=%g max=%g decay=%g nu=%g b1=%g b2=%g eps=%g rdev=%lf stripped=%d\n",
      gen->rounds, gen->rms, gen->max, gen->decay,
      gen->nu, gen->b1, gen->b2, gen->eps, gen->head->rdev, gen->stripped
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

    int limit = arg.get("limit", "-1");
    int reload = arg.get("reload", "0");
    double delay = arg.get("delay", "0.0");

    Kleption::Trav srctrav = Kleption::get_trav(arg.get("srctrav", "scan"));
    if (srctrav == Kleption::TRAV_NONE)
      error("srctrav must be scan or rand or refs");

    Kleption::Flags srcflags = 0;

    int repeat = arg.get("repeat", "0");
    if (repeat)
      srcflags |= Kleption::FLAG_REPEAT;

    std::string outcrop = arg.get("outcrop", "center");
    if (outcrop == "center")
      srcflags |= Kleption::FLAG_CENTER;
    else if (outcrop != "random")
      error("outcrop must be random or center");

    std::string outdim = arg.get("outdim", "0x0x0");
    int pw = 0, ph = 0, pc = 3;
    if (!parsedim(outdim, &pw, &ph, &pc))
      error("bad outdim format");

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

    Kleption *src = new Kleption(
      arg.get("src", "/dev/stdin"),
      pw, ph, pc,
      srcflags, srctrav, srckind,
      sw, sh, sc,
      refsfn
    );

    Cortex *gen = new Cortex(genfn, O_RDONLY);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
    }
    
   

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
    if (outkind == Kleption::KIND_UNK)
      error("unknown outkind");
    if (outkind != Kleption::KIND_SDL && outkind != Kleption::KIND_REF)
      arg.get("out");

    Kleption *out = new Kleption(
      arg.get("out", ""),
      gen->ow, gen->oh, gen->oc,
      Kleption::FLAG_WRITER, Kleption::TRAV_SCAN, outkind
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
    return 0;
  }

  if (cmd == "learnauto") {
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int repint = arg.get("repint", "64");
    double mul = arg.get("mul", "1.0");

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
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

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (ctx_is_dir && fexists(ctx + "/src.dat")) {
      srcfn = ctx + "/src.dat";
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, srckind,
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
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    int iw = arg.get("iw");
    int ih = arg.get("ih");
    int repint = arg.get("repint", "64");
    double mul = arg.get("mul", "1.0");

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
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

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (ctx_is_dir && fexists(ctx + "/src.dat")) {
      srcfn = ctx + "/src.dat";
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string tgtdim = arg.get("tgtdim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", ""));

    std::string tgtfn;
    if (arg.present("tgt")) {
      tgtfn = (std::string)arg.get("tgt");
    } else if (ctx_is_dir && fexists(ctx + "/tgt.dat")) {
      tgtfn = ctx + "/tgt.dat";
    } else {
      arg.get("tgt");
      assert(0);
    }

    Kleption *tgt = new Kleption(
      tgtfn, gen->ow, gen->oh, gen->oc,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, tgtkind,
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
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    int iw = arg.get("iw");
    int ih = arg.get("ih");

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", "64");
    double mul = arg.get("mul", "1.0");
    int lossreg = arg.get("lossreg", "1");

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
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
    } else if (ctx_is_dir && fexists(ctx + "/dis.ctx")) {
      dis = new Cortex(ctx + "/dis.ctx");
    } else {
      error("dis argument required");
    }

    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc)
      error("gen oc doesn't match dis ic");

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (ctx_is_dir && fexists(ctx + "/src.dat")) {
      srcfn = ctx + "/src.dat";
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string stydim = arg.get("stydim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(stydim, &tw, &th, &tc))
      error("bad stydim format");

    Kleption::Kind stykind = Kleption::get_kind(arg.get("stykind", ""));

    std::string styfn;
    if (arg.present("sty")) {
      styfn = (std::string)arg.get("sty");
    } else if (ctx_is_dir && fexists(ctx + "/sty.dat")) {
      styfn = ctx + "/sty.dat";
    } else {
      arg.get("sty");
      assert(0);
    }

    Kleption *sty = new Kleption(
      styfn, gen->ow, gen->oh, gen->oc,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, stykind,
      tw, th, tc
    );

    if (!arg.unused.empty())
      error("unrecognized options");

    learnstyl(src, sty, gen, dis, enc, repint, mul, lossreg);

    delete src;
    delete sty;

    delete gen;
    if (enc)
      delete enc;
    delete dis;
    return 0;
  }



  if (cmd == "learnhans") {
    std::string genfn = ctx_is_dir ? ctx + "/gen.ctx" : ctx;

    int iw = arg.get("iw");
    int ih = arg.get("ih");

    if (iw <= 0)
      uerror("input width must be positive");
    if (ih <= 0)
      uerror("input height must be positive");
    int repint = arg.get("repint", "64");
    double mul = arg.get("mul", "1.0");
    int lossreg = arg.get("lossreg", "1");
    double noise = arg.get("noise", "0.5");

    Cortex *gen = new Cortex(genfn);

    Cortex *enc = NULL;
    if (arg.present("enc")) {
      enc = new Cortex(arg.get("enc"));
    } else if (ctx_is_dir && fexists(ctx + "/enc.ctx")) {
      enc = new Cortex(ctx + "/enc.ctx");
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
    } else if (ctx_is_dir && fexists(ctx + "/dis.ctx")) {
      dis = new Cortex(ctx + "/dis.ctx");
    } else {
      error("dis argument required");
    }

    dis->prepare(gen->ow, gen->oh);
    if (dis->ic != gen->oc + ic)
      error("gen oc+ic doesn't match dis ic");

    std::string srcdim = arg.get("srcdim", "0x0x0");
    int sw = 0, sh = 0, sc = 0;
    if (!parsedim(srcdim, &sw, &sh, &sc))
      error("bad srcdim format");

    Kleption::Kind srckind = Kleption::get_kind(arg.get("srckind", ""));
    if (srckind == Kleption::KIND_UNK)
      error("unknown srckind");

    std::string srcfn;
    if (arg.present("src")) {
      srcfn = (std::string)arg.get("src");
    } else if (ctx_is_dir && fexists(ctx + "/src.dat")) {
      srcfn = ctx + "/src.dat";
    } else {
      arg.get("src");
      assert(0);
    }

    Kleption *src = new Kleption(
      srcfn, iw, ih, ic,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, srckind,
      sw, sh, sc
    );

    std::string tgtdim = arg.get("tgtdim", "0x0x0");
    int tw = 0, th = 0, tc = 0;
    if (!parsedim(tgtdim, &tw, &th, &tc))
      error("bad tgtdim format");

    Kleption::Kind tgtkind = Kleption::get_kind(arg.get("tgtkind", ""));

    std::string tgtfn;
    if (arg.present("tgt")) {
      tgtfn = (std::string)arg.get("tgt");
    } else if (ctx_is_dir && fexists(ctx + "/tgt.dat")) {
      tgtfn = ctx + "/tgt.dat";
    } else {
      arg.get("tgt");
      assert(0);
    }

    Kleption *tgt = new Kleption(
      tgtfn, gen->ow, gen->oh, gen->oc,
      Kleption::FLAG_REPEAT, Kleption::TRAV_RAND, tgtkind,
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
