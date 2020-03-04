#define __MAKEMORE_KLEPTION_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <dirent.h>

#include "kleption.hh"
#include "random.hh"
#include "youtil.hh"
#include "colonel.hh"
#include "camera.hh"
#include "display.hh"
#include "picpipes.hh"
#include "rando.hh"

#include <algorithm>

namespace makemore {

std::string Kleption::picreader_cmd;
std::string Kleption::picwriter_cmd;
std::string Kleption::vidreader_cmd;
std::string Kleption::vidwriter_cmd;


Kleption::Kleption(
  const std::string &_fn,
  unsigned int _pw, unsigned int _ph, unsigned int _pc,
  Flags _flags, Trav _trav, Kind _kind,
  unsigned int _sw, unsigned int _sh, unsigned int _sc,
  const char *refsfn
) {
  fn = _fn;

  flags = _flags;
  trav = _trav;

  pw = _pw;
  ph = _ph;
  pc = _pc;
  sw = _sw;
  sh = _sh;
  sc = _sc;

  loaded = false;

  b = 0;
  dat = NULL;

  cam = NULL;
  frames = 0;

  dsp = NULL;

  idi = 0;

  vidreader = NULL;
  vidwriter = NULL;
  datwriter = NULL;
  refwriter = NULL;

  rvg = NULL;

  if (refsfn) {
    if (trav != TRAV_REFS)
      error("refsfn requires trav refs");
    refreader = fopen(refsfn, "r");
    if (!refreader)
      error(std::string("can't open ") + refsfn + ": " + strerror(errno));
  } else
    refreader = NULL;

  if (_kind == KIND_SDL) {
    if (!(flags & FLAG_WRITER))
      error("can't input from sdl");
    // assert(fn == "0");
    kind = KIND_SDL;
    dsp = new Display;
    dsp->open();
    return;
  }
  if (_kind == KIND_REF) {
    if (!(flags & FLAG_WRITER))
      error("can't input from ref");
    kind = KIND_REF;
    return;
  }
  if (_kind == KIND_CAM) {
    if (fn == "")
      fn = "/dev/video0";
    if (flags & FLAG_WRITER)
      error("can't output to camera");
  }

  struct stat buf;
  int ret = ::stat(fn.c_str(), &buf);
  if (ret != 0) {
    if (!(flags & FLAG_WRITER))
      error("failed to stat " + fn + ": " + strerror(errno));

    if (_kind == KIND_DIR) {
      if (0 == ::mkdir(fn.c_str(), 0755))
        warning("created directory " + fn);
      else if (errno != EEXIST)
        error("failed to create directory " + fn + ": " + strerror(errno));
      if (pw == 0 || ph == 0 || pc == 0)
        error("outdim required for dir kind");

      kind = KIND_DIR;
      return;
    }
    if (_kind == KIND_RND) {
      error("can't output to rnd kind");
    }
    if (_kind == KIND_RVG) {
      kind = KIND_RVG;
      return;
    }
    if (_kind == KIND_DAT) {
      kind = KIND_DAT;
      return;
    }
    if (_kind == KIND_RGB) {
      kind = KIND_RGB;
      return;
    }
    if (_kind == KIND_VID) {
      kind = KIND_VID;
      return;
    }
    if (_kind == KIND_PIC) {
      kind = KIND_PIC;
      return;
    }

    assert(_kind == KIND_ANY);

    if (endswith(fn, ".dat")) {
      kind = KIND_DAT;
    } else if (endswith(fn, ".rgb")) {
      kind = KIND_RGB;
    } else if (endswith(fn, ".rvg")) {
      kind = KIND_RVG;
    } else if (
      endswith(fn, ".mp4") ||
      endswith(fn, ".avi") ||
      endswith(fn, ".mkv")
    ) {
      kind = KIND_VID;
    } else if (
      endswith(fn, ".jpg") ||
      endswith(fn, ".png") ||
      endswith(fn, ".pbm") ||
      endswith(fn, ".pgm") ||
      endswith(fn, ".ppm")
    ) {
      kind = KIND_PIC;
    } else {
      warning(std::string("can't identify kind from extension, assuming pic for ") + fn);
      kind = KIND_PIC;
    }
    return;
  }
 
  if (S_ISDIR(buf.st_mode)) {
    assert(_kind == KIND_DIR || _kind == KIND_ANY);
    if (pw == 0 || ph == 0 || pc == 0)
      error("outdim required for dir kind");
    kind = KIND_DIR;
    return;
  } else if (S_ISCHR(buf.st_mode) && ::major(buf.st_rdev) == 81) {
    if (flags & FLAG_WRITER)
      error("can't output to camera");
    if (trav != TRAV_SCAN)
      error("reading cam kind requires scan traversal");
    assert(_kind == KIND_CAM || _kind == KIND_ANY);
    kind = KIND_CAM;
    return;
  } else {
    if (_kind == KIND_DAT) {
      kind = KIND_DAT;
      return;
    }
    if (_kind == KIND_RGB) {
      kind = KIND_RGB;
      return;
    }
    if (_kind == KIND_VID) {
      if (!(flags & FLAG_WRITER))
        if (trav != TRAV_SCAN)
          error("reading vid kind requires scan traversal");
      kind = KIND_VID;
      return;
    }
    if (_kind == KIND_PIC) {
      kind = KIND_PIC;
      return;
    }
    if (_kind == KIND_RVG) {
      kind = KIND_RVG;
      return;
    }
    if (_kind == KIND_RND) {
      kind = KIND_RND;
      return;
    }

    if (endswith(fn, ".dat")) {
      kind = KIND_DAT;
    } else if (endswith(fn, ".rgb")) {
      kind = KIND_RGB;
    } else if (endswith(fn, ".rvg")) {
      kind = KIND_RVG;
    } else if (
      endswith(fn, ".mp4") ||
      endswith(fn, ".avi") ||
      endswith(fn, ".mkv")
    ) {
      if (!(flags & FLAG_WRITER))
        if (trav != TRAV_SCAN)
          error("reading vid kind requires scan traversal");
      kind = KIND_VID;
    } else if (
      endswith(fn, ".jpg") ||
      endswith(fn, ".png") ||
      endswith(fn, ".pbm") ||
      endswith(fn, ".pgm") ||
      endswith(fn, ".ppm")
    ) {
      kind = KIND_PIC;
    } else {
      warning(std::string("can't identify kind from extension, assuming pic for ") + fn);
      kind = KIND_PIC;
    }
    return;
  }
}

Kleption::~Kleption() {
  if (loaded)
    unload();
  if (dsp)
    delete dsp;
  if (vidwriter)
    delete vidwriter;
  if (datwriter)
    fclose(datwriter);
  if (refreader)
    fclose(refreader);
  if (refwriter)
    fclose(refwriter);
  
  if (rvg) {
    assert(flags & FLAG_WRITER);
    rvg->save(fn);
    delete rvg;
  }
}

void Kleption::unload() {
  if (!loaded)
    return;

  switch (kind) {
  case KIND_RND:
    break;
  case KIND_RVG:
    assert(rvg);
    delete rvg;
    rvg = NULL;
    break;
  case KIND_CAM:
    assert(cam);
    delete cam;
    cam = NULL;
    assert(dat);
    delete[] dat;
    dat = NULL;
    b = 0;
    idi = 0;
    break;
  case KIND_VID:
    if (vidreader) {
      delete vidreader;
      vidreader = NULL;
    }
    assert(dat);
    delete[] dat;
    dat = NULL;
    b = 0;
    idi = 0;
    break;
  case KIND_DIR:
    for (auto psi = id_sub.begin(); psi != id_sub.end(); ++psi) {
      Kleption *subkl = psi->second;
      delete subkl;
    }
    ids.clear();
    id_sub.clear();
    idi = 0;
    break;
  case KIND_PIC:
  case KIND_DAT:
  case KIND_RGB:
    assert(dat);
    delete[] dat;
    dat = NULL;
    b = 0;
    idi = 0;
    break;
  }

  loaded = false;
}

void Kleption::load() {
  if (loaded)
    return;

  switch (kind) {
  case KIND_VID:
    {
      assert(!vidreader);
      vidreader = new Picreader(vidreader_cmd);
      vidreader->open(fn);

      assert(!dat);
      unsigned int w, h;

      assert(vidreader->read(&dat, &w, &h));

      if (w != sw) {
        if (sw)
          error("vid width doesn't match");
        sw = w;
      }
      if (h != sh) {
        if (sh)
          error("vid height doesn't match");
        sh = h;
      }
      if (3 != sc) {
        if (sc)
          error("vid channels doesn't match");
        sc = 3;
      }

      if (pw == 0)
        pw = sw;
      if (ph == 0)
        ph = sh;
      if (pc == 0)
        pc = 3 + ((flags & FLAG_ADDGEO) ? 4 : 0);

      assert(pw <= w);
      assert(ph <= h);
      assert(pc == 3 + ((flags & FLAG_ADDGEO) ? 4 : 0));

      b = 1;
    }
    break;
  case KIND_RND:
    {
      if (pw == 0)
        pw = sw;
      if (ph == 0)
        ph = sh;
      if (pc == 0)
        pc = sc + ((flags & FLAG_ADDGEO) ? 4 : 0);

      if (sw == 0)
        sw = pw;
      if (sh == 0)
        sh = ph;
      if (sc == 0)
        sc = pc - ((flags & FLAG_ADDGEO) ? 4 : 0);

      assert(pw <= sw);
      assert(ph <= sh);
      assert(pw > 0);
      assert(ph > 0);
      assert(pc == sc + ((flags & FLAG_ADDGEO) ? 4 : 0));
    }
    break;
  case KIND_RVG:
    {
      if (pw == 0)
        pw = sw;
      if (ph == 0)
        ph = sh;
      if (pc == 0)
        pc = sc + ((flags & FLAG_ADDGEO) ? 4 : 0);

      if (sw == 0)
        sw = pw;
      if (sh == 0)
        sh = ph;
      if (sc == 0)
        sc = pc - ((flags & FLAG_ADDGEO) ? 4 : 0);

      assert(pw <= sw);
      assert(ph <= sh);
      assert(pw > 0);
      assert(ph > 0);
      assert(pc == sc + ((flags & FLAG_ADDGEO) ? 4 : 0));

      assert(!rvg);
      rvg = new Rando(sc);
      rvg->load(fn);
    }
    break;
  case KIND_CAM:
    {
      assert(!cam);
      cam = new Camera(fn, sw, sh);
      cam->open();

      if (sw && cam->w != sw)
        error("cam width doesn't match");
      if (sh && cam->h != sh)
        error("cam height doesn't match");
      if (sc && 3 != sh)
        error("cam channels doesn't match");

      sw = cam->w;
      sh = cam->h;
      sc = 3;

      if (pw == 0)
        pw = sw;
      if (ph == 0)
        ph = sh;
      if (pc == 0)
        pc = sc + ((flags & FLAG_ADDGEO) ? 4 : 0);

      assert(pw <= sw);
      assert(ph <= sh);
      assert(pw > 0);
      assert(ph > 0);
      assert(pc == 3 + ((flags & FLAG_ADDGEO) ? 4 : 0));

      assert(!dat);
      dat = new uint8_t[sw * sh * 3];

      b = 1;
    }
    break;
  case KIND_DIR:
    {
      assert(ids.empty());
      assert(id_sub.empty());
      assert(pw > 0);
      assert(ph > 0);
      assert(pc > 0);

      DIR *dp = ::opendir(fn.c_str());
      assert(dp);

      struct dirent *de;
      while ((de = ::readdir(dp))) {
        if (*de->d_name == '.')
          continue;
        std::string subfn = de->d_name;

        Kleption *subkl = new Kleption(
          fn + "/" + subfn, pw, ph, pc,
          flags & ~FLAG_REPEAT, trav, KIND_ANY,
          sw, sh, sc
        );

        if (subkl->kind == KIND_CAM) {
          delete subkl;
          error("v4l2 device found in dir, dir can contain only pics");
        }
        if (subkl->kind == KIND_VID) {
          delete subkl;
          error("vid found in dir, dir can contain only pics");
        }
        assert(
          subkl->kind == KIND_DIR ||
          subkl->kind == KIND_DAT ||
          subkl->kind == KIND_RGB ||
          subkl->kind == KIND_PIC
        );

        id_sub.insert(std::make_pair(subfn, subkl));
        ids.push_back(subfn);
      }
  
      ::closedir(dp);

      assert(ids.size() > 0);
      std::sort(ids.begin(), ids.end());
    }
    break;
  case KIND_PIC:
    assert(!dat);
    unsigned int w, h;
    {
      Picreader picreader(picreader_cmd);
      picreader.open(fn);
      picreader.read(&dat, &w, &h);
    }
    b = 1;

    if (w != sw) {
      if (sw)
        error("pic width doesn't match");
      sw = w;
    }
    if (h != sh) {
      if (sh)
        error("pic height doesn't match");
      sh = h;
    }
    if (3 != sc) {
      if (sc)
        error("pic channels doesn't match");
      sc = 3;
    }

    if (pw == 0)
      pw = sw;
    if (ph == 0)
      ph = sh;
    if (pc == 0)
      pc = sc + ((flags & FLAG_ADDGEO) ? 4 : 0);

    assert(pw <= sw);
    assert(ph <= sh);
    assert(pc == sc + ((flags & FLAG_ADDGEO) ? 4 : 0));

    assert(pw > 0);
    assert(sw > 0);
    assert(ph > 0);
    assert(sh > 0);
    assert(pc > 0);
    assert(sc == 3);

    break;
  case KIND_RGB:
  case KIND_DAT:
    {
      assert(!dat);
      if (!sw) {
        sw = pw;
        if (!sw)
          error("dat width required");
      }
      if (!sh) {
        sh = ph;
        if (!sh)
          error("dat height required");
      }
      if (!sc) {
        if (flags & FLAG_ADDGEO) {
          if (pc == 0)
            error("dat channels required");
          if (pc <= 4)
            error("too few out channels for addgeo");
          sc = pc - 4;
        } else {
          sc = pc;
          if (!sc)
            error("dat channels required");
        }
      }

      if (pw == 0)
        pw = sw;
      if (ph == 0)
        ph = sh;
      if (pc == 0)
        pc = sc + ((flags & FLAG_ADDGEO) ? 4 : 0);

      size_t datn;
      dat = slurp(fn, &datn);
      unsigned int swhc = sw * sh * sc;
      assert(datn % swhc == 0);

      b = datn / swhc;

      break;
    }
  default:
    assert(0);
  }

  loaded = true;
}

static void _addgeo(double *edat, double x, double y, double w, double h) {
  *edat++ = x / w;
  *edat++ = y / h;
  *edat++ = 1.0 - fabs(2.0 * x / w - 1.0);
  *edat++ = 1.0 - fabs(2.0 * y / h - 1.0);
}

static void _outcrop(
  const uint8_t *tmpdat, double *kdat, Kleption::Flags flags,
  int sw, int sh, int sc, int pw, int ph, int pc,
  unsigned int *x0p, unsigned int *y0p
) {
  assert(pw <= sw);
  assert(ph <= sh);
  assert(pc == sc + ((flags & Kleption::FLAG_ADDGEO) ? 4 : 0));

  unsigned int x0, y0, x1, y1;
  if (flags & Kleption::FLAG_CENTER) {
    x0 = (sw - pw) / 2;
    y0 = (sh - ph) / 2;
  } else {
    x0 = randuint() % (sw - pw + 1);
    y0 = randuint() % (sh - ph + 1);
  }
  x1 = x0 + pw - 1;
  y1 = y0 + ph - 1;

  unsigned int pwhc = pw * ph * pc;
  double *ddat = new double[pwhc];
  double *edat = ddat;

  for (unsigned int y = y0; y <= y1; ++y)
    for (unsigned int x = x0; x <= x1; ++x) {
      for (unsigned int z = 0; z < sc; ++z)
        *edat++ = (double)tmpdat[z + sc * (x + sw * y)] / 255.0;
      if (flags & Kleption::FLAG_ADDGEO)
        _addgeo(edat, x, y, sw, sh);
    }

  enk(ddat, pwhc, kdat);
  delete[] ddat;

  if (x0p)
    *x0p = x0;
  if (y0p)
    *y0p = y0;
}
static void _outcrop(
  const double *tmpdat, double *kdat, Kleption::Flags flags,
  int sw, int sh, int sc, int pw, int ph, int pc,
  unsigned int *x0p, unsigned int *y0p
) {
  assert(pw <= sw);
  assert(ph <= sh);
  assert(pc == sc + ((flags & Kleption::FLAG_ADDGEO) ? 4 : 0));

  unsigned int x0, y0, x1, y1;
  if (flags & Kleption::FLAG_CENTER) {
    x0 = (sw - pw) / 2;
    y0 = (sh - ph) / 2;
  } else {
    x0 = randuint() % (sw - pw + 1);
    y0 = randuint() % (sh - ph + 1);
  }
  x1 = x0 + pw - 1;
  y1 = y0 + ph - 1;

  unsigned int pwhc = pw * ph * pc;
  double *ddat = new double[pwhc];
  double *edat = ddat;

  for (unsigned int y = y0; y <= y1; ++y)
    for (unsigned int x = x0; x <= x1; ++x) {
      for (unsigned int z = 0; z < sc; ++z)
        *edat++ = tmpdat[z + sc * (x + sw * y)];
      if (flags & Kleption::FLAG_ADDGEO)
        _addgeo(edat, x, y, sw, sh);
    }

  enk(ddat, pwhc, kdat);
  delete[] ddat;

  if (x0p)
    *x0p = x0;
  if (y0p)
    *y0p = y0;
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

bool Kleption::pick(double *kdat, std::string *idp) {
  load();

  switch (kind) {
  case KIND_VID:
    {
      if (!vidreader) {
        unload();
        load();
        return false;
      }

      unsigned int x0, y0;
      _outcrop(dat, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      assert(vidreader);
      if (!vidreader->read(dat, sw, sh)) {
        if (flags & FLAG_REPEAT) {
          unload();
          load();
        } else {
          delete(vidreader);
          vidreader = NULL;
        }
      }

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u.ppm", pw, ph, sc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
  case KIND_CAM:
    {
      assert(cam);
      if (flags & FLAG_ADDGEO)
        assert(pc == sc + 4);
      else
        assert(pc == sc);

      assert(cam->w == sw);
      assert(cam->h == sh);
      assert(sc == 3);
      cam->read(dat);

      unsigned int x0, y0;
      _outcrop(dat, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u.ppm", pw, ph, sc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
  case KIND_RND:
    {
      unsigned int swh = sw * sh;
      unsigned int swhc = swh * sc;
      double *rnd = new double[swhc];
      for (int j = 0; j < swhc; ++j)
        rnd[j] = randgauss();

      unsigned int x0, y0;
      _outcrop(rnd, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      delete[] rnd;

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u.ppm", pw, ph, sc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
  case KIND_RVG:
    {
      assert(rvg);
      assert(rvg->dim == sc);

      unsigned int swh = sw * sh;
      unsigned int swhc = swh * sc;
      double *rnd = new double[swhc];
      for (int xy = 0; xy < swh; ++xy)
        rvg->generate(rnd + xy * sc);

      unsigned int x0, y0;
      _outcrop(rnd, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      delete[] rnd;

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u.ppm", pw, ph, sc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
 
 
  case KIND_DIR:
    {
      unsigned int idn = ids.size();
      unsigned int idj;
      if (trav == TRAV_SCAN) {
        if (idi >= idn) {
          idi = 0;
          if (!(flags & FLAG_REPEAT))
            return false;
        }
        idj = idi;
      } else if (trav == TRAV_RAND) {
        idj = randuint() % idn;
      } else if (trav == TRAV_REFS) {
        std::string ref;
        if (!read_line(refreader, &ref)) {
          rewind(refreader);
          assert(read_line(refreader, &ref));
          if (!(flags & FLAG_REPEAT)) {
            rewind(refreader);
            return false;
          }
        }
        find(ref, kdat);
        if (idp)
          *idp = ref;
        return true;
      }
      assert(idj < idn);

      std::string id = ids[idj];
      Kleption *subkl = id_sub[id];
      assert(subkl != NULL);

      std::string idq;
      bool ret = subkl->pick(kdat, idp ? &idq : NULL);
      if (!ret) {
        assert(trav == TRAV_SCAN);
        ++idi;

        if (idi >= idn) {
          idi = 0;
          if (!(flags & FLAG_REPEAT))
            return false;
        }

        idj = idi;
        id = ids[idj];
        subkl = id_sub[id];

        ret = subkl->pick(kdat, idp ? &idq : NULL);
        assert(ret);
      }

      if (idp)
        *idp = id + "/" + idq;
      return true;
    }
  case KIND_PIC:
    {
      if (trav == TRAV_SCAN) {
        if (!(flags & FLAG_REPEAT) && idi > 0) {
          assert(idi == 1);
          idi = 0;
          return false;
        }
      } else if (trav == TRAV_REFS) {
        std::string ref;
        if (!read_line(refreader, &ref)) {
          rewind(refreader);
          assert(read_line(refreader, &ref));
          if (!(flags & FLAG_REPEAT)) {
            rewind(refreader);
            return false;
          }
        }
        find(ref, kdat);
        if (idp)
          *idp = ref;
        return true;
      }

      assert(dat);
      assert(sw >= pw);
      assert(sh >= ph);
      assert(sc == 3);
      if (flags & FLAG_ADDGEO)
        assert(pc == sc + 4);
      else
        assert(pc == sc);

      unsigned int x0, y0;
      _outcrop(dat, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      if ((flags & FLAG_LOWMEM) && kind == KIND_PIC)
        unload();

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u.ppm", pw, ph, sc, x0, y0);
        *idp = buf;
      }

      if (trav == TRAV_SCAN) {
        assert(idi == 0);
        if (!(flags & FLAG_REPEAT))
          ++idi;
      }

      return true;
    }
  case KIND_RGB:
  case KIND_DAT:
    {
      assert(dat);
      if (flags & FLAG_ADDGEO)
        assert(pc == sc + 4);
      else
        assert(pc == sc);

      assert(b > 0);
      unsigned int v;
      if (trav == TRAV_SCAN) {
        if (idi >= b) {
          idi = 0;
          if (!(flags & FLAG_REPEAT))
            return false;
        }
        v = idi;
        ++idi;
      } else if (trav == TRAV_REFS) {
        std::string ref;
        if (!read_line(refreader, &ref)) {
          rewind(refreader);
          assert(read_line(refreader, &ref));
          if (!(flags & FLAG_REPEAT)) {
            rewind(refreader);
            return false;
          }
        }
        find(ref, kdat);
        if (idp)
          *idp = ref;
        return true;
      } else if (trav == TRAV_RAND) {
        v = randuint() % b;
      } else {
        assert(0);
      }
      assert(v < b);

      unsigned int swhc = sw * sh * sc;
      const uint8_t *tmpdat = dat + v * swhc;
      unsigned int x0, y0;
      _outcrop(tmpdat, kdat, flags, sw, sh, sc, pw, ph, pc, &x0, &y0);

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u.ppm", pw, ph, sc, x0, y0, v);
        *idp = buf;
      }
      return true;
    }
  default:
    assert(0);
  }

  assert(0);
}

void Kleption::find(const std::string &id, double *kdat) {
  const char *cid = id.c_str();
  std::string pid, qid;
  if (const char *p = ::strchr(cid, '/')) {
    pid = std::string(cid, p - cid);
    qid = p + 1;
  } else {
    pid = id;
    qid = "";
  }

  load();

  switch (kind) {
  case KIND_RND:
  case KIND_RVG:
  case KIND_CAM:
  case KIND_VID:
    assert(0);
  case KIND_DIR:
    {
      assert(qid != "");
      Kleption *subkl = id_sub[pid];
      assert(subkl != NULL);
      subkl->find(qid, kdat);
      break;
    }
  case KIND_PIC:
    {
      assert(qid == "");
      assert(dat);

      unsigned int vpw, vph, vpc, x0, y0;
      sscanf(pid.c_str(), "%ux%ux%u+%u+%u.ppm", &vpw, &vph, &vpc, &x0, &y0);
      assert(sw >= pw);
      assert(sh >= ph);
      assert(vpw == pw);
      assert(vph == ph);
      assert(vpc == pc);
      assert(sc == 3);
      if (flags & FLAG_ADDGEO)
        assert(pc == sc + 4);
      else
        assert(pc == sc);
      assert(x0 >= 0);
      assert(x0 < sw);
      assert(y0 >= 0);
      assert(y0 < sh);

      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;
      assert(x1 >= x0);
      assert(x1 < sw);
      assert(y1 >= y0);
      assert(y1 < sh);

      unsigned int pwhc = pw * ph * pc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < sc; ++z)
            *edat++ = (double)dat[z + sc * (x + sw * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, sw, sh);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      break;
    }

  case KIND_RGB:
  case KIND_DAT:
    {
      assert(dat);
      assert(qid == "");

      unsigned int vpw, vph, vpc, x0, y0, v;
      sscanf(pid.c_str(), "%ux%ux%u+%u+%u+%u.ppm", &vpw, &vph, &vpc, &x0, &y0, &v);
      assert(v < b);
 
      assert(vpw == pw);
      assert(vph == ph);
      assert(vpc == pc);

      assert(sw == pw);
      assert(sh == ph);
      if (flags & FLAG_ADDGEO)
        assert(pc == sc + 4);
      else
        assert(pc == sc);
      assert(x0 >= 0);
      assert(x0 < sw);
      assert(y0 >= 0);
      assert(y0 < sh);

      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;
      assert(x1 >= x0);
      assert(x1 < sw);
      assert(y1 >= y0);
      assert(y1 < sh);

      unsigned int pwhc = pw * ph * pc;
      unsigned int swhc = sw * sh * sc;
      const uint8_t *tmpdat = dat + v * swhc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < sc; ++z)
            *edat++ = (double)tmpdat[z + sc * (x + sw * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, sw, sh);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;
    }
    break;
  default:
    assert(0);
  }
}

bool Kleption::place(const std::string &id, const double *kdat) {
  info(fmt("placing %s", id.c_str()));

  assert(flags & FLAG_WRITER);
  assert(pw > 0);
  assert(ph > 0);
  assert(pc > 0);

  const char *cid = id.c_str();
  std::string pid, qid;
  if (const char *p = ::strchr(cid, '/')) {
    pid = std::string(cid, p - cid);
    qid = p + 1;
  } else {
    pid = id;
    qid = "";
  }

  switch (kind) {
  case KIND_RND:
    error("can't place to kind rnd");
    break;
  case KIND_RVG:
    {
      assert(!(flags & FLAG_ADDGEO));

      if (!rvg)
        rvg = new Rando(pc);

      unsigned int pwh = pw * ph;
      unsigned int pwhc = pwh * pc;
      double *ddat = new double[pwhc];
      dek(kdat, pwhc, ddat);

      for (int xy = 0; xy < pwh; ++xy)
        rvg->observe(ddat + xy * pc);
      delete[] ddat;
    }
    return true;

  case KIND_RGB:
  case KIND_DAT:
    {
      if (!datwriter) {
        datwriter = fopen(fn.c_str(), "w");
        if (!datwriter)
          error(std::string("failed to open ") + fn + ": " + strerror(errno));
      }

      unsigned int pwhc = pw * ph * pc;
      double *dtmp = new double[pwhc];
      uint8_t *tmp = new uint8_t[pwhc];
      dek(kdat, pwhc, dtmp);
      dedub(dtmp, pwhc, tmp);

      fwrite(tmp, 1, pw * ph * pc, datwriter);

      delete[] tmp;
      delete[] dtmp;
    }
    return true;

  case KIND_VID:
    {
      if (!vidwriter) {
        vidwriter = new Picwriter(vidwriter_cmd);
        vidwriter->open(fn);
      }

      unsigned int pwhc = pw * ph * pc;
      double *dtmp = new double[pwhc];
      uint8_t *tmp = new uint8_t[pwhc];
      dek(kdat, pwhc, dtmp);
      dedub(dtmp, pwhc, tmp);

      vidwriter->write(tmp, pw, ph);

      delete[] tmp;
      delete[] dtmp;
    }

    return true;
  case KIND_CAM:
    assert(0);
  case KIND_DIR:
    {
      std::string fnp = fn;
      while (const char *p = ::strchr(qid.c_str(), '/')) {
        fnp += std::string("/" + pid);
        if (0 == ::mkdir(fnp.c_str(), 0755))
          warning("created directory " + fnp);
        else if (errno != EEXIST)
          error("failed to create directory " + fnp + ": " + strerror(errno));

        pid = std::string(qid.c_str(), p - qid.c_str());
        qid = std::string(p + 1);
      }

      Picwriter *picwriter = new Picwriter(picwriter_cmd);
      picwriter->open(fnp + "/" + pid);

      assert(pc == 3);

      unsigned int pwhc = pw * ph * pc;
      double *dtmp = new double[pwhc];
      uint8_t *tmp = new uint8_t[pwhc];
      dek(kdat, pwhc, dtmp);
      dedub(dtmp, pwhc, tmp);

      picwriter->write(tmp, pw, ph);

      delete picwriter;
      delete[] tmp;
      delete[] dtmp;
    }

    return true;

  case KIND_PIC:
    {
      Picwriter *picwriter = new Picwriter(picwriter_cmd);
      picwriter->open(fn);

      unsigned int pwhc = pw * ph * pc;
      double *dtmp = new double[pwhc];
      uint8_t *tmp = new uint8_t[pwhc];
      dek(kdat, pwhc, dtmp);
      dedub(dtmp, pwhc, tmp);

      picwriter->write(tmp, pw, ph);

      delete picwriter;
      delete[] tmp;
      delete[] dtmp;
    }
    return true;

  case KIND_SDL:
    {
      assert(dsp);
      assert(pc == 3);

      unsigned int pwhc = pw * ph * pc;
      double *dtmp = new double[pwhc];
      uint8_t *tmp = new uint8_t[pwhc];
      dek(kdat, pwhc, dtmp);
      dedub(dtmp, pwhc, tmp);

      dsp->update(tmp, pw, ph);
      dsp->present();

      delete[] tmp;
      delete[] dtmp;
    }
    return !dsp->done();

  case KIND_REF:
    if (!refwriter) {
      refwriter = fopen(fn.c_str(), "w");
      if (!refwriter)
        error(std::string("failed to open ") + fn + ": " + strerror(errno));
      setbuf(refwriter, NULL);
    }

    fprintf(refwriter, "%s\n", id.c_str());
    return true;

  default:
    assert(0);
  }
  assert(0);
}

// static 
std::string Kleption::pick_pair(
  Kleption *kl0, double *kdat0,
  Kleption *kl1, double *kdat1
) {
  std::string id;
  bool ret = kl0->pick(kdat0, &id);
  assert(ret);
  kl1->find(id, kdat1);
  return id;
}

}
