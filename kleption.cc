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

#include <algorithm>

namespace makemore {

Kleption::Kleption(const std::string &_fn, unsigned int _pw, unsigned int _ph, unsigned int _pc, Flags _flags, Kind _kind) {
  fn = _fn;

  flags = _flags;
  if (
    !(flags & FLAG_WRITER) &&
    !(flags & FLAG_REPEAT) &&
    !(flags & FLAG_LINEAR)
  )
    error("need repeat or linear");

  pw = _pw;
  ph = _ph;
  pc = _pc;

  loaded = false;

  w = 0;
  h = 0;
  c = 0;
  b = 0;
  dat = NULL;

  cam = NULL;
  frames = 0;

  dsp = NULL;

  idi = 0;

  vidreader = NULL;
  vidwriter = NULL;
  datwriter = NULL;

  if (_kind == KIND_SDL) {
    if (!(flags & FLAG_WRITER))
      error("can't input from sdl");
    // assert(fn == "0");
    kind = KIND_SDL;
    dsp = new Display;
    dsp->open();
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

      kind = KIND_DIR;
      return;
    }
    if (_kind == KIND_DAT) {
      kind = KIND_DAT;
      return;
    }
    if (_kind == KIND_VID) {
      if (!(flags & FLAG_WRITER) && !(flags & FLAG_LINEAR))
        error("reading vid kind requires linear flag");
      kind = KIND_VID;
      return;
    }
    if (_kind == KIND_PIC) {
      kind = KIND_PIC;
      return;
    }

    assert(_kind == KIND_UNK);

    if (endswith(fn, ".dat")) {
      kind = KIND_DAT;
    } else if (
      endswith(fn, ".mp4") ||
      endswith(fn, ".avi") ||
      endswith(fn, ".mkv")
    ) {
      if (!(flags & FLAG_WRITER) && !(flags & FLAG_LINEAR))
        error("reading vid kind requires linear flag");
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
      error("can't identify output kind from extension");
    }
    return;
  }
 
  if (S_ISDIR(buf.st_mode)) {
    assert(_kind == KIND_DIR || _kind == KIND_UNK);
    kind = KIND_DIR;
    return;
  } else if (S_ISCHR(buf.st_mode) && ::major(buf.st_rdev) == 81) {
    if (flags & FLAG_WRITER)
      error("can't output to camera");
    assert(_kind == KIND_CAM || _kind == KIND_UNK);
    kind = KIND_CAM;
    return;
  } else {
    if (_kind == KIND_DAT) {
      kind = KIND_DAT;
      return;
    }
    if (_kind == KIND_VID) {
      if (!(flags & FLAG_WRITER) && !(flags & FLAG_LINEAR))
        error("reading vid kind requires linear flag");
      kind = KIND_VID;
      return;
    }
    if (_kind == KIND_PIC) {
      kind = KIND_PIC;
      return;
    }

    if (endswith(fn, ".dat")) {
      kind = KIND_DAT;
    } else if (
      endswith(fn, ".mp4") ||
      endswith(fn, ".avi") ||
      endswith(fn, ".mkv")
    ) {
      if (!(flags & FLAG_WRITER) && !(flags & FLAG_LINEAR))
        error("reading vid kind requires linear flag");
      kind = KIND_VID;
    } else {
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
}

void Kleption::unload() {
  if (!loaded)
    return;

  switch (kind) {
  case KIND_CAM:
    assert(cam);
    delete cam;
    cam = NULL;
    assert(dat);
    delete[] dat;
    dat = NULL;
    w = h = c = 0; b = 0;
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
    w = h = c = 0; b = 0;
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
    assert(dat);
    delete[] dat;
    dat = NULL;
    w = h = c = 0; b = 0;
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
      vidreader = new Picreader;
      vidreader->open(fn);

      assert(!dat);
      unsigned int vw, vh;

      assert(vidreader->read(&dat, &w, &h));

      if (pw == 0)
        pw = w;
      if (ph == 0)
        ph = h;
      if (pc == 0)
        pc = 3 + ((flags & FLAG_ADDGEO) ? 4 : 0);
      assert(pw <= w);
      assert(ph <= h);
      assert(pc == 3 + ((flags & FLAG_ADDGEO) ? 4 : 0));

      c = 3;
      b = 1;
    }
    break;
  case KIND_CAM:
    assert(!cam);
    cam = new Camera(fn, pw, ph);
    cam->open();
    w = cam->w;
    h = cam->h;
    if (pw == 0)
      pw = w;
    if (ph == 0)
      ph = h;
    if (pc == 0)
      pc = 3 + ((flags & FLAG_ADDGEO) ? 4 : 0);

    assert(pw <= w);
    assert(ph <= h);
    assert(pw > 0);
    assert(ph > 0);
    assert(pc == 3 + ((flags & FLAG_ADDGEO) ? 4 : 0));

    assert(!dat);
    dat = new uint8_t[w * h * 3];

    c = 3;
    b = 1;
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

        Flags subflags = flags;
        if (flags & FLAG_LINEAR)
          subflags = sub_flags(flags, FLAG_REPEAT);

        Kleption *subkl = new Kleption(fn + "/" + subfn, pw, ph, pc, subflags);

        if (subkl->kind == KIND_CAM) {
          delete subkl;
          error("unexpected v4l2 device found in directory");
        }

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
    assert(w == 0);
    assert(h == 0);
    {
      Picreader picreader;
      picreader.open(fn);
      picreader.read(&dat, &w, &h);
    }
    c = 3;
    b = 1;

    if (pw == 0)
      pw = w;
    if (ph == 0)
      ph = h;
    if (pc == 0)
      pc = c + ((flags & FLAG_ADDGEO) ? 4 : 0);

    assert(pw <= w);
    assert(ph <= h);
    assert(pc == c + ((flags & FLAG_ADDGEO) ? 4 : 0));

    break;
  case KIND_DAT:
    {
      assert(!dat);
      assert(pw > 0);
      assert(ph > 0);
      assert(pc > 0);

      w = pw;
      h = ph;
      c = pc - ((flags & FLAG_ADDGEO) ? 4 : 0);
      assert(c > 0);

      size_t datn;
      dat = slurp(fn, &datn);
      unsigned int whc = w * h * c;
      assert(datn % whc == 0);

      b = datn / whc;

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

      unsigned int x0 = randuint() % (w - pw + 1);
      unsigned int y0 = randuint() % (h - ph + 1);
      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;

      unsigned int pwhc = pw * ph * pc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      assert(vidreader);
      if (!vidreader->read(dat, w, h)) {
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
        sprintf(buf, "%ux%ux%u+%u+%u+%u", pw, ph, pc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
  case KIND_CAM:
    {
      assert(cam);
      assert(cam->w == pw);
      assert(cam->h == ph);
      if (flags & FLAG_ADDGEO)
        assert(pc == 3 + 4);
      else
        assert(pc == 3);

      cam->read(dat);

      unsigned int x0 = randuint() % (w - pw + 1);
      unsigned int y0 = randuint() % (h - ph + 1);
      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;

      unsigned int pwhc = pw * ph * pc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u+%u", pw, ph, pc, x0, y0, frames);
        *idp = buf;
      }
      ++frames;
      return true;
    }
 
  case KIND_DIR:
    {
      unsigned int idn = ids.size();
      unsigned int idj;
      if (flags & FLAG_LINEAR) {
        if (idi >= idn) {
          idi = 0;
          if (!(flags & FLAG_REPEAT))
            return false;
        }
        idj = idi;
      } else {
        idj = randuint() % idn;
      }
      assert(idj < idn);

      std::string id = ids[idj];
      Kleption *subkl = id_sub[id];
      assert(subkl != NULL);

      std::string idq;
      if (flags & FLAG_LINEAR) {
        bool ret = subkl->pick(kdat, idp ? &idq : NULL);
        if (!ret) {
          ++idi;

          if (idi >= idn) {
            idi = 0;
            if (!(flags & FLAG_REPEAT))
              return false;
          }

          idj = idi;
          id = ids[idj];
          subkl = id_sub[id];

          bool ret = subkl->pick(kdat, idp ? &idq : NULL);
          assert(ret);
        }
      } else {
        bool ret = subkl->pick(kdat, idp ? &idq : NULL);
        assert(ret);
      }

      if (idp)
        *idp = id + "/" + idq;
      return true;
    }
  case KIND_PIC:
    {
      if (!(flags & FLAG_REPEAT) && idi > 0) {
        assert(idi == 1);
        idi = 0;
        return false;
      }

      assert(dat);
      assert(w >= pw);
      assert(h >= ph);
      assert(c == 3);
      if (flags & FLAG_ADDGEO)
        assert(pc == c + 4);
      else
        assert(pc == c);

      unsigned int x0 = randuint() % (w - pw + 1);
      unsigned int y0 = randuint() % (h - ph + 1);
      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;

      unsigned int pwhc = pw * ph * pc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      if ((flags & FLAG_LOWMEM) && kind == KIND_PIC)
        unload();

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u", pw, ph, pc, x0, y0);
        *idp = buf;
      }

      assert(idi == 0);
      if (!(flags & FLAG_REPEAT))
        ++idi;

      return true;
    }
  case KIND_DAT:
    {
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      if (flags & FLAG_ADDGEO)
        assert(pc == c + 4);
      else
        assert(pc == c);

      assert(b > 0);
      unsigned int v;
      if (flags & FLAG_LINEAR) {
        if (idi >= b) {
          idi = 0;
          if (!(flags & FLAG_REPEAT))
            return false;
        }
        v = idi;
      } else {
        v = randuint() % b;
      }
      assert(v < b);

      unsigned int pwhc = pw * ph * pc;
      const uint8_t *tmpdat = dat + v * pwhc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = 0; y < h; ++y)
        for (unsigned int x = 0; x < w; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)*tmpdat++ / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      if (idp) {
        char buf[256];
        sprintf(buf, "%u", v);
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
      assert(w >= pw);
      assert(h >= ph);

      unsigned int vpw, vph, vpc, x0, y0;
      sscanf(pid.c_str(), "%ux%ux%u+%u+%u", &vpw, &vph, &vpc, &x0, &y0);
      assert(vpw == pw);
      assert(vph == ph);
      assert(vpc == pc);
      assert(c == 3);
      if (flags & FLAG_ADDGEO)
        assert(pc == c + 4);
      else
        assert(pc == c);
      assert(x0 >= 0);
      assert(x0 < w);
      assert(y0 >= 0);
      assert(y0 < h);

      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;
      assert(x1 >= x0);
      assert(x1 < w);
      assert(y1 >= y0);
      assert(y1 < h);

      unsigned int pwhc = pw * ph * pc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      break;
    }
  case KIND_DAT:
    {
      assert(qid == "");

      load();
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      if (flags & FLAG_ADDGEO)
        assert(pc == c + 4);
      else
        assert(pc == c);

      assert(isdigit(pid[0]));
      unsigned int v = strtoul(pid.c_str(), NULL, 0);

      unsigned int pwhc = pw * ph * pc;
      const uint8_t *tmpdat = dat + v * pwhc;
      double *ddat = new double[pwhc];
      double *edat = ddat;

      for (unsigned int y = 0; y < h; ++y)
        for (unsigned int x = 0; x < w; ++x) {
          for (unsigned int z = 0; z < c; ++z)
            *edat++ = (double)*tmpdat++ / 255.0;
          if (flags & FLAG_ADDGEO)
            _addgeo(edat, x, y, w, h);
        }

      enk(ddat, pwhc, kdat);
      delete[] ddat;

      break;
    }
  default:
    assert(0);
  }
}

bool Kleption::place(const std::string &id, const double *kdat) {
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
        vidwriter = new Picwriter;
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

      Picwriter *picwriter = new Picwriter;
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
      Picwriter *picwriter = new Picwriter;
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

  default:
    assert(0);
  }
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
