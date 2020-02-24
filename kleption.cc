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
#include "picpipes.hh"

#include <algorithm>

namespace makemore {

Kleption::Kleption(const std::string &_fn, unsigned int _pw, unsigned int _ph, unsigned int _pc, Flags _flags) {
  fn = _fn;
  flags = _flags;
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

  struct stat buf;
  int ret = ::stat(fn.c_str(), &buf);
  assert(ret == 0);

  idi = 0;
 
  if (S_ISDIR(buf.st_mode)) {
    type = TYPE_DIR;
  } else if (S_ISCHR(buf.st_mode) && ::major(buf.st_rdev) == 81) {
    type = TYPE_CAM;
  } else {
    if (endswith(fn, ".dat")) {
      type = TYPE_DAT;
    } else if (
      endswith(fn, ".mp4") ||
      endswith(fn, ".avi") ||
      endswith(fn, ".mkv")
    ) {
      type = TYPE_VID;
    } else {
      type = TYPE_PIC;
    }
  }
}

Kleption::~Kleption() {
  if (loaded)
    unload();
}

void Kleption::unload() {
  if (!loaded)
    return;

  switch (type) {
  case TYPE_CAM:
    assert(cam);
    delete cam;
    cam = NULL;
    assert(dat);
    delete[] dat;
    dat = NULL;
    w = h = c = 0; b = 0;
    idi = 0;
    break;
  case TYPE_VID:
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
  case TYPE_DIR:
    for (auto psi = id_sub.begin(); psi != id_sub.end(); ++psi) {
      Kleption *subkl = psi->second;
      delete subkl;
    }
    ids.clear();
    id_sub.clear();
    idi = 0;
    break;
  case TYPE_PIC:
  case TYPE_DAT:
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

  switch (type) {
  case TYPE_VID:
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
  case TYPE_CAM:
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
  case TYPE_DIR:
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
          subflags = add_flags(flags, FLAG_NOLOOP);

        Kleption *subkl = new Kleption(fn + "/" + subfn, pw, ph, pc, subflags);
        id_sub.insert(std::make_pair(subfn, subkl));
        ids.push_back(subfn);
      }
  
      ::closedir(dp);

      assert(ids.size() > 0);
      std::sort(ids.begin(), ids.end());
    }
    break;
  case TYPE_PIC:
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
  case TYPE_DAT:
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

again:
  switch (type) {
  case TYPE_VID:
    {
      if (!vidreader)
        return false;

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
        if (flags & FLAG_NOLOOP) {
          delete(vidreader);
          vidreader = NULL;
        } else {
          unload();
          load();
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
  case TYPE_CAM:
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
 
  case TYPE_DIR:
    {
      unsigned int idn = ids.size();
      unsigned int idj;
      if (flags & FLAG_LINEAR) {
        if (idi >= idn) {
          idi = 0;
          if (flags & FLAG_NOLOOP)
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
            if (flags & FLAG_NOLOOP)
              return false;
            idi = 0;
          }

          idj = idi;
          id = ids[idj];
          subkl = id_sub[id];

          ret = subkl->pick(kdat, idp ? &idq : NULL);
          assert(ret);
        }
      } else {
        bool ret = subkl->pick(kdat, idp ? &idq : NULL);
        if (!ret) {
          assert(flags & FLAG_NOLOOP);
          ret = subkl->pick(kdat, idp ? &idq : NULL);
          assert(ret);
        }
      }

      if (idp)
        *idp = id + "/" + idq;
      return true;
    }
  case TYPE_PIC:
    {
      if ((flags & FLAG_NOLOOP) && idi > 0) {
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

      if ((flags & FLAG_LOWMEM) && type == TYPE_PIC)
        unload();

      if (idp) {
        char buf[256];
        sprintf(buf, "%ux%ux%u+%u+%u", pw, ph, pc, x0, y0);
        *idp = buf;
      }

      assert(idi == 0);
      if (flags & FLAG_NOLOOP)
        ++idi;

      return true;
    }
  case TYPE_DAT:
    {
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      if (flags & FLAG_ADDGEO)
        assert(pc == c + 4);
      else
        assert(pc == c);

      unsigned int pwhc = pw * ph * pc;
      unsigned int v = randuint() % b;
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

  switch (type) {
  case TYPE_CAM:
  case TYPE_VID:
    assert(0);
  case TYPE_DIR:
    {
      assert(qid != "");
      Kleption *subkl = id_sub[pid];
      assert(subkl != NULL);
      subkl->find(qid, kdat);
      break;
    }
  case TYPE_PIC:
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
  case TYPE_DAT:
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
