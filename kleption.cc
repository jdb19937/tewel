#define __MAKEMORE_KLEPTION_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>

#include "kleption.hh"
#include "random.hh"
#include "youtil.hh"
#include "colonel.hh"

namespace makemore {

Kleption::Kleption(const std::string &_fn, unsigned int _pw, unsigned int _ph, unsigned int _pc, Flags _flags) {
  w = 0;
  h = 0;
  c = 0;
  b = 1;
  dat = NULL;

  fn = _fn;
  flags = _flags;
  pw = _pw;
  assert(pw > 0);
  ph = _ph;
  assert(ph > 0);
  pc = _pc;
  assert(pc > 0);

  struct dirent *de;
  DIR *dp = ::opendir(fn.c_str());

  if (dp) {
    type = TYPE_DIR;

    while ((de = ::readdir(dp))) {
      if (*de->d_name == '.')
        continue;
      std::string subfn = de->d_name;

      Kleption *subkl = new Kleption(fn + "/" + subfn, pw, ph, pc);
      id_sub.insert(std::make_pair(subfn, subkl));
      ids.push_back(subfn);
    }
  
    ::closedir(dp);

    assert(ids.size());
    shuffle(&ids);
  } else {
    assert(errno == ENOTDIR);
    if (endswith(fn, ".dat")) {
      type = TYPE_DAT;
    } else {
      type = TYPE_IMG;
    }
  }
}

Kleption::~Kleption() {
  if (dat)
    delete[] dat;

  for (auto psi = id_sub.begin(); psi != id_sub.end(); ++psi) {
    Kleption *subkl = psi->second;
    delete subkl;
  }
}

void Kleption::_unload() {
  switch (type) {
  case TYPE_IMG:
  case TYPE_DAT:
    if (!dat)
      return;
    delete[] dat;
    dat = NULL;
    w = h = c = 0; b = 1;
    break;
  }
}

void Kleption::_load() {
  switch (type) {
  case TYPE_IMG:
    if (dat)
      return;
    load_img(fn, &dat, &w, &h);
    c = 3;
    b = 1;
    break;
  case TYPE_DAT:
    {
      if (dat)
        return;

      w = pw;
      h = ph;
      c = pc;

      size_t datn;
      dat = slurp(fn, &datn);
      unsigned int whc = w * h * c;
      assert(datn % whc == 0);

      b = datn / whc;

      break;
    }
  default:
    break;
  }
}

std::string Kleption::pick(double *kdat) {
  switch (type) {
  case TYPE_DIR:
    {
      unsigned int idn = ids.size();
      unsigned int idi = randuint() % idn;
      std::string id = ids[idi];
      Kleption *subkl = id_sub[id];
      assert(subkl != NULL);
      return (id + "/" + subkl->pick(kdat));
    }
  case TYPE_IMG:
    {
      _load();
      assert(dat);
      assert(w >= pw);
      assert(h >= ph);
      assert(c == pc);
      assert(c == 3);

      unsigned int x0 = randuint() % (w - pw + 1);
      unsigned int y0 = randuint() % (h - ph + 1);
      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;

      if (flags & FLAG_ADDGEO) {
        unsigned int pwhc4 = pw * ph * (pc + 4);
        double *ddat = new double[pwhc4];
        double *edat = ddat;

        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;

            *edat++ = (double)x / (double)w;
            *edat++ = (double)y / (double)h;
            *edat++ = 1.0 - fabs(2.0 * (double)x / (double)w - 1.0);
            *edat++ = 1.0 - fabs(2.0 * (double)y / (double)h - 1.0);
          }

        enk(ddat, pwhc4, kdat);
        delete[] ddat;
      } else {
        unsigned int pwhc = pw * ph * pc;
        double *ddat = new double[pwhc];
        double *edat = ddat;

        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x)
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;

        enk(ddat, pwhc, kdat);
        delete[] ddat;
      }

      if (flags & FLAG_SAVEMEM)
        _unload();

      char buf[256];
      sprintf(buf, "%ux%ux%u+%u+%u", pw, ph, pc, x0, y0);
      return std::string(buf);
    }
  case TYPE_DAT:
    {
      _load();
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      assert(c == pc);

      unsigned int pwhc = pw * ph * pc;
      unsigned int v = randuint() % b;
      const uint8_t *tmpdat = dat + v * pwhc;

      if (flags & FLAG_ADDGEO) {
        unsigned int pwhc4 = pw * ph * (pc + 4);
        double *ddat = new double[pwhc4];
        double *edat = ddat;

        for (unsigned int y = 0; y < h; ++y)
          for (unsigned int x = 0; x < w; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)*tmpdat++ / 255.0;

            *edat++ = (double)x / (double)w;
            *edat++ = (double)y / (double)h;
            *edat++ = 1.0 - fabs(2.0 * (double)x / (double)w - 1.0);
            *edat++ = 1.0 - fabs(2.0 * (double)y / (double)h - 1.0);
          }

        enk(ddat, pwhc4, kdat);
        delete[] ddat;
      } else {
        double *ddat = new double[pwhc];
        double *edat = ddat;

        for (unsigned int i = 0; i < pwhc; ++i)
          *edat++ = (double)*tmpdat++ / 255.0;

        enk(ddat, pwhc, kdat);
        delete[] ddat;
      }

      char buf[256];
      sprintf(buf, "%u", v);
      return std::string(buf);
    }
  default:
    assert(0);
  }
}

void Kleption::find(const std::string &id, double *kdat) {
  const char *cid = id.c_str();
  std::string pid, qid;
  if (const char *p = ::strchr(cid, '/')) {
    pid = std::string(cid, p - cid);
    qid = p + 1;
  } else {
    pid = id;
  }

  switch (type) {
  case TYPE_DIR:
    {
      Kleption *subkl = id_sub[pid];
      assert(subkl != NULL);
      subkl->find(qid, kdat);
      break;
    }
  case TYPE_IMG:
    {
      assert(qid == "");

      _load();
      assert(dat);
      assert(w >= pw);
      assert(h >= ph);

      unsigned int vpw, vph, vpc, x0, y0;
      sscanf(pid.c_str(), "%ux%ux%u+%u+%u", &vpw, &vph, &vpc, &x0, &y0);
      assert(vpw == pw);
      assert(vph == ph);
      assert(vpc == pc);
      assert(pc == 3);
      assert(x0 > 0);
      assert(x0 < w);
      assert(y0 > 0);
      assert(y0 < h);

      unsigned int x1 = x0 + pw - 1;
      unsigned int y1 = y0 + ph - 1;
      assert(x1 >= x0);
      assert(x1 < w);
      assert(y1 >= y0);
      assert(y1 < h);

      if (flags & FLAG_ADDGEO) {
        unsigned int pwhc4 = pw * ph * (pc + 4);
        double *ddat = new double[pwhc4];
        double *edat = ddat;

        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;

            *edat++ = (double)x / (double)w;
            *edat++ = (double)y / (double)h;
            *edat++ = 1.0 - fabs(2.0 * (double)x / (double)w - 1.0);
            *edat++ = 1.0 - fabs(2.0 * (double)y / (double)h - 1.0);
          }

        enk(ddat, pwhc4, kdat);
        delete[] ddat;
      } else {
        unsigned int pwhc = pw * ph * pc;
        double *ddat = new double[pwhc];
        double *edat = ddat;

        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x)
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)dat[z + c * (x + w * y)] / 255.0;

        enk(ddat, pwhc, kdat);
        delete[] ddat;
      }

      break;
    }
  case TYPE_DAT:
    {
      assert(qid == "");

      _load();
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      assert(c == pc);

      assert(isdigit(pid[0]));
      unsigned int v = strtoul(pid.c_str(), NULL, 0);

      unsigned int pwhc = pw * ph * pc;
      const uint8_t *tmpdat = dat + v * pwhc;

      if (flags & FLAG_ADDGEO) {
        unsigned int pwhc4 = pw * ph * (pc + 4);
        double *ddat = new double[pwhc4];
        double *edat = ddat;

        for (unsigned int y = 0; y < h; ++y)
          for (unsigned int x = 0; x < w; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *edat++ = (double)*tmpdat++ / 255.0;

            *edat++ = (double)x / (double)w;
            *edat++ = (double)y / (double)h;
            *edat++ = 1.0 - fabs(2.0 * (double)x / (double)w - 1.0);
            *edat++ = 1.0 - fabs(2.0 * (double)y / (double)h - 1.0);
          }

        enk(ddat, pwhc4, kdat);
        delete[] ddat;
      } else {
        double *ddat = new double[pwhc];
        double *edat = ddat;

        for (unsigned int i = 0; i < pwhc; ++i)
          *edat++ = (double)*tmpdat++ / 255.0;

        enk(ddat, pwhc, kdat);
        delete[] ddat;
      }

      break;
    }
  default:
    assert(0);
  }
}

}
