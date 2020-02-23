#define __MAKEMORE_KLEPTION_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <dirent.h>

#include "kleption.hh"
#include "random.hh"
#include "youtil.hh"
#include "colonel.hh"

#include <algorithm>

namespace makemore {

static bool is_dir(const std::string &fn) {
}

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

  struct stat buf;
  int ret = ::stat(fn.c_str(), &buf);
  assert(ret == 0);

  if (S_ISDIR(buf.st_mode)) {
    DIR *dp = ::opendir(fn.c_str());
    assert(dp);

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
    std::sort(ids.begin(), ids.end());
  } else {
    if (endswith(fn, ".dat")) {
      type = TYPE_DAT;
    } else {
      type = TYPE_PIC;
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
  case TYPE_PIC:
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
  case TYPE_PIC:
    if (dat)
      return;
    load_pic(fn, &dat, &w, &h);
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

static void _addgeo(double *edat, double x, double y, double w, double h) {
  *edat++ = x / w;
  *edat++ = y / h;
  *edat++ = 1.0 - fabs(2.0 * x / w - 1.0);
  *edat++ = 1.0 - fabs(2.0 * y / h - 1.0);
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
  case TYPE_PIC:
    {
      _load();
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

      if (flags & FLAG_LOWMEM)
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
    qid = "";
  }

  switch (type) {
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

      _load();
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

      _load();
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
  std::string id = kl0->pick(kdat0);
  kl1->find(id, kdat1);
  return id;
}

}
