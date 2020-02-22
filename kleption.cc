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
      path_sub.insert(std::make_pair(subfn, subkl));
      paths.push_back(subfn);
    }
  
    ::closedir(dp);

    assert(paths.size());
    shuffle(&paths);
  } else {
    assert(errno == ENOTDIR);
    if (endswith(fn, ".dat")) {
      type = TYPE_DAT;
    } else {
      type = TYPE_FILE;
    }
  }
}

Kleption::~Kleption() {
  if (dat)
    delete[] dat;

  for (auto psi = path_sub.begin(); psi != path_sub.end(); ++psi) {
    Kleption *subkl = psi->second;
    delete subkl;
  }
}

void Kleption::_unload() {
  switch (type) {
  case TYPE_FILE:
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
  case TYPE_FILE:
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

std::string Kleption::pick(uint8_t *rgb) {
  switch (type) {
  case TYPE_DIR:
    {
      unsigned int pathn = paths.size();
      unsigned int pathi = randuint() % pathn;
      std::string path = paths[pathi];
      Kleption *subkl = path_sub[path];
      assert(subkl != NULL);
      return (path + "/" + subkl->pick(rgb));
    }
  case TYPE_FILE:
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

      for (unsigned int y = y0; y <= y1; ++y)
        for (unsigned int x = x0; x <= x1; ++x)
          for (unsigned int z = 0; z < c; ++z)
            *rgb++ = dat[z + c * (x + w * y)];

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

      unsigned int whc = w * h * c;
      unsigned int v = randuint() % b;
      memcpy(rgb, dat + v * whc, whc);

      char buf[256];
      sprintf(buf, "%u", v);
      return std::string(buf);
    }
  default:
    assert(0);
  }
}

void Kleption::find(const std::string &path, uint8_t *rgb) {
  const char *cpath = path.c_str();
  std::string ppath, qpath;
  if (const char *p = ::strchr(cpath, '/')) {
    ppath = std::string(cpath, p - cpath);
    qpath = p + 1;
  } else {
    ppath = path;
  }

  switch (type) {
  case TYPE_DIR:
    {
      Kleption *subkl = path_sub[ppath];
      assert(subkl != NULL);
      subkl->find(qpath, rgb);
      break;
    }
  case TYPE_FILE:
    {
      assert(qpath == "");

      _load();
      assert(dat);
      assert(w >= pw);
      assert(h >= ph);

      unsigned int vpw, vph, vpc, x0, y0;
      sscanf(ppath.c_str(), "%ux%ux%u+%u+%u", &vpw, &vph, &vpc, &x0, &y0);
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
        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *rgb++ = dat[z + c * (x + w * y)];

            *rgb++ = (x * 255) / w;
            *rgb++ = (y * 255) / h;
            *rgb++ = 255 - abs(255 * ((int)w - 2 * (int)x) / (int)w);
            *rgb++ = 255 - abs(255 * ((int)h - 2 * (int)y) / (int)h);
          }
      } else {
        for (unsigned int y = y0; y <= y1; ++y)
          for (unsigned int x = x0; x <= x1; ++x)
            for (unsigned int z = 0; z < c; ++z)
              *rgb++ = dat[z + c * (x + w * y)];
      }
      break;
    }
  case TYPE_DAT:
    {
      assert(qpath == "");

      _load();
      assert(dat);
      assert(w == pw);
      assert(h == ph);
      assert(c == pc);

      assert(isdigit(ppath[0]));
      unsigned int v = strtoul(ppath.c_str(), NULL, 0);

      unsigned int whc = w * h * c;
      const uint8_t *tmpdat = dat + v * whc;

      if (flags & FLAG_ADDGEO) {
        for (unsigned int y = 0; y < h; ++y)
          for (unsigned int x = 0; x < w; ++x) {
            for (unsigned int z = 0; z < c; ++z)
              *rgb++ = *tmpdat++;
            *rgb++ = (x * 255) / w;
            *rgb++ = (y * 255) / h;
            *rgb++ = 255 - abs(255 * ((int)w - 2 * (int)x) / (int)w);
            *rgb++ = 255 - abs(255 * ((int)h - 2 * (int)y) / (int)h);
          }
      } else {
        memcpy(rgb, tmpdat, whc);
      }
      break;
    }
  default:
    assert(0);
  }
}

}
