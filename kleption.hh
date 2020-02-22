#ifndef __MAKEMORE_KLEPTION_HH__
#define __MAKEMORE_KLEPTION_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <map>
#include <random>

namespace makemore {

struct Kleption {
  enum Type {
    TYPE_DIR,
    TYPE_IMG,
    TYPE_DAT
  };

  enum Flags {
    FLAG_LOWMEM		= (1 << 0),
    FLAG_ADDGEO		= (1 << 1),

    FLAGS_NONE		= 0
  };

  std::string fn;
  Type type;
  Flags flags;
  unsigned int pw, ph, pc;

  uint8_t *dat;
  unsigned int w, h, c, b;

  std::vector<std::string> ids;
  std::map<std::string, Kleption *> id_sub;

  Kleption(
    const std::string &_fn,
    unsigned int _pw, unsigned int _ph, unsigned int _pc,
    Flags _flags = FLAGS_NONE
  );
  ~Kleption();

  void _load();
  void _unload();

  std::string pick(double *kdat);
  void find(const std::string &id, double *kdat);

  static std::string pick_pair(
    Kleption *kl0, double *kdat0,
    Kleption *kl1, double *kdat1
  );
};

}

#endif
