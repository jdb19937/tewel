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
    TYPE_FILE,
    TYPE_DAT
  };

  enum Flags {
    FLAG_SAVEMEM	= (1 << 0),
    FLAG_ADDGEO		= (1 << 1),
    FLAGS_NONE		= 0
  };

  std::string fn;
  Type type;
  Flags flags;
  unsigned int pw, ph, pc;

  uint8_t *dat;
  unsigned int w, h, c, b;

  std::vector<std::string> paths;
  std::map<std::string, Kleption *> path_sub;

  Kleption(const std::string &_fn, unsigned int _pw, unsigned int _ph, unsigned int _pc = 3, Flags _flags = FLAGS_NONE);
  ~Kleption();

  void _load();
  void _unload();

  std::string pick(uint8_t *rgb);
  void find(const std::string &path, uint8_t *rgb);
};

}

#endif
