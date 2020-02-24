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
    TYPE_PIC,
    TYPE_DAT,
    TYPE_CAM,
    TYPE_VID,
    TYPE_SDL,

    TYPE_UNK = -1
  };

  enum Flags {
    FLAG_LOWMEM		= (1 << 0),
    FLAG_ADDGEO		= (1 << 1),
    FLAG_NOLOOP		= (1 << 2),
    FLAG_LINEAR		= (1 << 3),
    FLAG_WRITER		= (1 << 4),

    FLAGS_NONE		= 0
  };

  static Flags add_flags(Flags a, Flags b) {
    return ((Flags)((unsigned long)a | (unsigned long)b));
  }
  static Flags sub_flags(Flags a, Flags b) {
    return ((Flags)((unsigned long)a & ~(unsigned long)b));
  }

  std::string fn;
  Type type;
  Flags flags;
  unsigned int pw, ph, pc;
  bool loaded;

  // pic, dat
  uint8_t *dat;
  unsigned int w, h, c, b;

  // cam
  class Camera *cam;
  unsigned int frames;

  // vid
  class Picreader *vidreader;

  // dir
  std::vector<std::string> ids;
  std::map<std::string, Kleption *> id_sub;
  unsigned int idi;

  // sdl
  class Display *dsp;

  Kleption(
    const std::string &_fn,
    unsigned int _pw, unsigned int _ph, unsigned int _pc,
    Flags _flags = FLAGS_NONE
  );
  ~Kleption();

  void load();
  void unload();

  void reset();

  bool pick(double *kdat, std::string *idp = NULL);
  void find(const std::string &id, double *kdat);

  static std::string pick_pair(
    Kleption *kl0, double *kdat0,
    Kleption *kl1, double *kdat1
  );

  bool place(const std::string &id, const double *kdat);
};

}

#endif
