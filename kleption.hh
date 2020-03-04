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
  static std::string picreader_cmd;
  static void set_picreader_cmd(const std::string &cmd) {
    picreader_cmd = cmd;
  }

  static std::string picwriter_cmd;
  static void set_picwriter_cmd(const std::string &cmd) {
    picwriter_cmd = cmd;
  }

  static std::string vidreader_cmd;
  static void set_vidreader_cmd(const std::string &cmd) {
    vidreader_cmd = cmd;
  }

  static std::string vidwriter_cmd;
  static void set_vidwriter_cmd(const std::string &cmd) {
    vidwriter_cmd = cmd;
  }

  enum Kind {
    KIND_ANY = 0,

    KIND_DIR,
    KIND_PIC,
    KIND_DAT,
    KIND_CAM,
    KIND_VID,
    KIND_SDL,
    KIND_REF,
    KIND_RVG,
    KIND_RND,
    KIND_RGB,

    KIND_UNK = -1,
  };

  static Kind get_kind(const std::string &kindstr) {
    if (kindstr == "")    return KIND_ANY;
    if (kindstr == "dir") return KIND_DIR;
    if (kindstr == "pic") return KIND_PIC;
    if (kindstr == "dat") return KIND_DAT;
    if (kindstr == "cam") return KIND_CAM;
    if (kindstr == "vid") return KIND_VID;
    if (kindstr == "sdl") return KIND_SDL;
    if (kindstr == "ref") return KIND_REF;
    if (kindstr == "rvg") return KIND_RVG;
    if (kindstr == "rnd") return KIND_RND;
    if (kindstr == "rgb") return KIND_RGB;
    return KIND_UNK;
  }

  typedef uint32_t Flags;
  const static Flags FLAG_LOWMEM = (1 << 0);
  const static Flags FLAG_ADDGEO = (1 << 1);
  const static Flags FLAG_REPEAT = (1 << 2);
  const static Flags FLAG_WRITER = (1 << 4);
  const static Flags FLAG_CENTER = (1 << 5);

  typedef enum {
    TRAV_NONE = 0,

    TRAV_RAND,
    TRAV_SCAN,
    TRAV_REFS
  } Trav;

  static Trav get_trav(const std::string &travstr) {
    if (travstr == "rand") return TRAV_RAND;
    if (travstr == "scan") return TRAV_SCAN;
    if (travstr == "refs") return TRAV_REFS;
    return TRAV_NONE;
  }

  std::string fn;
  Kind kind;
  Flags flags;
  Trav trav;
  unsigned int pw, ph, pc;
  unsigned int sw, sh, sc;
  bool loaded;

  // pic, dat
  uint8_t *dat;
  unsigned int b;
  FILE *datwriter;

  // ref
  FILE *refwriter;
  FILE *refreader;

  // cam
  class Camera *cam;
  unsigned int frames;

  // vid
  class Picreader *vidreader;
  class Picwriter *vidwriter;

  // dir
  std::vector<std::string> ids;
  std::map<std::string, Kleption *> id_sub;
  unsigned int idi;

  // sdl
  class Display *dsp;

  // rvg
  class Rando *rvg;

  Kleption(
    const std::string &_fn,
    unsigned int _pw, unsigned int _ph, unsigned int _pc,
    Flags _flags = 0, Trav _trav = TRAV_RAND, Kind _kind = KIND_UNK,
    unsigned int _sw = 0, unsigned int _sh = 0, unsigned int _sc = 0,
    const char *refsfn = NULL
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
