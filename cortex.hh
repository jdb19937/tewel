#ifndef __MAKEMORE_CORTEX_HH__
#define __MAKEMORE_CORTEX_HH__ 1

#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>

namespace makemore {

typedef uint32_t layer_header_t[16];

struct Cortex {
  struct Head {
    char magic[8];
    uint32_t version;
    uint32_t stripped;
    uint64_t rounds;
    double nu, b1, b2, eps;
    char blank[4040];
  };

  bool is_open, is_prep;
  std::string fn;

  int fd;
  Head *head;
  uint8_t *base;
  size_t basen;
  uint8_t *kbase;

  int iw, ih, ic, iwhc;
  int ow, oh, oc, owhc;

  uint8_t *kbuf;
  double nu, b1, b2, eps;

  double *kfakebuf;

  double *kinp, *kout;

  double rms;
  double max;
  double decay;
  uint64_t rounds;
  bool stripped;

  uint64_t new_rounds;

  Cortex();
  Cortex(const std::string &_fn, int flags = O_RDWR);
  ~Cortex();

  void _clear();

  static void create(const std::string &);
  void open(const std::string &_fn, int flags = O_RDWR);
  void close();

  void dump(FILE * = stdout, const uint8_t *cutvec = NULL);
  void unprepare();
  bool prepare(int _iw, int _ih);
  void load();
  void save();

  void push(const std::string &type, int ic, int oc);
  void scram(double dev = 1.0);

  void target(const double *ktgt);
  double *synth(const double *_kinp);
  double *synth();
  void stats();
  double *propback();
  double *learn(double mul);
  double *learn(const double *_kout, double mul);
};

}

#endif
