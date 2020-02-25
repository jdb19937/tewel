#ifndef __MAKEMORE_CORTEX_HH__
#define __MAKEMORE_CORTEX_HH__ 1

#include <stdio.h>
#include <sys/types.h>
#include <stdint.h>

#include <string>

namespace makemore {

typedef uint32_t layer_header_t[16];

struct Cortex {
  bool is_open, is_prep;
  std::string fn;

  int fd;
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

  Cortex();
  Cortex(const std::string &_fn);
  ~Cortex();

  void _clear();

  static void create(const std::string &);
  void open(const std::string &_fn);
  void close();

  void dump(FILE * = stdout);
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
