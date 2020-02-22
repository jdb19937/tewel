#ifndef __MAKEMORE_PIPELINE_HH__
#define __MAKEMORE_PIPELINE_HH__ 1

#include <stdio.h>
#include <sys/types.h>
#include <stdint.h>

#include <string>

namespace makemore {

typedef uint32_t layer_header_t[16];

struct Pipeline {
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

  double rms;
  double max;
  double decay;
  uint64_t rounds;

  Pipeline();
  Pipeline(const std::string &_fn);
  ~Pipeline();

  void _clear();

  void create(const std::string &_fn);
  void open(const std::string &_fn);
  void close();

  void dump(FILE * = stdout);
  void unprepare();
  bool prepare(int _iw, int _ih);
  void report(FILE * = stdout);
  void save();

  void push(const std::string &type, int ic, int oc);
  void scram(double dev = 1.0);

  double *synth(const double *kin);

  double *_synth(const double *kin = NULL);
  void _stats(const double *kout);
  double *_learn(double mul = 1.0);

  void learnauto(const double *kintgt, double mul = 1.0);
  void learnfunc(const double *kin, const double *ktgt, double mul = 1.0);
  void learnreal(const double *kfake, const double *kreal, double mul = 1.0);
  void learnstyl(const double *kin, const double *kreal, Pipeline *dis, double mul = 1.0, double dismul = 1.0);
  void learnhans(const double *kin, const double *ktgt, Pipeline *dis, double nz, double mul = 1.0, double dismul = 1.0);

  double *helpfake(const double *kfake);

  void _target_real(double *kout);
  void _target_fake(double *kout);
};

}

#endif
