#ifndef __MAKEMORE_CORTEX_HH__
#define __MAKEMORE_CORTEX_HH__ 1

#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <list>

namespace makemore {

struct Cortex {
  struct Head {
    char magic[8];
    uint32_t version;
    uint32_t ___stripped;
    uint64_t rounds;
    double nu, b1, b2, eps;
    double rms, max, decay;
    double rdev;
    double clip;
    char blank[4000];
  };

  struct Segment {
    char magic[8];
    uint32_t dim;
    uint32_t type;
    uint32_t len;
    uint32_t ic, oc;
    uint32_t iw, ih, ow, oh;

    uint32_t rad;
    double mul;

    uint32_t freeze;
    uint32_t relu;
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
  unsigned long kbufn;
  double nu, b1, b2, eps, clip;

  double *kdw;
  unsigned long dwn;

  double *kinp, *kout;

  double rms;
  double max;
  double decay;
  uint64_t rounds;

  uint64_t new_rounds;

  double rmul;


  std::list<uint8_t*> bufstack;
  void pushbuf();
  void popbuf();




  Cortex();
  Cortex(const std::string &_fn, int flags = O_RDWR);
  ~Cortex();

  void _clear();

  static void create(const std::string &, bool clobber = false);
  void open(const std::string &_fn, int flags = O_RDWR);
  void close();

  void dump(FILE * = stdout);
  void bindump(FILE *, unsigned int a, unsigned int b);
  void unprepare();
  bool prepare(int _iw, int _ih);
  void load();
  void save();

  void unfreeze(int last);

  void push(const std::string &type, int ic, int oc, int viw = 0, int vih = 0, int vow = 0, int voh = 0, double mul = 0.0, int rad = 1, int freeze = 0, int relu = 0, int dim = 2);
  void scram(double dev = 1.0);

  void target(const double *ktgt);
  void setloss(const double *ktgt);
  double *synth(const double *_kinp);
  double *synth();
  void stats();
  double *propback();
  double *accumulate();
  double *learn(double mul);
  double *learn(const double *_kout, double mul);

  void _get_head_op(double **vecp, double **matp, int *wp, int *hp);
  void _put_head_op(const double *vec, const double *mat, int w, int h);
  void _get_tail_op(double **vecp, double **matp, int *wp, int *hp);
  void _put_tail_op(const double *vec, const double *mat, int w, int h);

};

}

#endif
