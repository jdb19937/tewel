#ifndef __MAKEMORE_CORTEX_HH__
#define __MAKEMORE_CORTEX_HH__ 1

#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <list>
#include <vector>

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
  int bn;   // batch size; kinp/kout hold bn samples

  uint8_t *kbuf;
  unsigned long kbufn;
  float nu, b1, b2, eps, clip;

  float *kdw;
  unsigned long dwn;

  float *kinp, *kout;

  // style side input (per-sample vector, [bn][sdim]) for styl layers; not owned
  const float *kstyle;
  float *kstylegrad;   // gradient w.r.t. the style, accumulated by learn/accumulate/propback
  int sdim;            // 0 when the net has no styl layers
  std::vector<float *> kstash;   // per-mcon-layer forward outputs kept for backward

  float rms;
  float max;
  float decay;
  uint64_t rounds;

  uint64_t new_rounds;

  float rmul;


  std::list<uint8_t*> bufstack;
  void pushbuf();
  void popbuf();




  Cortex();
  Cortex(const std::string &_fn, int flags = O_RDWR);
  ~Cortex();

  void _clear();

  static void create(const std::string &, bool clobber = false);
  static void set_noise_seed(long seed);           // -1: random noise windows (training); >= 0: deterministic
  static long hash_input(const float *x, long n);  // host-side hash of an input vector, for seeding
  void open(const std::string &_fn, int flags = O_RDWR);
  void close();

  void dump(FILE * = stdout);
  void bindump(FILE *, unsigned int a, unsigned int b);
  void unprepare();
  bool prepare(int _iw, int _ih, int _bn = 1);
  void load();
  void save();

  void unfreeze(int last);

  void push(const std::string &type, int ic, int oc, int viw = 0, int vih = 0, int vow = 0, int voh = 0, float mul = 0.0, int rad = 1, int freeze = 0, int relu = 0, int dim = 2);
  void scram(float dev = 1.0, bool fanin = false);
  void ema(const Cortex &src, float beta);

  void target(const float *ktgt);
  void setloss(const float *ktgt);
  float *synth(const float *_kinp);
  float *synth();
  void stats();
  float *propback();
  float *accumulate();
  float *learn(float mul);
  float *learn(const float *_kout, float mul);

  void _get_head_op(float **vecp, float **matp, int *wp, int *hp);
  void _put_head_op(const float *vec, const float *mat, int w, int h);
  void _get_tail_op(float **vecp, float **matp, int *wp, int *hp);
  void _put_tail_op(const float *vec, const float *mat, int w, int h);

};

}

#endif
