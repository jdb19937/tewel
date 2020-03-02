#ifndef __MAKEMORE_RANDO_HH__
#define __MAKEMORE_RANDO_HH__ 1

#include <string>

namespace makemore {

struct Rando {
  int dim;
  double *mean, *chol;
  double *sum, *summ;
  int sumn;

  Rando(int _dim = 0);
  ~Rando();

  void load(const std::string &fn);
  void observe(const double *dat);
  void save(const std::string &fn);
  void generate(double *dat, double mul = 1.0);
};

}

#endif
