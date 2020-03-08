#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "rando.hh"
#include "youtil.hh"
#include "random.hh"

namespace makemore {

Rando::Rando(int _dim) {
  dim = _dim;
  mean = NULL;
  chol = NULL;
  sum = NULL;
  summ = NULL;
  sumn = 0;
  buf = NULL;
}

Rando::~Rando() {
  if (mean)
    delete[] mean;
  if (chol)
    delete[] chol;
  if (sum)
    delete[] sum;
  if (summ)
    delete[] summ;
  if (buf)
    delete[] buf;
}

void Rando::load(const std::string &fn) {
  FILE *fp = fopen(fn.c_str(), "r");
  assert(fp);

  uint32_t _dim;
  int ret = (int)fread(&_dim, 1, 4, fp);
  assert(ret == 4);
  int vdim = (int)ntohl(_dim);
  if (dim == 0)
    dim = vdim;
  assert(dim == vdim);

  mean = new double[dim];
  assert(sizeof(double) == 8);
  ret = (int)fread(mean, 1, dim * 8, fp);
  assert(ret == dim * 8);
  
  chol = new double[dim * dim];
  assert(sizeof(double) == 8);
  ret = (int)fread(chol, 1, dim * dim * 8, fp);
  assert(ret == dim * dim * 8);

  fclose(fp);
}

static void cpumatvec(const double *a, const double *b, int aw, int ahbw, double *c) {
  int ah = ahbw;
  int bw = ahbw;
  int cw = aw;

  for (int cx = 0; cx < cw; ++cx) {
    double v = 0;
    for (int i = 0; i < ahbw; ++i) {
      v += a[cx + aw * i] * b[i];
    }
    c[cx] = v;
  }
}

void Rando::generate(double *dat, double mul, double evolve) {
  assert(chol);
  assert(mean);

  if (!buf)
    buf = new double[dim]();

  assert(evolve >= 0 && evolve <= 1);
  double sw = sqrt(evolve);
  double sw1 = sqrt(1.0 - evolve);

  for (int i = 0; i < dim; ++i)
    buf[i] = sw * buf[i] + sw1 * randgauss() * mul;

  cpumatvec(chol, buf, dim, dim, dat);

  for (int i = 0; i < dim; ++i)
    dat[i] += mean[i];
}

void Rando::observe(const double *dat) {
  assert(dim > 0);

  if (!sum)
    sum = new double[dim]();
  if (!summ)
    summ = new double[dim * dim]();

  for (unsigned int z = 0; z < dim; ++z) {
    sum[z] += dat[z];
  }
  for (unsigned int z0 = 0; z0 < dim; ++z0) {
    for (unsigned int z1 = 0; z1 < dim; ++z1) {
      summ[z0 + z1 * dim] += dat[z0] * dat[z1];
    }
  }

  ++sumn;
}

static void cpuchol(double *m, unsigned int dim) {
  unsigned int i, j, k; 

  for (k = 0; k < dim; k++) {
    if (m[k * dim + k] < 0)
      m[k * dim + k] = 0;

    m[k * dim + k] = sqrt(m[k * dim + k]);
    
    if (m[k * dim + k] > 0) {
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] /= m[k * dim + k];
    } else {
      m[k * dim + k] = 1.0;
      for (j = k + 1; j < dim; j++)
        m[k * dim + j] = 0;
    }
             
    for (i = k + 1; i < dim; i++)
      for (j = i; j < dim; j++)
        m[i * dim + j] -= m[k * dim + i] * m[k * dim + j];

  }

  for (i = 0; i < dim; i++)
    for (j = 0; j < i; j++)
      m[i * dim + j] = 0.0;
}

void Rando::save(const std::string &fn) {
  assert(sumn > 1);
  assert(dim > 0);

  if (!mean)
    mean = new double[dim];
  if (!chol)
    chol = new double[dim * dim];

  for (int z = 0; z < dim; ++z) {
    mean[z] = sum[z] / (double)sumn;
  }
  for (int z0 = 0; z0 < dim; ++z0) {
    for (int z1 = 0; z1 < dim; ++z1) {
      chol[z0 + z1 * dim] = summ[z0 + z1 * dim] / (double)sumn - mean[z0] * mean[z1];
    }
  }

  cpuchol(chol, dim);

  FILE *fp = fopen(fn.c_str(), "w");
  int ndim = htonl(dim);
  size_t ret = fwrite(&ndim, 1, 4, fp);
  assert(ret == 4);
  ret = fwrite(mean, 1, dim * 8, fp);
  assert(ret == dim * 8);
  ret = fwrite(chol, 1, dim * dim * 8, fp);
  assert(ret == dim * dim * 8);
  fclose(fp);
}

}
