#define __MAKEMORE_COLONEL_CU__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <math.h>

#include "colonel.hh"

namespace makemore {

int kdev = 0;
int kbs = 256;

void setkdev(int i) {
  assert(i >= 0);
  assert(i <= kndevs());
  if (i > 0)
    assert(0 == cudaSetDevice(i - 1));
  kdev = i;
}

void setkbs(int i) {
  assert(i > 0);
  kbs = i;
}

int kndevs() {
  int ndevs = 0;
  cudaGetDeviceCount(&ndevs);
  assert(ndevs >= 0);
  return (1 + ndevs);
}


#undef syncthreads
#define syncthreads() __syncthreads()

#undef DEFN_KERNEL
#define DEFN_KERNEL(f, args...) \
  __global__ void _gpu_ ## f(long __n, args)
#undef PREF_KERNEL
#define PREF_KERNEL \
  long i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i >= __n) \
    return;

#undef PRE
#define PRE(x) _gpu_ ## x

#include "colonel-core.inc"



#undef syncthreads
#define syncthreads() assert(!"no syncthreads in cpu mode")

#undef __device__
#define __device__ 

#undef PRE
#define PRE(x) _cpu_ ## x


#undef DEFN_KERNEL
#define DEFN_KERNEL(f, args...) \
  void _cpu_ ## f (long i, long __n, args)
#undef PREF_KERNEL
#define PREF_KERNEL \
  if (i >= __n) \
    return;

#include "colonel-core.inc"



#define CALL_KERNEL(f, _n, args...) do { \
  long __n = (_n); \
  if (kdev) { \
    int __bs = kbs, __gs = ((__n + __bs - 1) / __bs); \
    _gpu_ ## f <<<__gs, __bs>>>(__n, args); \
  } else { \
    for (long __i = __n - 1; __i >= 0; --__i) { \
      _cpu_ ## f (__i, __n, args); \
    } \
  } \
} while (0);



void enkv(const void *a, unsigned int n, void *da) {
  if (kdev) {
    ::cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
  } else {
    ::memcpy(da, a, n);
  }
}

void dekv(const void *da, unsigned int n, void *a) {
  if (kdev)
    ::cudaMemcpy(a, da, n, cudaMemcpyDeviceToHost);
  else
    ::memcpy(a, da, n);
}

void kmakev(void **dp, unsigned int n) {
  if (kdev) {
    void *d = NULL;
    // assert(n > 0);
    int ret = ::cudaMalloc((void **)&d, n);
    // assert(d != NULL);
    assert(ret == 0);
    *dp = d;
  } else {
    *dp = (void *)(new char[n]);
    assert(*dp);
  }
}

void kfreev(void *x) {
  if (kdev)
    ::cudaFree(x);
  else
    delete[] ((char *)x);
}

void kzerov(void *x, unsigned int n) {
  if (kdev)
    ::cudaMemset((void *)x, 0, n);
  else
    ::memset(x, 0, n);
}

void kfill(double *x, unsigned int n, double v) {
  if (kdev) {
    double *y = new double[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = v;
    enk(y, n, x);
    delete[] y;
  } else {
    for (unsigned int i = 0; i < n; ++i)
      x[i] = v;
  }
}

void kcopyv(const void *x, unsigned int n, void *y) {
  if (kdev)
    ::cudaMemcpy(y, x, n, cudaMemcpyDeviceToDevice);
  else
    ::memcpy(y, x, n);
}

#include "colonel-common.inc"

}
