#define __MAKEMORE_COLONEL_CU__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <math.h>
#include <map>

#include <mma.h>
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

#undef KATOMIC_ADD
#define KATOMIC_ADD(p, v) atomicAdd((p), (v))

#include "colonel-core.inc"
#include "colonel-gemm.inc"



#undef syncthreads
#define syncthreads() assert(!"no syncthreads in cpu mode")

#undef __device__
#define __device__ 

#undef PRE
#define PRE(x) _cpu_ ## x

#undef KATOMIC_ADD
#define KATOMIC_ADD(p, v) (*(p) += (v))


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

// Caching device allocator: cudaFree synchronizes the whole device, so freed blocks are kept
// on a per-size free list and reused by the next allocation of the same size.
static std::multimap<unsigned int, void *> *g_kfree = NULL;
static std::map<void *, unsigned int> *g_klive = NULL;

void kmakev(void **dp, unsigned int n) {
  if (kdev) {
    if (!g_kfree) { g_kfree = new std::multimap<unsigned int, void *>; g_klive = new std::map<void *, unsigned int>; }
    auto it = g_kfree->find(n);
    if (it != g_kfree->end()) {
      *dp = it->second;
      g_kfree->erase(it);
      (*g_klive)[*dp] = n;
      return;
    }
    void *d = NULL;
    int ret = ::cudaMalloc((void **)&d, n);
    if (ret != 0) {
      // out of memory: release the cache and retry once
      for (auto &kv : *g_kfree) ::cudaFree(kv.second);
      g_kfree->clear();
      ret = ::cudaMalloc((void **)&d, n);
    }
    if (ret != 0)
      fprintf(stderr, "kmakev: cudaMalloc(%u bytes) failed: %s\n", n, cudaGetErrorString((cudaError_t)ret));
    assert(ret == 0);
    (*g_klive)[d] = n;
    *dp = d;
  } else {
    *dp = (void *)(new char[n]);
    assert(*dp);
  }
}

void kfreev(void *x) {
  if (kdev) {
    if (!x) return;
    auto it = g_klive->find(x);
    assert(it != g_klive->end());
    g_kfree->insert(std::make_pair(it->second, x));
    g_klive->erase(it);
  } else {
    delete[] ((char *)x);
  }
}

void kzerov(void *x, unsigned int n) {
  if (kdev)
    ::cudaMemset((void *)x, 0, n);
  else
    ::memset(x, 0, n);
}

__global__ void _fill_kernel(float *x, unsigned int n, float v) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

void kfill(float *x, unsigned int n, float v) {
  if (kdev) {
    if (n > 0) _fill_kernel<<<(n + 255) / 256, 256>>>(x, n, v);
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
