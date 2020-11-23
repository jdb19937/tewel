#define __MAKEMORE_COLONEL_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <math.h>

#include <cmath>
#include <limits>

#include "colonel.hh"

namespace makemore {

int kdev = 0;
int kbs = 256;

void setkdev(int i) {
  assert(i == 0);
  kdev = 0;
}

void setkbs(int i) {
  assert(i > 0);
  kbs = i;
}

int kndevs() {
  return 1;
}


// MIT License
// 
// Copyright (c) 2017-2019 Lakshay Garg <lakshayg@outlook.in>

// Implementation of the inverse error function based on the rational
// approximation of percentage points of normal distribution available from
// https://www.jstor.org/stable/2347330.

// The code has been tested on an x86_64 machine with Intel Core i7, the
// tests provided in this repository might not pass for different hardware
// configuration due to difference in floating point operations.

// golang's math library uses the same implementation for erfinv


static long double erfinv(long double x) {

  if (x < -1 || x > 1) {
    return std::numeric_limits<long double>::quiet_NaN();
  } else if (x == 1.0) {
    return std::numeric_limits<long double>::infinity();
  } else if (x == -1.0) {
    return -std::numeric_limits<long double>::infinity();
  }

  const long double LN2 = 6.931471805599453094172321214581e-1L;

  const long double A0 = 1.1975323115670912564578e0L;
  const long double A1 = 4.7072688112383978012285e1L;
  const long double A2 = 6.9706266534389598238465e2L;
  const long double A3 = 4.8548868893843886794648e3L;
  const long double A4 = 1.6235862515167575384252e4L;
  const long double A5 = 2.3782041382114385731252e4L;
  const long double A6 = 1.1819493347062294404278e4L;
  const long double A7 = 8.8709406962545514830200e2L;

  const long double B0 = 1.0000000000000000000e0L;
  const long double B1 = 4.2313330701600911252e1L;
  const long double B2 = 6.8718700749205790830e2L;
  const long double B3 = 5.3941960214247511077e3L;
  const long double B4 = 2.1213794301586595867e4L;
  const long double B5 = 3.9307895800092710610e4L;
  const long double B6 = 2.8729085735721942674e4L;
  const long double B7 = 5.2264952788528545610e3L;

  const long double C0 = 1.42343711074968357734e0L;
  const long double C1 = 4.63033784615654529590e0L;
  const long double C2 = 5.76949722146069140550e0L;
  const long double C3 = 3.64784832476320460504e0L;
  const long double C4 = 1.27045825245236838258e0L;
  const long double C5 = 2.41780725177450611770e-1L;
  const long double C6 = 2.27238449892691845833e-2L;
  const long double C7 = 7.74545014278341407640e-4L;

  const long double D0 = 1.4142135623730950488016887e0L;
  const long double D1 = 2.9036514445419946173133295e0L;
  const long double D2 = 2.3707661626024532365971225e0L;
  const long double D3 = 9.7547832001787427186894837e-1L;
  const long double D4 = 2.0945065210512749128288442e-1L;
  const long double D5 = 2.1494160384252876777097297e-2L;
  const long double D6 = 7.7441459065157709165577218e-4L;
  const long double D7 = 1.4859850019840355905497876e-9L;

  const long double E0 = 6.65790464350110377720e0L;
  const long double E1 = 5.46378491116411436990e0L;
  const long double E2 = 1.78482653991729133580e0L;
  const long double E3 = 2.96560571828504891230e-1L;
  const long double E4 = 2.65321895265761230930e-2L;
  const long double E5 = 1.24266094738807843860e-3L;
  const long double E6 = 2.71155556874348757815e-5L;
  const long double E7 = 2.01033439929228813265e-7L;

  const long double F0 = 1.414213562373095048801689e0L;
  const long double F1 = 8.482908416595164588112026e-1L;
  const long double F2 = 1.936480946950659106176712e-1L;
  const long double F3 = 2.103693768272068968719679e-2L;
  const long double F4 = 1.112800997078859844711555e-3L;
  const long double F5 = 2.611088405080593625138020e-5L;
  const long double F6 = 2.010321207683943062279931e-7L;
  const long double F7 = 2.891024605872965461538222e-15L;

  long double abs_x = abs(x);

  if (abs_x <= 0.85L) {
    long double r =  0.180625L - 0.25L * x * x;
    long double num = (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0);
    long double den = (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + B0);
    return x * num / den; 
  }

  long double r = sqrt(LN2 - log(1.0L - abs_x));

  long double num, den;
  if (r <= 5.0L) {
    r = r - 1.6L;
    num = (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0);
    den = (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + D0);
  } else {
    r = r - 5.0L;
    num = (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0);
    den = (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + F0);
  }

  if (x < 0L) {
    return -num / den;
  } else {
    return num / den;
  }
}

static long double erfinv(long double x, int nr_iter) {
  long double k = 0.8862269254527580136490837416706L; // 0.5 * sqrt(pi)
  long double y = erfinv(x);
  while (nr_iter-- > 0) {
    y -= k * (std::erf(y) - x) / std::exp(-y * y);
  }
  return y;
}






#undef DEFN_KERNEL
#define DEFN_KERNEL(f, args...) \
  void _cpu_ ## f (long i, long __n, args)
#undef PREF_KERNEL
#define PREF_KERNEL \
  if (i >= __n) \
    return;

#undef syncthreads
#define syncthreads() assert(!"no syncthreads in cpu mode")

#undef __device__
#define __device__

#undef PRE
#define PRE(x) _cpu_ ## x

#include "colonel-core.inc"


#define CALL_KERNEL(f, _n, args...) do { \
  long __n = (_n); \
  for (long __i = __n - 1; __i >= 0; --__i) { \
    _cpu_ ## f (__i, __n, args); \
  } \
} while (0);



void enkv(const void *a, unsigned int n, void *da) {
  ::memcpy(da, a, n);
}

void dekv(const void *da, unsigned int n, void *a) {
  ::memcpy(a, da, n);
}

void kmakev(void **dp, unsigned int n) {
  *dp = (void *)(new char[n]);
  assert(*dp);
}

void kfreev(void *x) {
  delete[] ((char *)x);
}

void kzerov(void *x, unsigned int n) {
  ::memset(x, 0, n);
}

void kfill(double *x, unsigned int n, double v) {
  for (unsigned int i = 0; i < n; ++i)
    x[i] = v;
}

void kcopyv(const void *x, unsigned int n, void *y) {
  ::memcpy(y, x, n);
}

#include "colonel-common.inc"

}
