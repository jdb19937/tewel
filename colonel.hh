#ifndef __MAKEMORE_COLONEL_HH__
#define __MAKEMORE_COLONEL_HH__ 1

namespace makemore {

typedef void kvoid;
typedef double kdouble;

extern void dekv(const void *da, unsigned int n, void *a);
template <class T> inline void dek(const T *da, unsigned int n, T *a)
  { dekv(da, n * sizeof(T), a); }

extern void enkv(const void *a, unsigned int n, void *da);
template <class T> inline void enk(const T *a, unsigned int n, T *da)
  { enkv(a, n * sizeof(T), da); }

extern void kmakev(void **, unsigned int n = 1);
template <class T> inline void kmake(T **x, unsigned int n = 1)
  { kmakev((void **)x, n * sizeof(T)); }
template <class T> T *knew(unsigned int n = 1)
  { T *x; kmake(&x, n); return x; }

extern void kfreev(void *);
template <class T> void kfree(T *x)
  { kfreev(x); }

extern void kzerov(void *, unsigned int);
template <class T> inline void kzero(T *x, unsigned int n)
  { kzerov(x, n * sizeof(T)); }
extern void kfill(double *, unsigned int, double v);
  
extern void kcopyv(const void *, unsigned int, void *);
template <class T> inline void kcopy(const T *da, unsigned int n, T *db)
  { kcopyv(da, n * sizeof(T), db); }


extern void kaddvec(const double *a, const double *b, unsigned int n, double *c);
extern void ksubvec(const double *a, const double *b, unsigned int n, double *c);
extern double ksumsq(const double *a, unsigned int n);
extern double kmaxabs(const double *a, unsigned int n);


extern void kspliceadd(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
);
extern void ksplice(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
);


extern int size_conv(int d, int ic, int oc);

extern void synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
);

extern void learn_conv(
  double *fin, int iw, int ih,
  const double *fout,
  int d, int ic, int oc,
  double *wmv,
  double nu, double b1, double b2, double eps, double rounds
);


extern void synth_pad(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_pad(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_relu(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_relu(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);

extern void synth_abs(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_abs(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);


extern void synth_upscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
);

extern void learn_upscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
);

extern void synth_downscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc
);

extern void learn_downscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc
);

}

#endif
