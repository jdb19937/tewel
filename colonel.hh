#ifndef __MAKEMORE_COLONEL_HH__
#define __MAKEMORE_COLONEL_HH__ 1

namespace makemore {

extern void setkdev(int);
extern void setkbs(int);
extern int kndevs();

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


extern void kclip3(double *a, unsigned int n, double c);
extern void kaddvec(const double *a, const double *b, unsigned int n, double *c);
extern void ksubvec(const double *a, const double *b, unsigned int n, double *c);
extern void kmul(const double *a, const double b, unsigned int n, double *c);
extern double ksumsq(const double *a, unsigned int n);
extern double ksum(const double *a, unsigned int n);
extern double kmaxabs(const double *a, unsigned int n);


extern void kspliceadd(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
);
extern void ksplice(
  const double *x, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
);
extern void ksplice2(
  const double *x, const double *z, double l, int n, int xm, int xa, int xk,
  double *y, int ym, int ya
);

extern int size_norm(
  int ic, int ow, int oh, int oc
);

extern void synth_norm(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
);

extern void learn_norm(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
);

extern int size_bias(
  int ic, int ow, int oh, int oc
);

extern void synth_bias(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc,
  const double *wmv
);

extern void learn_bias(
  double *fin, int iw, int ih,
  const double *fout,

  int ic, int oc,

  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
);

extern int size_local(
  int d, int ic, int ow, int oh, int oc
);

extern void synth_local(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  const double *wmv
);

extern void learn_local(
  double *fin, int iw, int ih,
  const double *fout,

  int d, int ic, int oc,

  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
);

extern void synth_trim(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic
);

extern void learn_trim(
  double *fin, int iw, int ih,
  const double *fout,
  int d, int ic
);

extern void synth_crop(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic,
  int ix0, int iy0
);

extern void learn_crop(
  double *fin, int iw, int ih,
  const double *fout,
  int d, int ic
);

extern void synth_padd(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic
);

extern void learn_padd(
  double *fin, int iw, int ih,
  const double *fout,
  int d, int ic
);



extern int size_conv(int d, int ic, int oc, int dim);

extern void synth_conv(
  const double *in, int iw, int ih,
  double *out,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  const double *wmv
);

extern void learn_conv(
  double *fin, int iw, int ih,
  const double *fout,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  double *wmv, double *dwv,
  double nu, double b1, double b2, double eps, double clip, double rounds
);


extern void synth_zero(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_zero(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_tlab(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);
extern void synth_flab(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void synth_popp(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_popp(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_iden(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_iden(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_smax(
  double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_smax(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_mean(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_mean(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_median(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_median(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_blur(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_blur(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_diff(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_diff(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_sum(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc, const double *kbuf
);

extern void learn_sum(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
);

extern void synth_addgeo(
  const double *in, int iw, int ih,
  double *out,
  int ic, int oc
);

extern void learn_addgeo(
  double *fin, int iw, int ih,
  const double *fout,
  int ic, int oc
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

extern void synth_cent(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_cent(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);

extern void synth_sigm(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_sigm(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);

extern void synth_nerf(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_nerf(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);

extern void synth_inrf(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_inrf(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
);

extern void synth_clamp(
  const double *in, int iw, int ih,
  double *out,
  int ic
);

extern void learn_clamp(
  double *fin, int iw, int ih,
  const double *fout,
  int ic
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
  int s, int ic, int oc, int dim
);

extern void learn_upscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
);

extern void synth_downscale(
  const double *in, int iw, int ih,
  double *out,
  int s, int ic, int oc, int dim
);

extern void learn_downscale(
  double *fin, int iw, int ih,
  const double *fout,
  int s, int ic, int oc, int dim
);

}

#endif
