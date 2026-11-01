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
extern void kfill(float *, unsigned int, float v);
  
extern void kcopyv(const void *, unsigned int, void *);
template <class T> inline void kcopy(const T *da, unsigned int n, T *db)
  { kcopyv(da, n * sizeof(T), db); }


extern void kclip3(float *a, unsigned int n, float c);
extern void kaddvec(const float *a, const float *b, unsigned int n, float *c);
extern void kaxpy(const float *x, float a, unsigned int n, float *y);
extern void kcharb(float *e, unsigned int n, float delta);
extern void kema(float *a, const float *b, unsigned int n, float beta);
extern void klogis(float *x, unsigned int n, int sign);
extern int size_mcon(int ic, int oc, int sdim);
extern void synth_mcon(const float *in, int iw, int ih, float *out, int ic, int oc, int bn, const float *wmv, const float *kstyle, int sdim, float wscale, float wscaleA, float *stash);
extern void learn_mcon(float *fin, int iw, int ih, float *fout, int ic, int oc, int bn, float *wmv, float *dwv, const float *kstyle, float *kstylegrad, int sdim, float wscale, float wscaleA, const float *stash, float nu, float b1, float b2, float eps, float clip, float rounds);
extern int size_styl(int ic, int sdim);
extern void synth_styl(const float *in, int iw, int ih, float *out, int ic, int bn, const float *wmv, const float *kstyle, int sdim, float wscale);
extern void learn_styl(float *fin, int iw, int ih, const float *fout, int ic, int bn, float *wmv, float *dwv, const float *kstyle, float *kstylegrad, int sdim, float wscale, float nu, float b1, float b2, float eps, float clip, float rounds);
extern void synth_mbsd(const float *in, int iw, int ih, float *out, int ic, int bn);
extern void learn_mbsd(float *fin, int iw, int ih, const float *fout, int ic, int bn);
extern void synth_upbl(const float *in, int iw, int ih, float *out, int ic, int bn);
extern void learn_upbl(float *fin, int iw, int ih, const float *fout, int ic, int bn);
extern void synth_dnav(const float *in, int iw, int ih, float *out, int ic);
extern void learn_dnav(float *fin, int iw, int ih, const float *fout, int ic);
extern void synth_lrlu(const float *in, int iw, int ih, float *out, int ic, float slope);
extern void learn_lrlu(float *fin, int iw, int ih, const float *fout, int ic, float slope);
extern void synth_pnrm(const float *in, int iw, int ih, float *out, int ic);
extern void learn_pnrm(float *fin, int iw, int ih, const float *fout, int ic);
extern void ksubvec(const float *a, const float *b, unsigned int n, float *c);
extern void kmul(const float *a, const float b, unsigned int n, float *c);
extern float ksumsq(const float *a, unsigned int n);
extern float ksum(const float *a, unsigned int n);
extern float kmaxabs(const float *a, unsigned int n);


extern void kspliceadd(
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
);
extern void ksplice(
  const float *x, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
);
extern void ksplice2(
  const float *x, const float *z, float l, int n, int xm, int xa, int xk,
  float *y, int ym, int ya
);

extern int size_norm(
  int ic, int ow, int oh, int oc
);

extern void synth_norm(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
);

extern void learn_norm(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
);

extern int size_bias(
  int ic, int ow, int oh, int oc
);

extern void synth_bias(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc,
  const float *wmv
);

extern void learn_bias(
  float *fin, int iw, int ih,
  const float *fout,

  int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
);

extern int size_local(
  int d, int ic, int ow, int oh, int oc
);

extern void synth_local(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  const float *wmv
);

extern void learn_local(
  float *fin, int iw, int ih,
  const float *fout,

  int d, int ic, int oc,

  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds
);

extern void synth_trim(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic
);

extern void learn_trim(
  float *fin, int iw, int ih,
  const float *fout,
  int d, int ic
);

extern void synth_crop(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic,
  int ix0, int iy0
);

extern void learn_crop(
  float *fin, int iw, int ih,
  const float *fout,
  int d, int ic
);

extern void synth_padd(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic
);

extern void learn_padd(
  float *fin, int iw, int ih,
  const float *fout,
  int d, int ic
);



extern int size_conv(int d, int ic, int oc, int dim);

extern void synth_conv(
  const float *in, int iw, int ih,
  float *out,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  const float *wmv,
  int bn = 1, float wscale = 1.0
);

extern void learn_conv(
  float *fin, int iw, int ih,
  const float *fout,
  int d, int ic, int oc,
  bool relu, int freeze, int dim,
  float *wmv, float *dwv,
  float nu, float b1, float b2, float eps, float clip, float rounds,
  int bn = 1, float wscale = 1.0
);

// batched tiled-GEMM convolution (GPU only; stubs in the no-CUDA build)
extern void gpu_synth_conv(const float *in, int bn, int iw, int ih, float *out, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale);
extern void gpu_learn_conv_w(const float *in, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, float *dwv, float wscale);
extern int kprec;
extern void setkprec(int);
extern void gpu_styl_affine(const float *wmv, const float *kstyle, int bn, int ic, int sdim, float wscale, float *ksc);
extern void gpu_synth_conv16(const float *in, int bn, int iw, int ih, float *out, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale);
extern void gpu_learn_conv_w16(const float *in, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, float *dwv, float wscale);
extern void gpu_learn_conv_in16(float *fin, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale);
extern void gpu_learn_pnrm(float *fin, int pn, int ic, const float *fout);
extern void gpu_synth_pnrm(const float *in, long pn, int ic, float *out);
extern void gpu_styl_dsc(const float *x, const float *g, long pn, int ic, int bn, float *kd);
extern void gpu_learn_conv_in(float *fin, int bn, int iw, int ih, const float *fout, int d, int ic, int oc, bool relu, int freeze, int dim, const float *wmv, float wscale);


extern void synth_zero(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_zero(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_tlab(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);
extern void synth_flab(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void synth_popp(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_popp(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_iden(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_iden(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_smax(
  float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_smax(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_mean(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_mean(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_median(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_median(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_blur(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_blur(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_diff(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_diff(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_sum(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_sum(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_addgeo(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc
);

extern void learn_addgeo(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_pad(
  const float *in, int iw, int ih,
  float *out,
  int ic, int oc, const float *kbuf
);

extern void learn_pad(
  float *fin, int iw, int ih,
  const float *fout,
  int ic, int oc
);

extern void synth_cent(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_cent(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_sigm(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_sigm(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_nerf(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_nerf(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_inrf(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_inrf(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_clamp(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_clamp(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_relu(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_relu(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);

extern void synth_abs(
  const float *in, int iw, int ih,
  float *out,
  int ic
);

extern void learn_abs(
  float *fin, int iw, int ih,
  const float *fout,
  int ic
);


extern void synth_upscale(
  const float *in, int iw, int ih,
  float *out,
  int s, int ic, int oc, int dim
);

extern void learn_upscale(
  float *fin, int iw, int ih,
  const float *fout,
  int s, int ic, int oc, int dim
);

extern void synth_downscale(
  const float *in, int iw, int ih,
  float *out,
  int s, int ic, int oc, int dim
);

extern void learn_downscale(
  float *fin, int iw, int ih,
  const float *fout,
  int s, int ic, int oc, int dim
);

}

#endif
