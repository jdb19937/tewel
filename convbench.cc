// Time the conv kernels (fwd, bwd_in, bwd_w) for given shapes at fp32 and tensor-core precision.
// Usage: convbench bn iw ic oc [reps]
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "colonel.hh"
#include "random.hh"
using namespace makemore;
int main(int argc, char **argv) {
  int bn = atoi(argv[1]), iw = atoi(argv[2]), ic = atoi(argv[3]), oc = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 5;
  int ih = iw;
  setkdev(1); setkbs(256);
  long inn = (long)bn * iw * ih * ic, outn = (long)bn * iw * ih * oc;
  int wn = oc + oc * ic * 9;
  float *in, *out, *fout, *wmv, *dwv;
  kmake(&in, inn); kmake(&out, outn); kmake(&fout, outn); kmake(&wmv, 3 * wn); kmake(&dwv, wn);
  kzero(in, inn); kaddnoise(in, inn, 1.0); kzero(fout, outn); kaddnoise(fout, outn, 1.0); kzero(wmv, 3 * wn); kaddnoise(wmv, wn, 0.05); kzero(dwv, wn);
  double flop = 2.0 * (double)bn * iw * ih * oc * ic * 9;
  for (int prec = 32; prec >= 16; prec -= 16) {
    setkprec(prec);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t[3] = {0, 0, 0};
    for (int r = -1; r < reps; ++r) {   // one warmup
      cudaEventRecord(e0); synth_conv(in, iw, ih, out, 1, ic, oc, false, 0, 2, wmv, bn, 1.0f); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (r >= 0) t[0] += ms;
      cudaEventRecord(e0); (prec == 16 ? gpu_learn_conv_in16 : gpu_learn_conv_in)(in, bn, iw, ih, fout, 1, ic, oc, false, 0, 2, wmv, 1.0f); cudaEventRecord(e1); cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1); if (r >= 0) t[1] += ms;
      cudaEventRecord(e0); (prec == 16 ? gpu_learn_conv_w16 : gpu_learn_conv_w)(in, bn, iw, ih, fout, 1, ic, oc, false, 0, 2, dwv, 1.0f); cudaEventRecord(e1); cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1); if (r >= 0) t[2] += ms;
    }
    printf("bn=%d %dx%d ic=%d oc=%d prec=%d: fwd %.2f ms (%.1f TFLOPS)  bwd_in %.2f ms (%.1f)  bwd_w %.2f ms (%.1f)\n", bn, iw, ih, ic, oc, prec,
      t[0] / reps, flop / (t[0] / reps * 1e-3) / 1e12, t[1] / reps, flop / (t[1] / reps * 1e-3) / 1e12, t[2] / reps, flop / (t[2] / reps * 1e-3) / 1e12);
  }
  return 0;
}
