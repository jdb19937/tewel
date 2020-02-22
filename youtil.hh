#ifndef __MAKEMORE_YOUTIL_HH__
#define __MAKEMORE_YOUTIL_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <string>

namespace makemore {

extern void endub(const uint8_t *x, unsigned int n, double *y);
extern void dedub(const double *y, unsigned int n, uint8_t *x);

inline bool endswith(const std::string &str, const std::string &suf) {
  return (
    suf.length() <= str.length() &&
    suf == str.substr(str.length() - suf.length(), suf.length())
  );
}

extern bool load_ppm(FILE *fp, uint8_t **rgbp, unsigned int *wp, unsigned int *hp);
extern void save_ppm(FILE *fp, uint8_t *rgb, unsigned int w, unsigned int h);
extern void load_img(const std::string &fn, uint8_t **rgbp, unsigned int *wp, unsigned int *hp);

extern uint8_t *slurp(const std::string &fn, size_t *np);

}

#endif
