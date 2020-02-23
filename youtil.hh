#ifndef __MAKEMORE_YOUTIL_HH__
#define __MAKEMORE_YOUTIL_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <string>

namespace makemore {

extern void endub(const uint8_t *x, unsigned int n, double *y);
extern void dedub(const double *y, unsigned int n, uint8_t *x);

inline bool startswith(const std::string &str, const std::string &suf) {
  return (
    suf.length() <= str.length() &&
    suf == str.substr(0, suf.length())
  );
}

inline bool endswith(const std::string &str, const std::string &suf) {
  return (
    suf.length() <= str.length() &&
    suf == str.substr(str.length() - suf.length(), suf.length())
  );
}

extern bool load_ppm(FILE *fp, uint8_t **rgbp, unsigned int *wp, unsigned int *hp);
extern void save_ppm(FILE *fp, uint8_t *rgb, unsigned int w, unsigned int h);
extern void load_pic(const std::string &fn, uint8_t **rgbp, unsigned int *wp, unsigned int *hp);

extern uint8_t *slurp(const std::string &fn, size_t *np);


inline double strtod(const std::string &str) {
  return ::strtod(str.c_str(), NULL);
}

inline unsigned int strtoui(const std::string &str) {
  return (unsigned int)::strtoul(str.c_str(), NULL, 0);
}

inline int strtoi(const std::string &str) {
  return (int)::strtol(str.c_str(), NULL, 0);
}

inline void warning(const std::string &str) {
  fprintf(stderr, "Warning: %s\n", str.c_str());
}

inline void error(const std::string &str) {
  fprintf(stderr, "Error: %s\n", str.c_str());
  exit(1);
}

}

#endif
