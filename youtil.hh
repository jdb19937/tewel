#ifndef __MAKEMORE_YOUTIL_HH__
#define __MAKEMORE_YOUTIL_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <netinet/in.h>

#include <string>
#include <map>
#include <vector>

namespace makemore {

inline double btod(uint8_t b) {
  return ((double)b / 256.0);
}

inline uint8_t dtob(double d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((uint8_t)(d + 0.5));
}

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

extern uint8_t *slurp(FILE *fp, size_t *np);
extern uint8_t *slurp(const std::string &fn, size_t *np);
extern void spit(const uint8_t *x, size_t n, FILE *fp);
extern void spit(const std::string &str, FILE *fp);
extern void spit(const std::string &str, const std::string &fn);
extern void spit(const uint8_t *x, size_t n, const std::string &fn);

inline double strtod(const std::string &str) {
  return ::strtod(str.c_str(), NULL);
}

inline unsigned int strtoui(const std::string &str) {
  return (unsigned int)::strtoul(str.c_str(), NULL, 0);
}

inline uint64_t strtoul(const std::string &str) {
  return (uint64_t)::strtoul(str.c_str(), NULL, 0);
}

inline int64_t strtol(const std::string &str) {
  return (int64_t)::strtol(str.c_str(), NULL, 0);
}

inline int strtoi(const std::string &str) {
  return (int)::strtol(str.c_str(), NULL, 0);
}

inline void warning(const std::string &str) {
  extern int verbose;
  if (verbose >= 0)
    fprintf(stderr, "warning: %s\n", str.c_str());
}

inline void error(const std::string &str) {
  extern int verbose;
  if (verbose >= -1)
    fprintf(stderr, "error: %s\n", str.c_str());
  exit(1);
}

inline void info(const std::string &str) {
  extern int verbose;
  if (verbose >= 1)
    fprintf(stderr, "info: %s\n", str.c_str());
}

extern bool parsedim(const std::string &dim, int *wp, int *hp, int *cp);
extern bool parsedim2(const std::string &dim, int *wp, int *hp);
extern bool parsedim23(const std::string &dim, int *wp, int *hp, int *cp);

inline double now() {
  struct timeval tv;
  assert(0 == gettimeofday(&tv, NULL));
  return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0);
}

extern bool read_line(FILE *fp, std::string *line);

inline uint64_t htonll(uint64_t x) { return htobe64(x); }
inline uint64_t ntohll(uint64_t x) { return be64toh(x); }

bool parsecut(const std::string &_cut, uint8_t *incp);
void parseargs(const std::string &str, std::vector<std::string> *vec);
void parseargstrad(const std::string &str, int *argcp, char ***argvp);
void freeargstrad(int argc, char **argv);

void argvec(int argc, char **argv, std::vector<std::string> *vec);

bool parserange(const std::string &str, unsigned int *ap, unsigned int *bp);

bool is_dir(const std::string &fn);
bool fexists(const std::string &fn);

std::string fmt(const std::string &x, ...);

typedef std::map<std::string,std::string> strmap;
int parsekv(const std::string &kvstr, strmap *kvp);
std::string kvstr(const strmap &kv);

inline std::string str(const char *x) {
  return std::string(x);
}

void parse_bitfmt(const std::string &bitfmt, char *cp, unsigned int *bitsp, bool *bep);

inline double clamp(double x, double x0 = 0.0, double x1 = 1.0) {
  if (x < x0)
    return x0;
  if (x > x1)
    return x1;
  return x;
}

bool pngrgb(
  uint8_t *png,
  unsigned long pngn,
  unsigned int *wp,
  unsigned int *hp,
  uint8_t **rgbp
);

bool rgbpng(
  const uint8_t *rgb,
  unsigned int w,
  unsigned int h,
  uint8_t **pngp,
  unsigned long *pngnp,
  const std::vector<std::string> *tags = NULL,
  const uint8_t *alpha = NULL
);

void rgbjpg(
  const uint8_t *rgb,
  unsigned int w,
  unsigned int h,
  uint8_t **jpgp,
  unsigned long *jpgnp
);

bool jpgrgb(
  const uint8_t *jpg,
  unsigned long jpgn,
  unsigned int *wp,
  unsigned int *hp,
  uint8_t **rgbp
);

}

#endif
