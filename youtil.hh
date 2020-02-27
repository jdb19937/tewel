#ifndef __MAKEMORE_YOUTIL_HH__
#define __MAKEMORE_YOUTIL_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <netinet/in.h>

#include <string>
#include <vector>

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

extern uint8_t *slurp(const std::string &fn, size_t *np);


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
  fprintf(stderr, "Warning: %s\n", str.c_str());
}

inline void error(const std::string &str) {
  fprintf(stderr, "Error: %s\n", str.c_str());
  exit(1);
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

}

#endif
