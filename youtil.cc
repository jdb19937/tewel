#define __MAKEMORE_YOUTIL_CC__ 1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <string>
#include <vector>

#include "youtil.hh"

namespace makemore {

static inline double btod(uint8_t b) {
  return ((double)b / 256.0);
}

static inline uint8_t dtob(double d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((uint8_t)(d + 0.5));
}

void endub(const uint8_t *b, unsigned int n, double *d) {
  for (unsigned int i = 0; i < n; ++i)
    d[i] = btod(b[i]);
}

void dedub(const double *d, unsigned int n, uint8_t *b) {
  for (unsigned int i = 0; i < n; ++i)
    b[i] = dtob(d[i]);
}


uint8_t *slurp(const std::string &fn, size_t *np) {
  FILE *fp = fopen(fn.c_str(), "r");
  assert(fp);
  int ret = fseek(fp, 0, SEEK_END);
  assert(ret == 0);
  long n = ftell(fp);
  ret = fseek(fp, 0, SEEK_SET);
  assert(ret == 0);

  uint8_t *x = NULL;
  if (n > 0) {
    x = new uint8_t[n];
    size_t ret2 = ::fread(x, 1, n, fp);
    assert(n == ret2);
  } else {
    x = NULL;
  }

  if (np)
    *np = n;
  fclose(fp);

  return x;
}

bool parsedim(const std::string &dim, int *wp, int *hp, int *cp) {
  const char *cdim = dim.c_str();

  if (*cdim == 'x') {
    ++cdim;
    int c = 0;
    while (isdigit(*cdim)) {
      c *= 10;
      c += (*cdim - '0');
      ++cdim;
    }
    if (*cdim)
      return false;
    if (cp)
      *cp = c;
    return true;
  }

  if (!isdigit(*cdim))
    return false;

  int w = 0;
  while (isdigit(*cdim)) {
    w *= 10;
    w += (*cdim - '0');
    ++cdim;
  }
  if (!*cdim) {
    if (wp)
      *wp = w;
    if (hp)
      *hp = w;
    return true;
  }

  if (*cdim != 'x')
    return false;
  ++cdim;

  int h = 0;
  while (isdigit(*cdim)) {
    h *= 10;
    h += (*cdim - '0');
    ++cdim;
  }

  if (!*cdim) {
    if (wp)
      *wp = w;
    if (hp)
      *hp = h;
    return true;
  }

  if (*cdim != 'x')
    return false;
  ++cdim;

  int c = 0;
  while (isdigit(*cdim)) {
    c *= 10;
    c += (*cdim - '0');
    ++cdim;
  }

  if (*cdim)
    return false;

  if (wp)
    *wp = w;
  if (hp)
    *hp = h;
  if (cp)
    *cp = c;
  return true;
}

bool read_line(FILE *fp, std::string *line) {
  char buf[4096];

  int c = getc(fp);
  if (c == EOF)
    return false;
  ungetc(c, fp);

  *buf = 0;
  char *unused = fgets(buf, sizeof(buf) - 1, fp);
  char *p = strchr(buf, '\n');
  if (!p)
    return false;
  *p = 0;

  *line = buf;
  return true;
}

bool parserange(const std::string &str, unsigned int *ap, unsigned int *bp) {
  const char *cut = str.c_str();

  unsigned int a, b;
  if (isdigit(*cut)) {
    a = 0;
    while (isdigit(*cut))
      a = a * 10 + *cut++ - '0';
  } else if (*cut == '-') {
    a = 0;
  } else {
    return false;
  }

  if (*cut == '-') {
    ++cut;
    if (isdigit(*cut)) {
      b = 0;
      while (isdigit(*cut))
        b = b * 10 + *cut++ - '0';
    } else if (*cut == ',' || *cut == '\0') {
      b = 255;
    } else {
      return false;
    }
  } else {
    b = a;
  }

  *ap = a;
  *bp = b;
  return true;
}


void parseargs(const std::string &str, std::vector<std::string> *vec) {
  const char *p = str.c_str();

  while (*p) {
    while (isspace(*p))
      ++p;
    if (!*p)
      return;

    const char *q = p + 1;
    while (*q && !isspace(*q))
      ++q;
    vec->push_back(std::string(p, q - p));

    p = q;
  }
}

void freeargstrad(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) {
    assert(argv[i]);
    delete[] argv[i];
  }
  assert(argv);
  delete[] argv;
}

void parseargstrad(const std::string &str, int *argcp, char ***argvp) {
  std::vector<std::string> vec;
  parseargs(str, &vec);

  int argc = vec.size();
  char **argv = new char *[argc];
  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[vec[i].length() + 1];
    strcpy(argv[i], vec[i].c_str());
  }

  *argcp = argc;
  *argvp = argv;
}

void argvec(int argc, char **argv, std::vector<std::string> *vec) {
  vec->resize(argc);
  for (int i = 0; i < argc; ++i)
    (*vec)[i] = std::string(argv[i]);
}

bool is_dir(const std::string &fn) {
  struct stat buf;
  int ret = ::stat(fn.c_str(), &buf);
  if (ret != 0)
    return false;
  return (S_ISDIR(buf.st_mode));
}

bool fexists(const std::string &fn) {
  return (access(fn.c_str(), F_OK) != -1);
}

}
