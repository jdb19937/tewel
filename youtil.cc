#define __MAKEMORE_YOUTIL_CC__ 1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <string>

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

}
