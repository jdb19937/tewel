#define __MAKEMORE_YOUTIL_CC__ 1

#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
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
#include <cmath>
#include <limits>

#include "youtil.hh"

namespace makemore {

int verbose = 0;

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

bool parsedim2(const std::string &dim, int *wp, int *hp) {
  const char *cdim = dim.c_str();

  if (!isdigit(*cdim))
    return false;

  int w = 0;
  while (isdigit(*cdim)) {
    w *= 10;
    w += (*cdim - '0');
    ++cdim;
  }
  if (!*cdim) {
    *wp = w;
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

  *wp = w;
  *hp = h;
  return true;
}

bool parsedim23(const std::string &dim, int *wp, int *hp, int *cp) {
  const char *cdim = dim.c_str();

  if (!isdigit(*cdim))
    return false;

  int w = 0;
  while (isdigit(*cdim)) {
    w *= 10;
    w += (*cdim - '0');
    ++cdim;
  }
  if (!*cdim) {
    *wp = w;
    *hp = w;
    *cp = 3;
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
    *wp = w;
    *hp = h;
    *cp = 3;
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

  if (!*cdim) {
    *wp = w;
    *hp = h;
    *cp = c;
    return true;
  }

  return false;
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
  return (::access(fn.c_str(), F_OK) != -1);
}

std::string fmt(const std::string &x, ...) {
  va_list ap;
  va_start(ap, x);

  va_list ap0;
  va_copy(ap0, ap);
  ssize_t yn = ::vsnprintf(NULL, 0, x.c_str(), ap0);
  assert(yn > 0);

  std::string y;
  y.resize(yn + 1);
  ssize_t vyn = ::vsnprintf((char *)y.data(), yn + 1, x.c_str(), ap);
  assert(yn == vyn);

  return y;
}

int parsekv(const std::string &kvstr, strmap *kvp) {
  int keys = 0;
  const char *p, *q;

  p = kvstr.c_str();
  while (1) {
    while (isspace(*p))
      ++p;
    if (!*p || *p == '#')
      return keys;
    const char *q = p;
    while (*q && *q != '=' && *q != '#')
      ++q;
    if (p == q)
      return -1;
    if (*q != '=')
      return -1;
    std::string k(p, q - p);

    p = ++q;
    while (*q && !isspace(*q) && *q != '#')
      ++q;
    if (p == q)
      return -1;
    std::string v(p, q - p);

    (*kvp)[k] = v;
    ++keys;

    p = q;
  }

  assert(0);
}

std::string kvstr(const strmap &kv) {
  std::string out;
  int i = 0;
  for (auto kvi = kv.begin(); kvi != kv.end(); ++kvi) {
    if (i > 0)
      out += " ";
    out += fmt("%s=%s", kvi->first.c_str(), kvi->second.c_str());
    ++i;
  }
  return out;
}

void parse_bitfmt(const std::string &bitfmt, char *cp, unsigned int *bitsp, bool *bep) {
  const char *b = bitfmt.c_str();
  char c = ::tolower(*b);

  switch (c) {
  case 'u':
  case 's':
  case 'f':
    break;
  default:
    error("bad bitfmt type");
  }

  ++b;

  unsigned int bits = 0;
  while (isdigit(*b)) {
    bits *= 10;
    bits += *b - '0';
    ++b;
  }

  switch (bits) {
  case 8:
  case 16:
    assert(c == 'u' || c == 's');
    break;
  case 32:
  case 64:
    break;
  default:
    error("bad bitfmt bits");
  }

  bool be = false;
  if (!::strcmp(b, "be")) {
    be = true;
  } else {
    assert(*b == 0 || !::strcmp(b, "le"));
  }

  *cp = c;
  *bitsp = bits;
  *bep = be;
}

void *encode_bitfmt(const std::string &bitfmt, const double *x, unsigned int n) {
  char c;
  unsigned int bits;
  bool be;

  parse_bitfmt(bitfmt, &c, &bits, &be);

  assert(bits % 8 == 0);
  unsigned int yn = n * (bits / 8);
  void *ret = NULL;

  if (be) {
    error("bigendian not supported");
  }

  if (c == 'u' && bits == 8) {
    uint8_t *y = new uint8_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (uint8_t)(clamp(x[i] * 256.0, 0.5, 255.5));
    ret = y;

  } else if (c == 'u' && bits == 16) {
    uint16_t *y = new uint16_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (uint16_t)(clamp(x[i] * 65536.0, 0.5, 65535.5));
    ret = y;

  } else if (c == 'u' && bits == 32) {
    uint32_t *y = new uint32_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (uint32_t)(clamp(x[i] * 4294967296, 0.5, 4294967295.5));
    ret = y;

  } else if (c == 's' && bits == 8) {
    int8_t *y = new int8_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (int8_t)(clamp(x[i] * 256.0, -127.5, 127.5));
    ret = y;

  } else if (c == 's' && bits == 16) {
    int16_t *y = new int16_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (int16_t)(clamp(x[i] * 65536.0, -32767.5, 32767.5));
    ret = y;

  } else if (c == 's' && bits == 32) {
    int32_t *y = new int32_t[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (int32_t)(clamp(x[i] * 2147483648.0, -2147483647.5, 2147483647.5));
    ret = y;

  } else if (c == 'f' && bits == 32) {
    float *y = new float[n];
    for (unsigned int i = 0; i < n; ++i)
      y[i] = (float)x[i];
    ret = y;

  } else if (c == 'f' && bits == 64) {
    double *y = new double[n];
    memcpy(y, x, n * sizeof(double));
    ret = y;

  } else {
    error("unsupported bitfmt");
  }

  return ret;
}

}
