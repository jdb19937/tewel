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


void save_ppm(FILE *fp, uint8_t *rgb, unsigned int w, unsigned int h) {
  assert(w > 0);
  assert(h > 0);
  fprintf(fp, "P6\n%d %d\n255\n", w, h);
  size_t ret = fwrite(rgb, 1, 3 * w * h, fp);
  assert(3 * w * h == ret);
}

bool load_ppm(FILE *fp, uint8_t **rgbp, unsigned int *wp, unsigned int *hp) {
  int ret = getc(fp);
  if (ret == EOF)
    return false;
  assert(ret == 'P');
  assert(getc(fp) == '6');
  assert(isspace(getc(fp)));

  int c;
  char buf[256], *p;

  p = buf;
  do { *p = getc(fp); } while (isspace(*p));
  assert(isdigit(*p));
  ++p;

  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  unsigned int w = atoi(buf);

  p = buf;
  do { *p = getc(fp); } while (isspace(*p));
  assert(isdigit(*p));
  ++p;

  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  unsigned int h = atoi(buf);

  p = buf;
  do { *p = getc(fp); } while (isspace(*p));
  assert(isdigit(*p));
  ++p;

  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  assert(!strcmp(buf, "255"));

  *rgbp = new uint8_t[3 * w * h];
  assert(3 * w * h == fread(*rgbp, 1, 3 * w * h, fp));

  *wp = w;
  *hp = h;
  return true;
}

void load_img(const std::string& fn, uint8_t **rgbp, unsigned int *wp, unsigned int *hp) {
  assert(fn[0] != '-');

  if (endswith(fn, ".ppm")) {
    FILE *fp = ::fopen(fn.c_str(), "r");
    assert(fp);
    assert(load_ppm(fp, rgbp, wp, hp));
    fclose(fp);
    return;
  }

  const char *magick = getenv("TEWEL_MAGICK");
  if (!magick)
    magick = "convert";

  int fds[2];
  int ret = ::pipe(fds);
  pid_t pid = ::fork();

  if (!pid) {
    ::close(fds[0]);
    ::close(1);
    assert(1 == ::dup2(fds[1], 1));
    ::close(fds[1]);
    ::execlp(
      magick, magick,
      fn.c_str(),
      "-depth", "8", "+set", "comment",
      "ppm:-",
      NULL
    );
    assert(0);
  }

  ::close(fds[1]);
  FILE *fp = ::fdopen(fds[0], "r");
  assert(fp);

  assert(load_ppm(fp, rgbp, wp, hp));

  fclose(fp);

  int wstatus;
  ret = ::waitpid(pid, &wstatus, 0); 
  assert(ret == pid);
  assert(WEXITSTATUS(wstatus) == 0);
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

}
