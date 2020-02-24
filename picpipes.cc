#define __MAKEMORE_PICPIPES_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <assert.h>

#include "picpipes.hh"
#include "youtil.hh"

namespace makemore {

std::string Picreader::cmd = "/opt/makemore/share/tewel/picreader-sample.pl";
std::string Picwriter::cmd = "/opt/makemore/share/tewel/picwriter-sample.pl";

void Picreader::set_cmd(const std::string &_cmd) {
  cmd = _cmd;
}

void Picwriter::set_cmd(const std::string &_cmd) {
  cmd = _cmd;
}



static void save_ppm(FILE *fp, const uint8_t *rgb, unsigned int w, unsigned int h) {
  assert(w > 0);
  assert(h > 0);
  fprintf(fp, "P6\n%d %d\n255\n", w, h);
  size_t ret = fwrite(rgb, 1, 3 * w * h, fp);
  assert(3 * w * h == ret);
}

static bool load_ppm(FILE *fp, uint8_t **rgbp, unsigned int *wp, unsigned int *hp) {
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





Picreader::Picreader() {
  fp = NULL;
  pid = 0;
}
Picwriter::Picwriter() {
  fp = NULL;
  pid = 0;
}


Picreader::~Picreader() {
  close();
}
Picwriter::~Picwriter() {
  close();
}



void Picreader::open(const std::string &_fn) {
  assert:(!fp);

  fn = _fn;

  if (endswith(fn, ".ppm")) {
    pid = 0;
    fp = fopen(fn.c_str(), "r");
    assert(fp);
    return;
  }

  int fd[2];
  int ret = ::pipe(fd);
  assert(ret == 0);

  pid = ::fork();
  if (!pid) {
    ::close(fd[0]);
    if (fd[1] != 1) {
      ::dup2(fd[1], 1);
      ::close(fd[1]);
    }
    ::execlp(cmd.c_str(), cmd.c_str(), fn.c_str(), NULL);
    assert(0);
  }

  ::close(fd[1]);
  fp = fdopen(fd[0], "r");
  assert(fp);
}


void Picwriter::open(const std::string &_fn) {
  assert:(!fp);

  fn = _fn;
  if (endswith(fn, ".ppm")) {
    pid = 0;
    fp = fopen(fn.c_str(), "w");
    assert(fp);
    return;
  }

  int fd[2];
  int ret = ::pipe(fd);
  assert(ret == 0);

  pid = ::fork();
  if (!pid) {
    ::close(fd[1]);
    if (fd[0] != 0) {
      ::dup2(fd[0], 0);
      ::close(fd[0]);
    }

    ::execlp(cmd.c_str(), cmd.c_str(), fn.c_str(), NULL);
    assert(0);
  }

  ::close(fd[0]);
  fp = fdopen(fd[1], "w");
  assert(fp);
}




void Picwriter::close() {
  if (!fp)
    return;
  fclose(fp);
  fp = NULL;

  if (pid > 0) {
    pid_t vpid = waitpid(pid, NULL, 0);
    assert(pid == vpid);
    pid = 0;
  }
}
void Picreader::close() {
  if (!fp)
    return;
  fclose(fp);
  fp = NULL;

  if (pid > 0) {
    pid_t vpid = waitpid(pid, NULL, 0);
    assert(pid == vpid);
    pid = 0;
  }
}




bool Picreader::read(uint8_t **rgbp, unsigned int *wp, unsigned int *hp) {
  return load_ppm(fp, rgbp, wp, hp);
}

bool Picreader::read(uint8_t *rgb, unsigned int w, unsigned int h) {
  uint8_t *new_rgb;
  unsigned int new_w, new_h;
  if (!load_ppm(fp, &new_rgb, &new_w, &new_h))
    return false;
  assert(new_w == w);
  assert(new_h == h);
  memcpy(rgb, new_rgb, w * h * 3);
  delete[] new_rgb;
  return true;
}

void Picwriter::write(const uint8_t *rgb, unsigned int w, unsigned int h) {
  save_ppm(fp, rgb, w, h);
}

}
