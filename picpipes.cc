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

Picreader::Picreader(const std::string &_cmd) {
  cmd = _cmd;
  fp = NULL;
  pid = 0;
}

Picreader::~Picreader() {
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



Picwriter::Picwriter(const std::string &_cmd) {
  cmd = _cmd;
  fp = NULL;
  pid = 0;
}

Picwriter::~Picwriter() {
  close();
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

void Picwriter::write(const uint8_t *rgb, unsigned int w, unsigned int h) {
  save_ppm(fp, rgb, w, h);
}

}
