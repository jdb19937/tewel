#ifndef __MAKEMORE_PICPIPES_HH__
#define __MAKEMORE_PICPIPES_HH__ 1

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>

#include <string>

namespace makemore {

struct Picreader {
  static std::string cmd, arg;

  std::string fn;
  FILE *fp;
  pid_t pid;

  Picreader(const std::string &_cmd);
  ~Picreader();

  void open(const std::string &_fn);
  void close();

  bool read(uint8_t **rgbp, unsigned int *wp, unsigned int *hp);
  bool read(uint8_t *rgb, unsigned int w, unsigned int h);
};

struct Picwriter {
  static std::string cmd, arg;

  std::string fn;
  FILE *fp;
  pid_t pid;

  Picwriter(const std::string &_cmd);
  ~Picwriter();

  void open(const std::string &_fn);
  void close();

  void write(const uint8_t *rgb, unsigned int w, unsigned int h);
};

}

#endif
