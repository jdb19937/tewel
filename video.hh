#ifndef __MAKEMORE_VIDEO_HH__
#define __MAKEMORE_VIDEO_HH__ 1

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>

#include <string>

namespace makemore {

struct Video {
  std::string fn;
  FILE *fp;
  int mode;
  pid_t pid;

  Video();
  ~Video();

  void open(const std::string &_fn, int _mode = O_RDONLY);
  void open_read(const std::string &_fn);
  void open_write(const std::string &_fn);

  void close();

  bool read(uint8_t **rgbp, unsigned int *wp, unsigned int *hp);
  bool read(uint8_t *rgb, unsigned int w, unsigned int h);
  void write(const uint8_t *rgb, unsigned int w, unsigned int h);
};

}

#endif
