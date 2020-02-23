#ifndef __MAKEMORE_CAMERA_HH__
#define __MAKEMORE_CAMERA_HH__ 1

namespace makemore {

struct Camera {
  int fd;
  std::string devfn;

  unsigned int w, h;

  struct Buffer {
    void *start;
    size_t length;
  } *buffers;
  uint8_t *tmp;

  unsigned int n_buffers;

  Camera(const std::string &_devfn = "/dev/video0", int _w = 640, int _h = 480);
  ~Camera();

  void open();
  void close();

  void read(uint8_t *rgb, bool reflect = false);
};

}

#endif
