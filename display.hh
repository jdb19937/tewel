#ifndef __MAKEMORE_DISPLAY_HH__
#define __MAKEMORE_DISPLAY_HH__ 1

#include <stdio.h>
#include <stdint.h>

class SDL_Window;
class SDL_Renderer;

namespace makemore {

struct Display {
  class SDL_Window *window;
  class SDL_Renderer *renderer;

  unsigned int w, h;
  bool is_open;

  Display();
  ~Display();

  void open();
  void close();

  void update(const uint8_t *rgb, unsigned int w, unsigned int h);
  void present();

  bool escpressed();
};

}

#endif
