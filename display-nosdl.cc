#define __MAKEMORE_DISPLAY_NOSDL_CC__ 1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "youtil.hh"
#include "display.hh"

class SDL_Window {
  int x;
};

class SDL_Renderer {
  int x;
};

namespace makemore {

Display::Display() {
  window = NULL;
  renderer = NULL;
  w = 0;
  h = 0;
  is_open = false;
}

Display::~Display() {
  close();
}

void Display::open() {
  error("no display");
}

void Display::close() {

}

void Display::update(const uint8_t *rgb, unsigned int pw, unsigned int ph) {
  error("no display");
}

void Display::present() {
  error("no display");
}

bool Display::escpressed() {
  return true;
}

}
