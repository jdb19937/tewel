#define __MAKEMORE_DISPLAY_SDL_CC__ 1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <SDL2/SDL.h>

#include "display.hh"

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
  assert(!is_open);

  int err;
  err = SDL_Init(SDL_INIT_EVERYTHING);
  assert(err == 0);

  SDL_DisplayMode *dm = new SDL_DisplayMode();
  err = SDL_GetCurrentDisplayMode(0, (SDL_DisplayMode *)dm);
  assert(err == 0);
  w = dm->w;
  h = dm->h;
  delete dm;
  // fprintf(stderr, "w=%u h=%u\n", w, h);

  assert(!window);
  window = SDL_CreateWindow("tewel", 0, 0, w, h, 0);
  assert(window);

  err = SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
  assert(err == 0);

  assert(!renderer);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  assert(renderer);

  SDL_RenderSetLogicalSize(renderer, w, h );

  is_open = true;
}

void Display::close() {
  if (!is_open) 
    return;

  assert(renderer);
  SDL_DestroyRenderer(renderer);
  renderer = NULL;

  assert(window);
  SDL_DestroyWindow(window);
  window = NULL;

  is_open = false;
}

void Display::update(const uint8_t *rgb, unsigned int pw, unsigned int ph) {
  assert(is_open);

  double asp = (double)w / (double)pw;
  if (asp > (double)h / (double)ph)
    asp = (double)h / (double)ph;

  SDL_Rect drect;
  drect.x = (double)w / 2.0 - asp * (double)pw / 2.0;
  drect.y = (double)h / 2.0 - asp * (double)ph / 2.0;
  drect.w = asp * (double)pw;
  drect.h = asp * (double)ph;
  SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(
    (void *)rgb, pw, ph, 24, pw * 3, 0xff, 0xff00, 0xff0000, 0
  );

  SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, surf);

  SDL_Rect srect;
  srect.x = 0;
  srect.y = 0;
  srect.w = pw;
  srect.h = ph;
  SDL_RenderCopy(renderer, text, &srect, &drect);

  SDL_DestroyTexture(text);
  SDL_FreeSurface(surf);
}

void Display::present() {
  assert(is_open);
  SDL_RenderPresent(renderer);
}

bool Display::done() {
  assert(is_open);
  SDL_Event event;
  while (SDL_PollEvent(&event))
    if (event.type == SDL_KEYDOWN) {
      int k = event.key.keysym.sym;
      if (k == SDLK_ESCAPE)
        return true;
    }
  return false;
}

}
