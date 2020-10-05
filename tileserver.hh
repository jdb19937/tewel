#ifndef __MAKEMORE_TILESERVER_HH__
#define __MAKEMORE_TILESERVER_HH__ 1

#include "client.hh"
#include "server.hh"

namespace makemore {

struct TileServer : Server {
  std::string root;
  std::string dir;

  class Chain *chn;
  int pw, ph;
  std::vector<std::string> ctx;

  int cuda;
  int kbs;

  TileServer(const std::string &_root, const std::string &_dir, const std::vector<std::string> &_cx, int _cuda, int _kbs);

  bool handle(class Client *);
  void prepare();

  std::string tilefn(uint32_t x, uint32_t y, uint32_t z);

  void put_tile(uint32_t x, uint32_t y, uint32_t z, const uint8_t *rgb);

  uint8_t *tile_rgb(uint32_t x, uint32_t y, uint32_t z, bool opt);
  void tile_png(uint32_t x, uint32_t y, uint32_t z, uint8_t **jpgp, unsigned long *jpgnp);
};

}

#endif
