#ifndef __MAKEMORE_TILESERVER_HH__
#define __MAKEMORE_TILESERVER_HH__ 1

#include "client.hh"
#include "server.hh"

namespace makemore {

struct TileServer : Server {
  class Chain *chn;
  int pw, ph;
  std::vector<std::string> ctx;

  int cuda;
  int kbs;

  TileServer(const std::vector<std::string> &_cx, int _cuda, int _kbs);

  bool handle(class Client *);
  void prepare();
};

}

#endif
