#ifndef __MAKEMORE_EGSERVER_HH__
#define __MAKEMORE_EGSERVER_HH__ 1

#include "client.hh"
#include "server.hh"

namespace makemore {

struct EGServer : Server {
  class Chain *chn;
  int pw, ph;
  std::vector<std::string> ctx;

  int cuda;
  int kbs;
  double reload;
  double last_reload;
  bool pngout;

  EGServer(const std::vector<std::string> &_cx, int _pw, int _ph, int _cuda, int _kbs, double _reload = 0.0, bool _pngout = false);

  bool handle(class Client *);
  void prepare();
  void extend();
};

}

#endif
