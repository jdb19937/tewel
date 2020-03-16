#ifndef __MAKEMORE_SERVER_HH__
#define __MAKEMORE_SERVER_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <list>

#include "client.hh"

namespace makemore {

struct Server {
  int s;
  uint16_t port;

  std::vector<pid_t> pids;

  class Chain *chn;
  int pw, ph;
  std::vector<std::string> ctx;


  fd_set fdsets[3];
  std::list<Client*> clients;

  int cuda;
  int kbs;
  double reload;
  double last_reload;
  bool pngout;

  Server(const std::vector<std::string> &_cx, int _pw, int _ph, int _cuda, int _kbs, double _reload = 0.0, bool _pngout = false);
  ~Server();

  void open();
  void close();
  void bind(uint16_t _port);
  void listen(int backlog = 256);
  void select();

  void start(unsigned int kids = 1);
  bool accept();
  bool handle(class Client *);
  void wait();
  void stop();
  void main();
};

}

#endif
