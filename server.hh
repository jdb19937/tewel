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

namespace makemore {

struct Server {
  static void nonblock(int fd);

  int s;
  uint16_t port;

  std::vector<pid_t> pids;

  class Chain *chn;
  int pw, ph;
  std::vector<std::string> ctx;

  Server(const std::vector<std::string> &_cx, int _pw, int _ph);
  ~Server();

  void open();
  void close();
  void bind(uint16_t _port);
  void listen(int backlog = 256);

  void start(unsigned int kids = 1);
  void accept();
  void handle(int c, uint32_t ip);
  void wait();
  void stop();

  void setup();
};

}

#endif
