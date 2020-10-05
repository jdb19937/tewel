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

  fd_set fdsets[3];
  std::list<Client*> clients;

  Server();
  virtual ~Server();

  virtual void open();
  virtual void close();
  virtual void bind(uint16_t _port);
  virtual void listen(int backlog = 256);
  virtual void select();

  virtual void start(unsigned int kids = 1);
  virtual bool accept();
  virtual bool handle(class Client *);
  virtual void wait();
  virtual void stop();
  virtual void main();
  virtual void prepare();
  virtual void extend();
};

}

#endif
