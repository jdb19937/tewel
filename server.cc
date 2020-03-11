#define __MAKEMORE_SERVER_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/signal.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <string.h>

#include <string>
#include <vector>
#include <list>

#include "server.hh"
#include "youtil.hh"
#include "chain.hh"
#include "colonel.hh"

namespace makemore {

using namespace std;

static void nonblock(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  assert(flags > 0);
  int ret = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  assert(ret == 0);
}

static inline double realtime() {
  clock_t c = clock();
  return ((double)c / (double)CLOCKS_PER_SEC);
}

Server::Server(const std::vector<std::string> &_ctx, int _pw, int _ph) {
  port = 0;
  s = -1;

  pw = _pw;
  ph = _ph;
  ctx = _ctx;

  chn = NULL;
}

Server::~Server() {
  close();
}


void Server::open() {
  int ret;

  this->close();
  s = socket(PF_INET, SOCK_STREAM, 0);
  assert(s >= 0);

  int reuse = 1;
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);

  nonblock(s);
}

void Server::close() {
  if (s >= 0) {
    this->stop();
    this->wait();

    ::close(s);
    port = 0;
    s = -1;
  }
}

void Server::bind(uint16_t _port) {
  assert(s >= 0);

  int ret;
  struct sockaddr_in sin;

  sin.sin_family = AF_INET;
  sin.sin_port = htons(_port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = ::bind(s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);

  port = _port;
}

void Server::listen(int backlog) {
  assert(s >= 0);
  int ret = ::listen(s, backlog);
  assert(ret == 0);
}

static void sigpipe(int) {
  info("SIGPIPE");
}

void Server::main() {
  chn = new Chain;
  for (auto ctxfn : ctx)
    chn->push(ctxfn, O_RDONLY);
  chn->prepare(pw, ph);

  signal(SIGPIPE, sigpipe);

  while (1) {
    this->select();

    if (FD_ISSET(s, fdsets + 0))
      while (this->accept());

    auto ci = clients.begin();
    while (ci != clients.end()) {
      Client *client = *ci;

      if (FD_ISSET(client->s, fdsets + 0)) {
        if (!client->slurp()) {
          info("slurp failed, deleting client");
          delete client;
          clients.erase(ci++);
          continue;
        }
      }

      if (FD_ISSET(client->s, fdsets + 1)) {
        if (!client->flush()) {
          info("flush failed, deleting client");
          delete client;
          clients.erase(ci++);
          continue;
        }
      }

      ++ci;
    }


    ci = clients.begin();
    while (ci != clients.end()) {
      Client *client = *ci;
      if (!handle(client)) {
        info("handle failed, deleting client");
        delete client;
        clients.erase(ci++);
        continue;
      }
      ++ci;
    }

  }
}


void Server::start(unsigned int kids) {
  for (unsigned int i = 0; i < kids; ++i) {
    fprintf(stderr, "forking\n");
    if (pid_t pid = fork()) {
      pids.push_back(pid);
      fprintf(stderr, "forked pid=%d\n", pid);
      continue;
    }

    this->main();

    assert(0);
  }
}

static std::string ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return std::string(buf);
}

bool Server::accept() {
  struct sockaddr_in cin;
  socklen_t cinlen = sizeof(cin);
  int c = ::accept(s, (struct sockaddr *)&cin, &cinlen);
  if (c < 0)
    return false;

  nonblock(c);

  uint32_t ip;
  assert(sizeof(struct in_addr) == 4);
  memcpy(&ip, &cin.sin_addr, 4);

  info(fmt("accepted %s", ipstr(ip).c_str()));
  
  Client *client = new Client(c, ip);
  clients.push_back(client);
  return true;
}

bool Server::handle(Client *client) {
  int inpn = chn->head->iw * chn->head->ih * chn->head->ic * sizeof(double);
  int outn = chn->tail->ow * chn->tail->oh * chn->tail->oc * sizeof(double);

  assert(inpn > 0);
  assert(outn > 0);

  while (client->can_read(inpn) && client->can_write(outn)) {
    info(fmt("synthing inpn=%d outn=%d", inpn, outn));
    enk(client->inpbuf, inpn, (uint8_t *)chn->kinp());
    chn->synth();
    dek((uint8_t *)chn->kout(), outn, client->outbuf + client->outbufk);

    assert(client->read(NULL, inpn));
    client->outbufk += outn;
  }

  return true;
}

void Server::wait() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    info(fmt("waiting for pid=%d", pid));
    assert(pid == ::waitpid(pid, NULL, 0));
    info("waited");
  }
  pids.clear();
}

void Server::stop() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    info(fmt("killing pid=%d", pid));
    ::kill(pid, 9);
    info("killed");
  }

  this->wait();
}

void Server::select() {
  FD_ZERO(fdsets + 0);
  FD_ZERO(fdsets + 1);
  FD_ZERO(fdsets + 2);

  int nfds = s + 1;
  FD_SET(s, fdsets + 0);

  for (auto client : clients) {
    if (client->s >= nfds)
      nfds = client->s + 1;
    if (client->need_flush())
      FD_SET(client->s, fdsets + 1);
    if (client->need_slurp())
      FD_SET(client->s, fdsets + 0);
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 50000;

  (void)::select(nfds, fdsets + 0, fdsets + 1, fdsets + 2, &timeout);
}


}
