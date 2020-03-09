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

void Server::nonblock(int fd) {
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
  fprintf(stderr, "SIGPIPE\n");
}

void Server::setup() {
  chn = new Chain;
  for (auto ctxfn : ctx)
    chn->push(ctxfn, O_RDONLY);
  chn->prepare(pw, ph);

  signal(SIGPIPE, sigpipe);
}


void Server::start(unsigned int kids) {
  for (unsigned int i = 0; i < kids; ++i) {
    fprintf(stderr, "forking\n");
    if (pid_t pid = fork()) {
      pids.push_back(pid);
      fprintf(stderr, "forked pid=%d\n", pid);
      continue;
    }

    this->setup();
    while (1) {
      info("calling accept");
      this->accept();
      info("done with accept");
    }

    assert(0);
  }
}

static std::string ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return std::string(buf);
}

void Server::accept() {
  fprintf(stderr, "accepting\n");

  struct sockaddr_in cin;
  socklen_t cinlen = sizeof(cin);
  int c = ::accept(s, (struct sockaddr *)&cin, &cinlen);
  if (c < 0) {
    error("accept failed");
    return;
  }

  uint32_t ip;
  assert(sizeof(struct in_addr) == 4);
  memcpy(&ip, &cin.sin_addr, 4);
  info(fmt("accepted %s", ipstr(ip).c_str()));

  handle(c, ip);

  ::close(c);
}

void Server::handle(int c, uint32_t ip) {
  int inpn = chn->head->iw * chn->head->ih * chn->head->ic * sizeof(double);
  int outn = chn->tail->ow * chn->tail->oh * chn->tail->oc * sizeof(double);

  assert(inpn > 0);
  assert(outn > 0);

  uint8_t *inpbuf = new uint8_t[inpn];
  uint8_t *outbuf = new uint8_t[outn];

  while (1) {
    int inpi = 0;
    while (inpi < inpn) {
      info(fmt("reading %d", inpn - inpi));
      int ret = ::read(c, inpbuf + inpi, inpn - inpi);
      info(fmt("done reading, ret=%d", ret));
      if (ret < 1) {
        info("failed to read, closing");
        delete[] inpbuf;
        delete[] outbuf;
        return;
      }
      inpi += ret;
    }

    info("synthing");
    enk(inpbuf, inpn, (uint8_t *)chn->kinp());
    chn->synth();
    dek((uint8_t *)chn->kout(), outn, outbuf);

    info("writing");
    int outi = 0;
    while (outi < outn) {
      info(fmt("writing %d", outn - outi));
      int ret = ::write(c, outbuf + outi, outn - outi);
      info(fmt("done writing, got %d", ret));
      if (ret < 1) {
        info("failed to write, closing");
        delete[] inpbuf;
        delete[] outbuf;
        return;
      }
      outi += ret;
    }
  }
}

void Server::wait() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "waiting for pid=%d\n", pid);
    assert(pid == ::waitpid(pid, NULL, 0));
    fprintf(stderr, "waited\n");
  }
  pids.clear();
}

void Server::stop() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "killing pid=%d\n", pid);
    ::kill(pid, 9);
    fprintf(stderr, "killed\n");
  }

  this->wait();
}

}
