#ifndef __MAKEMORE_CLIENT_HH__
#define __MAKEMORE_CLIENT_HH__ 1

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

struct Client {
  int s;
  uint32_t ip;

  uint8_t *outbuf;
  unsigned int outbufk, outbufn;

  uint8_t *inpbuf;
  unsigned int inpbufk, inpbufn;

  Client(int _s, uint32_t _ip) {
    s = _s;
    ip = _ip;

    outbufn = (10 << 20);
    outbufk = 0;
    outbuf = new uint8_t[outbufn];

    inpbufn = (1 << 20);
    inpbufk = 0;
    inpbuf = new uint8_t[inpbufn];
  }

  ~Client() {
    delete[] inpbuf;
    delete[] outbuf;
    ::close(s);
  }

  bool need_flush() {
    return (outbufk > 0);
  }

  bool need_slurp() {
    return (inpbufk < inpbufn);
  }

  bool write(const uint8_t *buf, unsigned int len);
  bool can_write(unsigned int len) {
    return (outbufk + len <= outbufn);
  }
  bool read(uint8_t *buf, unsigned int len);
  bool can_read(unsigned int len) {
    return (inpbufk >= len);
  }
  bool slurp();
  bool flush();
};

}

#endif
