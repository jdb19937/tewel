#define __MAKEMORE_CLIENT_CC__ 1

#include <assert.h>
#include <string.h>

#include "client.hh"
#include "youtil.hh"

namespace makemore {

bool Client::read(uint8_t *buf, unsigned int len) {
  if (inpbufk < len)
    return false;
  if (buf)
    memcpy(buf, inpbuf, len);
  memmove(inpbuf, inpbuf + len, inpbufk - len);
  inpbufk -= len;
  return true;
}

bool Client::slurp() {
  int ret = ::read(s, inpbuf + inpbufk, inpbufn - inpbufk);
  info(fmt("slurped ret=%d", ret));
  if (ret == 0)
    return false;
  if (ret < 0) {
    if (ret != EAGAIN)
      return false;
    ret = 0;
  }

  inpbufk += ret;
  return true;
}

bool Client::flush() {
  if (outbufk == 0)
    return true;
  int ret = ::write(s, outbuf, outbufk);
  info(fmt("flushed ret=%d", ret));

  if (ret == 0)
    return false;
  if (ret < 0) {
    if (ret != EAGAIN)
      return false;
    ret = 0;
  }
  if (ret == outbufk) {
    outbufk = 0;
    return true;
  }
  assert(ret < outbufk);

  memmove(outbuf, outbuf + ret, outbufk - ret);
  outbufk -= ret;
  return true;
}

  
  


bool Client::write(const uint8_t *buf, unsigned int len) {
  if (outbufk > 0) {
    if (outbufk + len > outbufn)
      return false;
    memcpy(outbuf + outbufk, buf, len);
    outbufk += len;
    return true;
  }

  int ret = ::write(s, buf, len);
  if (ret < 0) {
    if (errno != EAGAIN)
      return false;
    ret = 0;
  }
  if (ret == len)
    return true;
  assert(ret < len);

  buf += ret;
  len -= ret;

  if (outbufk + len > outbufn)
    return false;
  memcpy(outbuf + outbufk, buf, len);
  outbufk += len;
  return true;
}

}
