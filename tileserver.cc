#define __MAKEMORE_TILESERVER_CC__ 1
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

#include "tileserver.hh"
#include "youtil.hh"
#include "chain.hh"
#include "colonel.hh"

namespace makemore {

using namespace std;

TileServer::TileServer(const std::vector<std::string> &_ctx, int _cuda, int _kbs) : Server() {
  ctx = _ctx;

  chn = NULL;

  cuda = _cuda;
  kbs = _kbs;
}

void TileServer::prepare() {
  setkdev(cuda >= 0 ? cuda : kndevs() > 1 ? 1 : 0);
  setkbs(kbs);

  chn = new Chain;
  for (auto ctxfn : ctx)
    chn->push(ctxfn, O_RDONLY);
  chn->prepare(512, 512);

  assert(chn->head->iw == 512);
  assert(chn->head->ih == 512);
  assert(chn->head->ic == 3);

  Server::prepare();
}

bool TileServer::handle(Client *client) {
  return true;
}

}
