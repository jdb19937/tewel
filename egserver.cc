#define __MAKEMORE_EGSERVER_CC__ 1
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

#include "egserver.hh"
#include "youtil.hh"
#include "chain.hh"
#include "colonel.hh"

namespace makemore {

using namespace std;

EGServer::EGServer(const std::vector<std::string> &_ctx, int _pw, int _ph, int _cuda, int _kbs, double _reload, bool _pngout) : Server() {
  pw = _pw;
  ph = _ph;
  ctx = _ctx;

  chn = NULL;

  cuda = _cuda;
  kbs = _kbs;

  reload = _reload;
  pngout = _pngout;
  last_reload = 0.0;
}

void EGServer::prepare() {
  setkdev(cuda >= 0 ? cuda : kndevs() > 1 ? 1 : 0);
  setkbs(kbs);

  chn = new Chain;
  for (auto ctxfn : ctx)
    chn->push(ctxfn, O_RDONLY);
  chn->prepare(pw, ph);

  last_reload = now();

  Server::prepare();
}


void EGServer::extend() {
  if (reload > 0 && last_reload + reload < now()) {
    info("reloading chain");
    chn->load();
    last_reload = now();
  }
}

bool EGServer::handle(Client *client) {
  int inpn = chn->head->iw * chn->head->ih * chn->head->ic;

  assert(inpn > 0);

  while (client->can_read(256 + inpn * sizeof(double))) {
    uint8_t *hdr = client->inpbuf;

    int32_t stop;
    memcpy(&stop, hdr, 4);
    if (stop > 0 || stop < -4)
      return false;

    Paracortex *tail = chn->ctxv[chn->ctxv.size() + stop - 1];
    int outn = tail->ow * tail->oh * tail->oc;
    assert(outn > 0);
    double *buf = new double[outn];
    uint8_t *rgb = new uint8_t[outn];

    double *dat = (double *)(client->inpbuf + 256);

    enk(dat, inpn, chn->kinp());
    assert(client->read(NULL, 256 + inpn * sizeof(double)));

    info(fmt("synthing inpn=%d outn=%d", inpn, outn));
    chn->synth(stop);
    info("done with synth");

    dek(tail->kout(), outn, buf);
    if (pngout) {
      for (int i = 0; i < outn; ++i)
        rgb[i] = (uint8_t)clamp(buf[i] * 256.0, 0, 255);

      assert(tail->oc == 3);
      uint8_t *png;
      unsigned long pngn;
      rgbpng(rgb, tail->ow, tail->oh, &png, &pngn);

      delete[] buf;
      delete[] rgb;
    
      if (!client->write(png, pngn))
        return false;
    } else {
      if (!client->write((uint8_t *)buf, outn * sizeof(double)))
        return false;
    }
  }

  return true;
}

}
