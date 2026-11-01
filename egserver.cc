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
#include <sys/stat.h>
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


// true if any ctx file in the chain was replaced on disk (new inode), e.g. by `tewel make clobber=1`
static bool chain_replaced(Chain *chn) {
  for (auto pc : chn->ctxv)
    for (auto c : pc->ctxv) {
      struct stat sf, sd;
      if (::stat(c->fn.c_str(), &sf) != 0 || ::fstat(c->fd, &sd) != 0)
        continue;
      if (sf.st_ino != sd.st_ino || sf.st_dev != sd.st_dev)
        return true;
    }
  return false;
}

void EGServer::extend() {
  if (reload > 0 && last_reload + reload < now()) {
    if (chain_replaced(chn)) {
      info("ctx file replaced on disk, reopening chain");
      delete chn;
      chn = new Chain;
      for (auto ctxfn : ctx)
        chn->push(ctxfn, O_RDONLY);
      chn->prepare(pw, ph);
    } else {
      info("reloading chain");
      chn->load();
    }
    last_reload = now();
  }
}

bool EGServer::handle(Client *client) {
  int inpn = chn->head->iw * chn->head->ih * chn->head->ic;

  assert(inpn > 0);

  // wire format is little-endian doubles (the peaple client packs 'd*'); device buffers are float
  while (client->can_read(256 + inpn * sizeof(double))) {
    uint8_t *hdr = client->inpbuf;

    int32_t stop;
    memcpy(&stop, hdr, 4);
    if (stop > 0 || stop < -4)
      return false;
    if ((int)chn->ctxv.size() + stop - 1 < 0) {
      warning(fmt("stop=%d exceeds chain length %d, dropping client", stop, (int)chn->ctxv.size()));
      return false;
    }

    Paracortex *tail = chn->ctxv[chn->ctxv.size() + stop - 1];
    int outn = tail->ow * tail->oh * tail->oc;
    assert(outn > 0);
    float *buf = new float[outn];
    uint8_t *rgb = new uint8_t[outn];

    const double *ddat = (const double *)(client->inpbuf + 256);
    float *dat = new float[inpn];
    for (int i = 0; i < inpn; ++i)
      dat[i] = (float)ddat[i];

    enk(dat, inpn, chn->kinp());
    Cortex::set_noise_seed(Cortex::hash_input(dat, inpn));   // same input, same noise, same picture
    delete[] dat;
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
      double *dbuf = new double[outn];
      for (int i = 0; i < outn; ++i)
        dbuf[i] = buf[i];
      bool ok = client->write((uint8_t *)dbuf, outn * sizeof(double));
      delete[] dbuf;
      delete[] buf;
      delete[] rgb;
      if (!ok)
        return false;
    }
  }

  return true;
}

}
