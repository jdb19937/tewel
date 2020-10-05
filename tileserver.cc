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

TileServer::TileServer(const std::string &_root, const std::string &_dir, const std::vector<std::string> &_ctx, int _cuda, int _kbs) : Server() {
  root = _root;
  assert(root.length() < 4000);

  dir = _dir;
  assert(dir.length() < 4000);


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
  assert(chn->tail->ow == 512);
  assert(chn->tail->oh == 512);
  assert(chn->tail->oc == 3);

  Server::prepare();
}

static inline uint8_t dtob(double d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((uint8_t)(d + 0.5));
}

std::string TileServer::tilefn(uint32_t x, uint32_t y, uint32_t z) {
  assert(z < 32);
  x &= (1 << z) - 1;
  y &= (1 << z) - 1;

  if (z > 0) {
    char fn[4096];
    sprintf(fn, "%s/tile.%u.%u.%u.png", dir.c_str(), x, y, z);
    return fn;
  } else {
    return root;
  }
}

void TileServer::put_tile(uint32_t x, uint32_t y, uint32_t z, const uint8_t *rgb) {
  assert(z > 0);
  assert(z < 32);
  x &= (1 << z) - 1;
  y &= (1 << z) - 1;

  std::string fn = tilefn(x, y, z);
  {
    FILE *fp = fopen(fn.c_str(), "r");
    if (fp) {
      fclose(fp);
      return;
    }
  }

  uint8_t *png = NULL;
  unsigned long pngn;

  rgbpng(rgb, 256, 256, &png, &pngn);

  spit(png, pngn, fn);

  delete[] png;
}

uint8_t *TileServer::tile_rgb(uint32_t x, uint32_t y, uint32_t z, bool opt) {
  assert(z < 32);
  x &= (1 << z) - 1;
  y &= (1 << z) - 1;

  if (opt) {
    if (FILE *fp = fopen(tilefn(x, y, z).c_str(), "r")) {
      unsigned long pngn;
      uint8_t *png = slurp(fp, &pngn);
      fclose(fp);

      unsigned int w = 0, h = 0;
      uint8_t *rgb;
      pngrgb(png, pngn, &w, &h, &rgb);
      assert(w == 256 && h == 256);
      delete[] png;

      return rgb;
    }
  }

  assert(z > 0);

  uint32_t px = (x >> 1) + ((x & 1) ? 0 : -1);
  uint32_t py = (y >> 1) + ((y & 1) ? 0 : -1);
  uint32_t pz = z - 1;
  px &= (1 << pz) - 1;
  py &= (1 << pz) - 1;

  uint8_t *prgbp[4];
  prgbp[0] = tile_rgb(px, py, pz, true);
  prgbp[1] = tile_rgb(px + 1, py, pz, true);
  prgbp[2] = tile_rgb(px, py + 1, pz, true);
  prgbp[3] = tile_rgb(px + 1, py + 1, pz, true);


  double *qrgb = new double[512 * 512 * 3];
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      for (int c = 0; c < 3; ++c) {
        qrgb[c + 3 * (x + 512 * y)] = (double)prgbp[0][c + 3 * (x + 256 * y)] / 256.0;
        qrgb[c + 3 * (256 + x + 512 * y)] = (double)prgbp[1][c + 3 * (x + 256 * y)] / 256.0;
        qrgb[c + 3 * (x + 512 * (256 + y))] = (double)prgbp[2][c + 3 * (x + 256 * y)] / 256.0;
        qrgb[c + 3 * (256 + x + 512 * (256 + y))] = (double)prgbp[3][c + 3 * (x + 256 * y)] / 256.0;
      }
    }
  }





  assert(chn->head->iw == 512);
  assert(chn->head->ih == 512);
  assert(chn->head->ic == 3);
  assert(chn->tail->ow == 512);
  assert(chn->tail->oh == 512);
  assert(chn->tail->oc == 3);

  enk(qrgb, 512 * 512 * 3, chn->kinp());
  chn->synth();
  dek(chn->kout(), 512 * 512 * 3, qrgb);

  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      for (int c = 0; c < 3; ++c) {
        prgbp[0][c + 3 * (x + 256 * y)] = dtob(qrgb[c + 3 * (x + 512 * y)]);
        prgbp[1][c + 3 * (x + 256 * y)] = dtob(qrgb[c + 3 * (256 + x + 512 * y)]);
        prgbp[2][c + 3 * (x + 256 * y)] = dtob(qrgb[c + 3 * (x + 512 * (256 + y))]);
        prgbp[3][c + 3 * (x + 256 * y)] = dtob(qrgb[c + 3 * (256 + x + 512 * (256 + y))]);
      }
    }
  }

  delete[] qrgb;


  uint32_t pxp[4], pyp[4];
  pxp[0] = (px * 2 + 1) & ((1 << z) - 1);
  pyp[0] = (py * 2 + 1) & ((1 << z) - 1);
  pxp[1] = (px * 2 + 2) & ((1 << z) - 1);
  pyp[1] = (py * 2 + 1) & ((1 << z) - 1);
  pxp[2] = (px * 2 + 1) & ((1 << z) - 1);
  pyp[2] = (py * 2 + 2) & ((1 << z) - 1);
  pxp[3] = (px * 2 + 2) & ((1 << z) - 1);
  pyp[3] = (py * 2 + 2) & ((1 << z) - 1);
  
  put_tile(pxp[0], pyp[0], z, prgbp[0]);
  put_tile(pxp[1], pyp[1], z, prgbp[1]);
  put_tile(pxp[2], pyp[2], z, prgbp[2]);
  put_tile(pxp[3], pyp[3], z, prgbp[3]);

  uint8_t *rgb = NULL;
  if (pxp[0] == x && pyp[0] == y) {
    rgb = prgbp[0];
    delete prgbp[1];
    delete prgbp[2];
    delete prgbp[3];
  } else if (pxp[1] == x && pyp[1] == y) {
    delete prgbp[0];
    rgb = prgbp[1];
    delete prgbp[2];
    delete prgbp[3];
  } else if (pxp[2] == x && pyp[2] == y) {
    delete prgbp[0];
    delete prgbp[1];
    rgb = prgbp[2];
    delete prgbp[3];
  } else if (pxp[3] == x && pyp[3] == y) {
    delete prgbp[0];
    delete prgbp[1];
    delete prgbp[2];
    rgb = prgbp[3];
  } else {
    assert(0);
  }

  return rgb;
}

void TileServer::tile_png(uint32_t x, uint32_t y, uint32_t z, uint8_t **pngp, unsigned long *pngnp) {
  assert(z < 32);
  x &= (1 << z) - 1;
  y &= (1 << z) - 1;

  {
    if (FILE *fp = fopen(tilefn(x, y, z).c_str(), "r")) {
      *pngp = slurp(fp, pngnp);
      fclose(fp);
      return;
    }
  }
  assert(z > 0);

  uint8_t *rgb = tile_rgb(x, y, z, false);
  delete[] rgb;

  {
    if (FILE *fp = fopen(tilefn(x, y, z).c_str(), "r")) {
      *pngp = slurp(fp, pngnp);
      fclose(fp);
      return;
    }
  }

  assert(0);
}

bool TileServer::handle(Client *client) {
  while (client->can_read(12)) {
    uint32_t xyz[3];
    client->read((uint8_t *)xyz, 12);
    uint32_t x = ntohl(xyz[0]);
    uint32_t y = ntohl(xyz[1]);
    uint32_t z = ntohl(xyz[2]);

    uint8_t *png;
    unsigned long pngn;
    tile_png(x, y, z, &png, &pngn);

    if (!client->can_write(pngn + 4)) {
      delete[] png;
      return false;
    }
    uint32_t len = htonl(pngn);
    if (!client->write((uint8_t *)&len, 4)) {
      delete[] png;
      return false;
    }
    if (!client->write(png, pngn)) {
      delete[] png;
      return false;
    }
    delete[] png;
  }

  return true;
}

}
