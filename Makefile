CXX = g++
NVCC = nvcc
OPT = -g
NVCCFLAGS = $(OPT) -Xcompiler -fPIC
CXXFLAGS = $(OPT) -fPIC
LDFLAGS = -lm

VERSION = 0.1

TGT = tewel tewel-cuda-sdl tewel-nocuda-sdl tewel-cuda-nosdl tewel-nocuda-nosdl identity.gen
OBJ = cortex.o tewel.o random.o youtil.o kleption.o cmdline.o camera.o picpipes.o
HDR = colonel.hh cortex.hh random.hh youtil.hh kleption.hh cmdline.hh display.hh camera.hh picpipes.hh

SRC = \
  cortex.cc tewel.cc random.cc youtil.cc kleption.cc cmdline.cc camera.cc picpipes.cc \
  colonel-cuda.cu colonel-nocuda.cc colonel.inc display-sdl.cc display-nosdl.cc \
  $(HDR) Makefile README LICENSE colonel.inc \
  picreader.pl picwriter.pl decolorize.sh \
  zoom2x.sh zoom4x.sh shrink2x.sh shrink4x.sh degrade2x.sh degrade4x.sh

PACKAGE = tewel_$(VERSION)-1_amd64.deb
TARBALL = tewel-$(VERSION).tar.gz

.PHONY: all
all: $(TGT)

.PHONY: clean
clean:
	rm -f $(OBJ) $(TGT) colonel-cuda.o colonel-nocuda.o display-nosdl.o display-sdl.o
	rm -rf package tarball
	rm -f $(PACKAGE) $(TARBALL)

.PHONY: install
install: all
	install -d $(DESTDIR)/opt/makemore/bin
	install -m 0755 tewel $(DESTDIR)/opt/makemore/bin
	install -d $(DESTDIR)/opt/makemore/share/tewel
	install -m 0644 README $(DESTDIR)/opt/makemore/share/tewel/README
	install -m 0644 LICENSE $(DESTDIR)/opt/makemore/share/tewel/LICENSE
	install -m 0644 identity.gen $(DESTDIR)/opt/makemore/share/tewel
	install -m 0755 picreader.pl $(DESTDIR)/opt/makemore/share/tewel
	install -m 0755 picwriter.pl $(DESTDIR)/opt/makemore/share/tewel
	install -m 0755 *.sh $(DESTDIR)/opt/makemore/share/tewel

.PHONY: uninstall
uninstall:
	rm -f $(DESTDIR)/opt/makemore/bin/tewel
	rm -f $(DESTDIR)/opt/makemore/share/tewel/README
	rm -f $(DESTDIR)/opt/makemore/share/tewel/LICENSE
	rm -f $(DESTDIR)/opt/makemore/share/tewel/picreader.pl
	rm -f $(DESTDIR)/opt/makemore/share/tewel/picwriter.pl

.PHONY: package
package: $(PACKAGE)

$(PACKAGE): $(TARBALL)
	rm -rf package
	mkdir package
	cp tewel-$(VERSION).tar.gz package/
	cd package; tar -xzvf tewel-$(VERSION).tar.gz
	cd package; mv tewel-$(VERSION).tar.gz tewel_$(VERSION).orig.tar.gz
	cd package/tewel-$(VERSION); debuild -us -uc
	cp -f package/$(PACKAGE) $@


.PHONY: tarball
tarball: $(TARBALL)

$(TARBALL):
	rm -rf tarball
	mkdir tarball
	mkdir tarball/tewel-$(VERSION)
	cp -f $(SRC) tarball/tewel-$(VERSION)/
	cp -rpf debian tarball/tewel-$(VERSION)/
	cd tarball; tar -czvf tewel-$(VERSION).tar.gz tewel-$(VERSION)
	cp -f tarball/tewel-$(VERSION).tar.gz $@
	


%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

tewel: tewel-cuda-sdl
	cp -f $< $@
tewel-cuda-sdl: $(OBJ) colonel-cuda.o display-sdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) -lcudart -lSDL2
tewel-nocuda-sdl: $(OBJ) colonel-nocuda.o display-sdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) -lSDL2
tewel-cuda-nosdl: $(OBJ) colonel-cuda.o display-nosdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS) -lcudart
tewel-nocuda-nosdl: $(OBJ) colonel-nocuda.o display-nosdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

$(OBJ): $(HDR)
colonel-cuda.o: colonel.inc
colonel-nocuda.o: colonel.inc


identity.gen: tewel
	rm -f $@
	./tewel new gen=$@
	./tewel push gen=$@ t=iden ic=3 oc=3
