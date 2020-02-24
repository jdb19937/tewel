CXX = g++
NVCC = nvcc
OPT = -O6
NVCCFLAGS = $(OPT) -Xcompiler -fPIC
CXXFLAGS = $(OPT) -fPIC
LDFLAGS = -lm

TGT = tewel tewel-cuda-sdl tewel-nocuda-sdl tewel-cuda-nosdl tewel-nocuda-nosdl
OBJ = cortex.o tewel.o random.o youtil.o kleption.o cmdline.o camera.o video.o
HDR = colonel.hh cortex.hh random.hh youtil.hh kleption.hh cmdline.hh display.hh camera.hh video.hh

.PHONY: all
all: $(TGT)

.PHONY: clean
clean:
	rm -f $(OBJ) $(TGT) colonel-cuda.o colonel-nocuda.o display-nosdl.o display-sdl.o

.PHONY: install
install: tewel
	install -d /opt/makemore/bin
	install -m 0755 tewel /opt/makemore/bin
	install -d /opt/makemore/share/tewel
	install -m 0644 README /opt/makemore/share/tewel/README
	install -m 0755 picreader-sample.pl /opt/makemore/share/tewel
	install -m 0755 picwriter-sample.pl /opt/makemore/share/tewel

.PHONY: uninstall
uninstall:
	rm -f /opt/makemore/bin/tewel
	rm -f /opt/makemore/share/tewel/README
	rm -f /opt/makemore/share/tewel/picreader-sample.pl
	rm -f /opt/makemore/share/tewel/picwriter-sample.pl

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

tewel: tewel-cuda-sdl
	ln -f $< $@
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
