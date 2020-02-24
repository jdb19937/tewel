CXX = g++
NVCC = nvcc
NVCCFLAGS = -O3 -Xcompiler -fPIC
CXXFLAGS = -O3 -g -fPIC
LDFLAGS = -lm -lcudart -lSDL2

TGT = tewel tewel-nocuda tewel-nosdl tewel-nocuda-nosdl
OBJ = cortex.o tewel.o random.o youtil.o kleption.o cmdline.o camera.o video.o
HDR = colonel.hh cortex.hh random.hh youtil.hh kleption.hh cmdline.hh display.hh camera.hh video.hh

.PHONY: all
all: $(TGT)

.PHONY: clean
clean:
	rm -f $(OBJ) $(TGT) colonel-cuda.o colonel-nocuda.o

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

colonel-cuda.o: colonel.inc
colonel-nocuda.o: colonel.inc

tewel: $(OBJ) colonel-cuda.o display-sdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
tewel-nocuda: $(OBJ) colonel-nocuda.o display-sdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
tewel-nosdl: $(OBJ) colonel-cuda.o display-nosdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)
tewel-nocuda-nosdl: $(OBJ) colonel-nocuda.o display-nosdl.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

$(OBJ): $(HDR)
