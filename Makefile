CXX = g++
NVCC = nvcc
NVCCFLAGS = -O3 -Xcompiler -fPIC
CXXFLAGS = -O3 -g -fPIC
LDFLAGS = -lm -lcudart

TGT = tewel
OBJ = colonel.o cortex.o tewel.o random.o youtil.o kleption.o
HDR = colonel.hh cortex.hh random.hh youtil.hh kleption.hh

.PHONY: all
all: $(TGT)

.PHONY: clean
clean:
	rm -f $(OBJ) $(TGT)

%.o: %.cu
	$(NVCC) -o $@ $(NVCCFLAGS) -c $<

%.o: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -c $<

$(TGT): $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

$(OBJ): $(HDR)
