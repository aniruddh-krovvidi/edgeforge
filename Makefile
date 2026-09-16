# edgeforge build. Linux/CI: `make`. macOS with a broken Xcode shim:
#   make CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
#        SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra
SYSROOT  ?=
ifneq ($(SYSROOT),)
CXXFLAGS += -isysroot $(SYSROOT)
endif
NVCC ?= nvcc

all: edgeforge

edgeforge: runtime.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# GPU parity test; needs an NVIDIA GPU + CUDA toolkit. Not built by default.
edgeforge_cuda: cuda/edgeforge_cuda.cu runtime.cpp
	$(NVCC) -std=c++17 -O3 -DEF_NO_MAIN -x cu -o $@ cuda/edgeforge_cuda.cu

test: edgeforge
	./edgeforge eval data/model_fp32.efm data/mnist_test.bin
	./edgeforge eval data/model_int8.efm data/mnist_test.bin
	.venv/bin/python test_parity.py || python3 test_parity.py

bench: edgeforge
	./edgeforge bench data/model_fp32.efm data/mnist_test.bin 5000
	./edgeforge bench data/model_int8.efm data/mnist_test.bin 5000

clean:
	rm -f edgeforge edgeforge_cuda

.PHONY: all test bench clean
