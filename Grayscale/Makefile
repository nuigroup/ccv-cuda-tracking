.PHONY: grayscale

CUDA_LIBS=/usr/local/cuda/lib
LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
	CUDA_LIBS=/usr/local/cuda/lib64
endif

CUDA_FLAGS=-L$(CUDA_LIBS) -lcudart
OPENCV_FLAGS=$$(pkg-config --cflags --libs opencv)

grayscale:
	nvcc --cuda gpu_grayscale.cu
	g++ $(OPENCV_FLAGS) $(CUDA_FLAGS) -lcudart grayscale_test.c  gpu_grayscale.cu.cpp -o grayscale_test

all: grayscale

clean:
	-rm *.cu.cpp *.o *.swp
	-rm grayscale_test
