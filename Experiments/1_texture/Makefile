.PHONY: grayscale

# 64 bits platform ?
LBITS := $(shell getconf LONG_BIT)

# Cuda libs
CUDA_PATH := /usr/local/cuda
CUDA_LIBS := $(CUDA_PATH)/lib
NVCC := $(CUDA_PATH)/bin/nvcc
ifeq ($(LBITS),64)
	CUDA_LIBS := /usr/local/cuda/lib64
endif
CUDA_FLAGS := -L$(CUDA_LIBS) -lcudart

# OpenCV
OPENCV_FLAGS := $$(pkg-config --cflags --libs opencv)

# CXX Flags
CXX_FLAGS := -ggdb -Wall

grayscale:
	$(NVCC) --cuda -g -G gpu_grayscale.cu
	$(CXX) $(CXX_FLAGS) $(OPENCV_FLAGS) $(CUDA_FLAGS) -lcudart gpu_grayscale.cu.cpp main.cpp -o grayscale_test

all: grayscale

clean:
	-rm *.cu.cpp *.o *.swp
	-rm grayscale_test
