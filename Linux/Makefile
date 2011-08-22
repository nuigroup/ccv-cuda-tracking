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
	$(NVCC) --cuda -g -G -arch sm_21 Grayscale/gpu_grayscale.cu Threshold/gpu_threshold.cu GaussBlurTex/gpu_blur_tex.cu API/api.cu BgSub/gpu_sub.cu Amplify/gpu_amplify.cu Blobs/gpu_blob.cu
	$(CXX) $(CXX_FLAGS) $(OPENCV_FLAGS) $(CUDA_FLAGS) -lcudart main.c api.cu.cpp  gpu_grayscale.cu.cpp gpu_threshold.cu.cpp gpu_blur_tex.cu.cpp gpu_sub.cu.cpp gpu_amplify.cu.cpp gpu_blob.cu.cpp -o run

all: grayscale

clean:
	-rm *.cu.cpp *.o *.swp
	-rm grayscale_test
