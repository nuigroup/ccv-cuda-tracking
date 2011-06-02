#include "api.h"

#ifndef __GPU_GRAYSCALE_H
#define __GPU_GRAYSCALE_H

gpu_error_t gpu_grayscale(gpu_context *ctx);
__global__ void convert(int width, int height, unsigned char *gpu_in);

#endif
