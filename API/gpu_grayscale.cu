
#include "cuda.h"
//#ifndef _API_H
//#define _API_H
#include "api.h"
//#endif

/////////////// Grayscale Cuda Fucntion ////////////////////
__global__ void convert(int width, int height, unsigned char *gpu_in)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;

	if(offset < width * height)
	{
		float color = 0.3 * (gpu_in[offset * 4 + 0]) + 0.6 * (gpu_in[offset * 4 + 1]) + 0.1 * (gpu_in[offset * 4 + 2]);
		gpu_in[offset * 4 + 0] = color;
		gpu_in[offset * 4 + 1] = color;
		gpu_in[offset * 4 + 2] = color;
		gpu_in[offset * 4 + 3] = 0;
	}

}

///////////////// CUDA function call wrapper /////////////////
gpu_error_t gpu_grayscale(gpu_context_t *ctx)
{
	float elapsedtime;
	cudaEvent_t start, stop;
	gpu_error_t error = GPU_OK;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	////////////////////////// Time consuming Task //////////////////////////////////	

	dim3 grid(18,18);
	dim3 block(16,16);
	convert<<<grid,block>>>( ctx->width, ctx->height, ctx->gpu_buffer);

	/////////////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return error;
}

