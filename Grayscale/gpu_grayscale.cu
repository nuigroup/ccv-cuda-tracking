
#include "cuda.h"
//#ifndef _API_H
//#define _API_H
#include "../api.h"
//#endif

/////////////// Grayscale Cuda Fucntion ////////////////////
__global__ void convert(int width, int height, unsigned char *gpu_in_1, unsigned char *gpu_in_4)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;
	int th_value = 20;

	if(offset < width * height)
	{
		float color = 0.3 * (gpu_in_4[offset * 4 + 0]) + 0.6 * (gpu_in_4[offset * 4 + 1]) + 0.1 * (gpu_in_4[offset * 4 + 2]);
		gpu_in_4[offset * 4 + 0] = color;
		gpu_in_4[offset * 4 + 1] = color;
		gpu_in_4[offset * 4 + 2] = color;
		gpu_in_4[offset * 4 + 3] = 0;
		//buffer[offset] = color;			// Dont know if it will work ---> It cant be done, It doesnt work
		if(color < th_value)		
			gpu_in_1[offset] = 0;			// There is really no need to call this function again for threshold when
		else						// all we have to do is copy from ouput->buffer to gpu->buffer and threshold.
			gpu_in_1[offset] = 255;			// I will be calcilating threshold here only and strothe result in gpu_in_1.

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
	convert<<<grid,block>>>( ctx->width, ctx->height, ctx->gpu_buffer_1, ctx->gpu_buffer_4);

	cudaMemcpy( ctx->output_buffer_1, ctx->gpu_buffer_1, (ctx->width * ctx->height * sizeof(unsigned char)), cudaMemcpyDeviceToHost);

	/////////////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return error;
}

