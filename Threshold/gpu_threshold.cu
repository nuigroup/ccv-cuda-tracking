/*
	I have just made this lib. It is not being used anywhere in the programme yet......
*/

#include "cuda.h"
#include "../api.h"

///////////////////////// Threshold Cuda function ////////////////////////
__global__ void convert( int width, int height, int th_value, unsigned char *gpu_in)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;

	if( offset < width * height )
	{
		if( (gpu_in[offset]) < th_value )
			gpu_in[offset] = 0;
		else
			gpu_in[offset] = 255;
	}

}

/////////////////////////// Cuda Function Call wrapper ///////////////////////////////
gpu_error_t gpu_threshold( gpu_context_t *ctx, int th_value)
{
	//float elapsedtime;
	//cudaEvent_t start, stop;
	gpu_error_t error = GPU_OK;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start,0);

	////////////////////////// Time consuming Task //////////////////////////////////	

	dim3 grid(18,18);
	dim3 block(16,16);
	convert<<<grid,block>>>( ctx->width, ctx->height, th_value, ctx->gpu_buffer_1);

	/////////////////////////////////////////////////////////////////////////////////

	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedtime,start,stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	return error;
}

