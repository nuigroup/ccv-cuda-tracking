#include "cuda.h"
#include "../API/api.h"

///////////////////////// Threshold Cuda function ////////////////////////
__global__ void convert( int width, int height, int th_value, unsigned char *gpu_in)
{
	int 	tx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int 	ty = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	int offset = tx + ty * __mul24(blockDim.x, gridDim.x);

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
	gpu_error_t error = GPU_OK;
	
	////////////////////////// Time consuming Task //////////////////////////////////	

	//dim3 blocks(15,22);
	//dim3 threads(16,15);

	dim3 blocks(16,20);
	dim3 threads(15,16);
	
	convert<<<blocks,threads>>>( ctx->width, ctx->height, th_value, ctx->gpu_buffer_1);

	/////////////////////////////////////////////////////////////////////////////////

	cudaMemcpy(ctx->output_buffer_1, ctx->gpu_buffer_1, ctx->width * ctx->height , cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR();

	return error;
}

