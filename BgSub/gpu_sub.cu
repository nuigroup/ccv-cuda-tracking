/*
	This programme contains only subtraction code for image. Background subtraction and 
	highpass use this function for implemetation.
*/

#include "cuda.h"
#include "../API/api.h"
#include "assert.h"

// gpu_in is 8bit input image. StaticBg is the background to be subtracted. outTemp is for output.
// It is the duty of the programmer to contantly change staticBg for dynamic filtering.
__global__ void subtract( unsigned char *gpu_buffer_1, unsigned char *staticBg)
{
	int  		ix = threadIdx.x + __mul24( blockIdx.x, blockDim.x);
	int  		iy = threadIdx.y + __mul24( blockIdx.y, blockDim.y);
	int 	offset = ix + iy * blockDim.x * gridDim.x;

	if(ix >= 240 || iy >= 320)
        return;
        
	gpu_buffer_1[offset] = ( (gpu_buffer_1[offset] - staticBg[offset]) < 0 ? 0 : (gpu_buffer_1[offset] - staticBg[offset]) );
}

gpu_error_t gpu_sub( gpu_context_t *ctx, unsigned char *staticBg)
{
	assert( staticBg);
	
	gpu_error_t error = GPU_OK;
	
	unsigned char *temp;
	cudaMalloc( (void **)&temp, ctx->width * ctx->height);
	cudaMemcpy( temp, staticBg, ctx->width * ctx->height, cudaMemcpyHostToDevice);
	error = CHECK_CUDA_ERROR();
		
    //////////////////////////////////////////////////////////////////////////////

	dim3 threads( 16, 12);
	dim3 blocks( 15, 27);
	subtract<<< blocks, threads>>>( ctx->gpu_buffer_1, temp);

	/////////////////////////////////////////////////////////////////////////////
	
	cudaMemcpy( ctx->output_buffer_1, ctx->gpu_buffer_1, 240 * 320, cudaMemcpyDeviceToHost);
	error = CHECK_CUDA_ERROR();
	return error;
}
