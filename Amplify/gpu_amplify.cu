#include "cuda.h"
#include "../API/api.h"
#include "assert.h"

__global__ void amplify( unsigned char *in, float ampValue, int size)
{
	int 	 x = threadIdx.x + __mul24(blockIdx.x,blockDim.x);
	int      y = threadIdx.y + __mul24(blockIdx.y,blockDim.y);
	int offset = x + y * __mul24(blockDim.x, gridDim.x);
	
	float temp;

	if( offset < size )
	{
		temp = (float)in[offset];
		in[offset] = ((temp * (float)ampValue) > 255) ? 255 : (unsigned char)(temp * (float)ampValue);
	}
}


gpu_error_t gpu_amplify(gpu_context_t *ctx, float ampValue)
{
	assert(ampValue);
	assert(ctx);

	gpu_error_t error = GPU_OK;

	///////////////////////////////// CUDA Call //////////////////////////////////////

	int temp1 = ((ctx->width % 16) != 0 ? (ctx->width / 16) + 1 : ctx->width / 16 );
	int temp2 = ((ctx->height % 12) != 0 ? (ctx->height / 12) + 1 : ctx->height / 12 );

	dim3 threads(16,12);
	dim3 blocks(temp1,temp2);	

	amplify<<<blocks,threads>>>( ctx->gpu_buffer_1, ampValue, (ctx->width * ctx->height));
	
	//////////////////////////////////////////////////////////////////////////////////

	error = CHECK_CUDA_ERROR();
	if( error == GPU_OK )
		cudaMemcpy( ctx->output_buffer_1, ctx->gpu_buffer_1, ctx->width * ctx->height, cudaMemcpyDeviceToHost);

	error = CHECK_CUDA_ERROR();
	return error;
}
