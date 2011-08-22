#include "cuda_runtime.h"
#include "../API/api.h"
#include "assert.h"
#include "stdio.h"

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
	assert(th_value);

	float elapsedtime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	gpu_error_t error = GPU_OK;
	int threadsX = ctx->threadsX;
	int threadsY = ctx->threadsY;
	int temp1 = ((ctx->width % threadsX) != 0 ? (ctx->width / threadsX) + 1 : ctx->width / threadsX );
	int temp2 = ((ctx->height % threadsY) != 0 ? (ctx->height / threadsY) + 1 : ctx->height / threadsY );

	///////////////////////////// Kernel Call ///////////////////////////////////////
	dim3 blocks(threadsX,threadsY);
	dim3 threads(temp1,temp2);	
	convert<<<blocks,threads>>>( ctx->width, ctx->height, th_value, ctx->gpu_buffer_1);
	/////////////////////////////////////////////////////////////////////////////////

	if( cudaSuccess != cudaMemcpy(ctx->output_buffer_1, ctx->gpu_buffer_1, ctx->width * ctx->height , cudaMemcpyDeviceToHost))
		error = GPU_ERR_MEM;

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//FILE *file;
	//file = fopen("../timing.txt","a+");
	fprintf(stderr,"Threshold:%lf \n",elapsedtime);
	//fclose(file);

	return error;
}

