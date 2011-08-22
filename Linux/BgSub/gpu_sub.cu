/*
	This programme contains only subtraction code for image. Background subtraction and 
	highpass use this function for implemetation.
*/

#include "cuda.h"
#include "../API/api.h"
#include "assert.h"
#include "stdio.h"

// gpu_in is 8bit input image. StaticBg is the background to be subtracted. outTemp is for output.
// It is the duty of the programmer to contantly change staticBg for dynamic filtering.
__global__ void subtract( unsigned char *gpu_buffer_1, unsigned char *staticBg, int imageW, int imageH)
{
	int  		ix = threadIdx.x + __mul24( blockIdx.x, blockDim.x);
	int  		iy = threadIdx.y + __mul24( blockIdx.y, blockDim.y);
	int 	offset = ix + iy * __mul24( blockDim.x, gridDim.x);

	if(ix >= imageW || iy >= imageH)
        return;
        
	gpu_buffer_1[offset] = ( (gpu_buffer_1[offset] - staticBg[offset]) < 0 ? 0 : (gpu_buffer_1[offset] - staticBg[offset]) );
}

gpu_error_t gpu_sub( gpu_context_t *ctx, unsigned char *staticBg)
{
	assert(staticBg);
	assert(ctx);

	float elapsedtime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	gpu_error_t error = GPU_OK;
	unsigned char *temp;
	cudaMalloc( (void **)&temp, ctx->width * ctx->height);
	cudaMemcpy( temp, staticBg, ctx->width * ctx->height, cudaMemcpyHostToDevice);
	error = CHECK_CUDA_ERROR();

	int threadsX = ctx->threadsX;
	int threadsY = ctx->threadsY;
	int temp1 = ((ctx->width % threadsX) != 0 ? (ctx->width / threadsX) + 1 : ctx->width / threadsX );
	int temp2 = ((ctx->height % threadsY) != 0 ? (ctx->height / threadsY) + 1 : ctx->height / threadsY );
	
    //////////////////////////////////////////////////////////////////////////////

	dim3 threads( threadsX, threadsY);
	dim3 blocks( temp1, temp2);
	subtract<<< blocks, threads>>>( ctx->gpu_buffer_1, temp, ctx->width, ctx->height);
	cudaThreadSynchronize();
	/////////////////////////////////////////////////////////////////////////////
	
	/***************** There is somthing wrong here *******************************/
	
	if( cudaSuccess != cudaMemcpy(ctx->output_buffer_1, ctx->gpu_buffer_1, ctx->width * ctx->height , cudaMemcpyDeviceToHost));
	{
	//	fprintf(stderr,"mem_cpy_error\n");
		error = GPU_ERR_MEM;
	}
	
	error = checkCudaError();	
	if(error != GPU_OK)
	{	fprintf(stderr,"mem_cpy_error\n");
	}
	/*******************************************************************************/
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(temp);
	//FILE *file;
	//file = fopen("../timing.txt","a+");
	fprintf(stderr,"BgSubtract:%lf \n",elapsedtime);
	//fclose(file);
	
	return error;
}
