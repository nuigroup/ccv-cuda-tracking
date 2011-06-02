/*
	The idea behind this API is that, the application first will have to create a context of GPU which also includes a pointer	
	to the buffer that will contain a copy the frame of camera. This buffer can be pinned or pageable depending upon the 
	arguments passed to "host_flag". Gpu will create its own buffer and copy the contents of context buffer to it. Gpu context 
	buffer resides on host memory.

	IPLimage(cannot be made pinned memory) -----> Gpu_context buffer(can be made pinned) ----> Grayscale filter buffer(cudaMalloc).
									|		
									|	
									|------------------------> Threshold filter buffer(cudaMalloc).

	or

	IPLimage(cannot be made pinned memory) -----> Respective filter buffer on GPU(cudaMalloc).

	Right now following the first scenerio.
*/

#ifndef _CUDA_H
#define _CUDA_H
#include "cuda.h"
#endif

#include "stdio.h"

//#ifndef _API_H
//#define _API_H
#include "api.h"
//#endif


//////////// This code will check for any Cuda related error ////////////
gpu_error checkCudaError()
{
	cudaError_t err = cudaGetLastError();
	gpu_error error = No_error;
	if(err != cudaSuccess)
		error = Cuda_error;
	return error;
}

//////////// This code will create a gpu context /////////////
gpu_error gpu_context_create( gpu_context *ctx )
{
	gpu_error error = No_error;
		
	ctx = (gpu_context *)malloc( sizeof(gpu_context) );
	if(!ctx)
		error = Memory_Allocation_error;
	return error;

}

/////////////////////////////// This code will initialize the previously created contex ///////////////////////////////////////////////////
gpu_error gpu_context_init( gpu_context *ctx, int host_height, int host_width, int host_nchannels, int host_flag)
{
	gpu_error error = No_error;
	ctx->height = host_height;
	ctx->width = host_width;
	ctx->nchannels = host_nchannels;
	ctx->size = ctx->height * ctx->width * ctx->nchannels * sizeof(unsigned char);
	ctx->mem_flag = host_flag;
	
	if(0 == ctx->mem_flag)
	{
		ctx->buffer = (unsigned char *)malloc(ctx->size);
		if(!(ctx->buffer))
			error = Memory_Allocation_error;							
	}
	else if(1 == ctx->mem_flag)
	{
		cudaHostAlloc( (void **)&(ctx->buffer), ctx->size, cudaHostAllocWriteCombined);
		error = checkCudaError();		
	}
	else if(2 == ctx->mem_flag)
	{	
		cudaHostAlloc((void **)&(ctx->buffer), ctx->size, cudaHostAllocDefault);
		error = checkCudaError();		
	}
	else
	{error =  Memory_Allocation_error;}
	return error;
}

///////////// This code will set the context buffer to the input buffer //////////////
gpu_error gpu_set_input( gpu_context *ctx, unsigned char *idata)
{
	for(int i=0;i < (ctx->width)*(ctx->height);i++)
	{
		ctx->buffer[i * 4 + 0] = idata[i * 3 + 0];
		ctx->buffer[i * 4 + 1] = idata[i * 3 + 1];
		ctx->buffer[i * 4 + 2] = idata[i * 3 + 2];
	}
}

///////////// This code will set the input buffer to the context buffer //////////////
gpu_error gpu_get_input( gpu_context *ctx, unsigned char *odata)
{
	for(int i=0; i < (ctx->width)*(ctx->height) ; i++)
	{
		odata[i * 3 + 0] = ctx->buffer[i * 4 + 0];
		odata[i * 3 + 1] = ctx->buffer[i * 4 + 1];
		odata[i * 3 + 2] = ctx->buffer[i * 4 + 2];
	}
}

///// This code will deallocate all the memory held by context, including the memory on GPU //////
void gpu_context_free( gpu_context *ctx)
{
	if(0 == ctx->mem_flag)
	{free(ctx->buffer); free(ctx);}
	else
	{cudaFreeHost(ctx->buffer);free(ctx);}
}
