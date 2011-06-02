/*
	The idea behind this API is that, the application first will have to create a context of GPU which also includes a pointer	
	to the buffer that will contain a copy the frame of camera. This buffer can be pinned or pageable depending upon the 
	arguments passed to "host_flag". Gpu will create its own buffer and copy the contents of context buffer to it. Gpu context 
	buffer resides on host memory.

	IPLimage(cannot be made pinned memory) -----> Gpu_context buffer(can be made pinned) ----> Grayscale filter buffer(cudaMalloc).
									|		
									|	
									|------------------------> Threshold filter buffer(cudaMalloc).
	
	The scenerio being used right now (the one we dicussed via google doc as shown above) is not using pinned memory.
*/

#include "assert.h"
#ifndef _CUDA_H
#define _CUDA_H
#include "cuda.h"
#endif
#include "stdio.h"
#include "api.h"


//////////////////// Necessary Cuda calls ///////////////////////////////
/////////////// This call copies data from global memory ///////////////
__global__ void cuda_set_input(gpu_context *ctx, unsigned char *idata)
{
	for(int i=0;i < (ctx->width) * (ctx->height);i++)	
	{	
		ctx->gpu_buffer[i * 4 + 0] = idata[i * 3 + 0];
		ctx->gpu_buffer[i * 4 + 1] = idata[i * 3 + 1];
		ctx->gpu_buffer[i * 4 + 2] = idata[i * 3 + 2];
	}
}

/////////////// This call copies data to pinned memory /////////////////
__global__ void cuda_get_output(gpu_context *ctx)
{
	for(int i=0;i < (ctx->width) * (ctx->height);i++)	
	{	
		ctx->output_buffer[i * 3 + 0] = ctx->gpu_buffer[i * 4 + 0];
		ctx->output_buffer[i * 3 + 1] = ctx->gpu_buffer[i * 4 + 1];
		ctx->output_buffer[i * 3 + 2] = ctx->gpu_buffer[i * 4 + 2];
	}

}
/////////////////////////////////////////////////////////////////////////

////// This code will return error occured on GPU in a string format ////
const char * gpu_error()
{		
	cudaError_t err	= cudaGetLastError();
	const char * String = cudaGetErrorString(err);	
	return String;		
}

//////////// This code will check for any Cuda related error ////////////
gpu_error_t checkCudaError()
{
	cudaError_t err = cudaGetLastError();
	gpu_error_t error = No_error;
	if(err != cudaSuccess)
		error = Cuda_error;
	return error;
}

//////////// This code will create a gpu context /////////////
gpu_error_t gpu_context_create( gpu_context **ctx )
{
	gpu_error_t error = No_error;
		
	*ctx = (gpu_context *)malloc( sizeof(gpu_context) );
	if(!(*ctx))
		error = Memory_Allocation_error;
	return error;

}

/////////////////////////////// This code will initialize the previously created contex ///////////////////////////////////////////////////
gpu_error_t gpu_context_init( gpu_context *ctx, int host_height, int host_width, int host_nchannels, int host_flag)
{
	assert( host_height || host_width || (host_nchannels == 3));	

	gpu_error_t error = No_error;
	ctx->height = host_height;
	ctx->width = host_width;
	ctx->nchannels = host_nchannels;
	ctx->size = ctx->height * ctx->width * ctx->nchannels * sizeof(unsigned char);
	ctx->mem_flag = host_flag;
	
	cudaMalloc( (void **)ctx->gpu_buffer, ctx->size );
	error = checkCudaError();	
	if(error == No_error)
	{
		if(0 == ctx->mem_flag)
		{
			ctx->output_buffer = (unsigned char *)malloc(ctx->size);
			if(!(ctx->output_buffer))
				error = Memory_Allocation_error;							
		}
		else if(1 == ctx->mem_flag)
		{
			cudaHostAlloc( (void **)&(ctx->output_buffer), ctx->size, cudaHostAllocWriteCombined);
			error = checkCudaError();		
		}
		else if(2 == ctx->mem_flag)
		{	
			cudaHostAlloc((void **)&(ctx->output_buffer), ctx->size, cudaHostAllocDefault);
			error = checkCudaError();		
		}
		else
		{error =  Memory_Allocation_error;}
	}
	return error;
}

///////////// This code will set the context buffer to the input buffer //////////////
gpu_error_t gpu_set_input( gpu_context *ctx, unsigned char *idata)
{
	assert( ctx || idata );
	
	cuda_set_input( ctx, idata);
	/*for(int i=0;i < (ctx->width)*(ctx->height);i++)
	{
		ctx->output_buffer[i * 4 + 0] = idata[i * 3 + 0];
		ctx->output_buffer[i * 4 + 1] = idata[i * 3 + 1];
		ctx->output_buffer[i * 4 + 2] = idata[i * 3 + 2];
	}*/
	
	
}

///////////// This code will set the input buffer to the context buffer //////////////
gpu_error_t gpu_get_output( gpu_context *ctx)
{
	assert( ctx );

	cuda_get_output(ctx);
	/*for(int i=0; i < (ctx->width)*(ctx->height) ; i++)
	{
		odata[i * 3 + 0] = ctx->output_buffer[i * 4 + 0];
		odata[i * 3 + 1] = ctx->output_buffer[i * 4 + 1];
		odata[i * 3 + 2] = ctx->output_buffer[i * 4 + 2];
	}*/
	
	
}

///// This code will deallocate all the memory held by context, including the memory on GPU //////
void gpu_context_free( gpu_context *ctx)
{
	assert(ctx);
	if(0 == ctx->mem_flag)
	{free(ctx->output_buffer); cudaFreeHost(ctx->gpu_buffer); free(ctx);}
	else
	{cudaFreeHost(ctx->output_buffer); cudaFreeHost(ctx->gpu_buffer); free(ctx);}
}
