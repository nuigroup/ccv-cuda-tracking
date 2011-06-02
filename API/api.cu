/*
	The idea behind this API is that, the application first will have to create a context of GPU which also includes a pointer	
	to the buffer that will contain a copy the frame of camera. This buffer can be pinned or pageable depending upon the 
	arguments passed to "host_flag". Gpu will create its own buffer and copy the contents of context buffer to it. Gpu context 
	buffer resides on host memory.

	IPLimage(cannot be made pinned memory) -----> gpu_context_t buffer(can be made pinned) ----> Grayscale filter buffer(cudaMalloc).
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

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "api.h"

gpu_error_t last_error = GPU_OK;
cudaError_t last_cuda_error = cudaSuccess;

//////////////////// Necessary Cuda calls ///////////////////////////////
/////////////// This call copies data from global memory ///////////////
void cuda_set_input(gpu_context_t *ctx, unsigned char *idata)
{
	int i = 0,
		size = ctx->width * ctx->height;

	switch ( ctx->nchannels )
	{
		case 1:
			for( ; i < size; i++)
			{
				ctx->output_buffer[i * 4 + 0] = idata[i];
				// FIXME if the image is already gray (1 channel),
				// do we need to do that ?
				ctx->output_buffer[i * 4 + 1] = idata[i];
				ctx->output_buffer[i * 4 + 2] = idata[i];
			}

		case 3:
			for( ; i < size; i++)
			{
				ctx->output_buffer[i * 4 + 0] = idata[i * 3 + 0];
				ctx->output_buffer[i * 4 + 1] = idata[i * 3 + 1];
				ctx->output_buffer[i * 4 + 2] = idata[i * 3 + 2];
			}
			break;

		default:
			// this is because we don't know how to copy this input image to gpu buffer
			assert(0);
	}

	cudaMemcpy(ctx->output_buffer, ctx->gpu_buffer, size * 4, cudaMemcpyHostToDevice);
	checkCudaError();
}

/////////////////////////////////////////////////////////////////////////

////// This code will return error occured on GPU in a string format ////
const char *gpu_error()
{
	// reset the error for next call
	gpu_error_t error = last_error;
	cudaError_t cuda_error = last_cuda_error;
	last_error = GPU_OK;
	last_cuda_error = cudaSuccess;

	switch (error)
	{
		case GPU_OK:
			return "OK";
		case GPU_ERR_MEM:
			return "Memory allocation";
		case GPU_ERR_CUDA:
			return cudaGetErrorString(cuda_error);
	}

	return "Unknown";
}

//////////// This code will check for any Cuda related error ////////////
gpu_error_t checkCudaError()
{
	last_cuda_error = cudaGetLastError();
	if (last_cuda_error != cudaSuccess)
		return GPU_ERR_CUDA;
	return GPU_OK;
}

//////////// This code will create a gpu context /////////////
gpu_error_t gpu_context_create(gpu_context_t **ctx)
{
	last_error = GPU_OK;
	assert(ctx != NULL);

	// create the context and initialize it
	*ctx = (gpu_context_t *)malloc( sizeof(gpu_context_t) );
	if (*ctx == NULL)
		last_error = GPU_ERR_MEM;
	else
		memset(*ctx, 0, sizeof(gpu_context_t));

	return last_error;

}

/////////////////////////////// This code will initialize the previously created contex ///////////////////////////////////////////////////
gpu_error_t gpu_context_init(gpu_context_t *ctx, int host_height, int host_width, int host_nchannels, gpu_context_memory_t host_flag)
{
	assert(ctx != NULL);
	assert(host_height > 0);
	assert(host_width > 0);
	assert(host_nchannels == 3);

	ctx->height = host_height;
	ctx->width = host_width;
	ctx->nchannels = host_nchannels;
	ctx->mem_flag = host_flag;
	// whatever the source channels is, we always use 4 channels images
	ctx->size = ctx->height * ctx->width * 4 * sizeof(unsigned char);

	cudaMalloc( (void **)&ctx->gpu_buffer, ctx->size);
	last_error = checkCudaError();
	if(last_error == GPU_OK)
	{
		switch (ctx->mem_flag)
		{
			case GPU_MEMORY_HOST:
				ctx->output_buffer = (unsigned char *)malloc(ctx->size);
				if(!(ctx->output_buffer))
					last_error = GPU_ERR_MEM;
				break;

			case GPU_MEMORY_PINNED_WRITE_COMBINED:
				cudaHostAlloc( (void **)&ctx->output_buffer, ctx->size, cudaHostAllocWriteCombined);
				last_error = checkCudaError();
				break;

			case GPU_MEMORY_PINNED:
				cudaHostAlloc((void **)&ctx->output_buffer, ctx->size, cudaHostAllocDefault);
				last_error = checkCudaError();
				break;

			default:
				// should never happen
				assert(0);
				last_error = GPU_ERR_MEM;
				break;
		}
	}

	return last_error;
}

///////////// This code will set the context buffer to the input buffer //////////////
gpu_error_t gpu_set_input( gpu_context_t *ctx, unsigned char *idata)
{
	assert( ctx || idata );

	cuda_set_input(ctx, idata);
	/*for(int i=0;i < (ctx->width)*(ctx->height);i++)
	{
		ctx->output_buffer[i * 4 + 0] = idata[i * 3 + 0];
		ctx->output_buffer[i * 4 + 1] = idata[i * 3 + 1];
		ctx->output_buffer[i * 4 + 2] = idata[i * 3 + 2];
	}*/

	return GPU_OK;
}

///////////// This code will set the input buffer to the context buffer //////////////
gpu_error_t gpu_get_output(gpu_context_t *ctx, unsigned char **output)
{
	assert( ctx );
	assert( output != NULL );

	// copy back the gpu buffer to host buffer
	cudaMemcpy(ctx->output_buffer, ctx->gpu_buffer, ctx->width * ctx->height * 4, cudaMemcpyDeviceToHost);
	last_error = checkCudaError();
	if ( last_error == GPU_OK )
		*output = ctx->output_buffer;

	return last_error;
}

///// This code will deallocate all the memory held by context, including the memory on GPU //////
void gpu_context_free( gpu_context_t *ctx)
{
	assert(ctx);
	switch ( ctx->mem_flag )
	{
		case GPU_MEMORY_HOST:
			free(ctx->output_buffer);
			cudaFreeHost(ctx->gpu_buffer);
			break;

		case GPU_MEMORY_PINNED_WRITE_COMBINED:
		case GPU_MEMORY_PINNED:
			cudaFreeHost(ctx->output_buffer);
			cudaFreeHost(ctx->gpu_buffer);
			break;

		default:
			// should never happen
			assert(0);
			break;
	}
	free(ctx);
}
