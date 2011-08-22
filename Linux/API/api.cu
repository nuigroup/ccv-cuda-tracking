/*
// This source file contains the API code various gpu methods provided by the library.
// It is a part of Cuda Image Processing Library ).
// Copyright (C) 2011 Remaldeep Singh

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
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

///////////////////////////////// Necessary Cuda calls /////////////////////////////////////////

/////////////// This call copies data from global memory ////////////////
void cuda_set_input(gpu_context_t *ctx, unsigned char *idata)
{
	int i = 0,
	size = ctx->width * ctx->height;

	switch ( ctx->nchannels )
	{
		case 1:
			for( ; i < size; i++)
			{
				ctx->output_buffer_1[i] = idata[i];
				// FIXME if the image is already gray (1 channel),
				// do we need to do that ? @ mathieu, No we dont need the rest of them.....
				//ctx->output_buffer_4[i * 4 + 1] = idata[i];
				//ctx->output_buffer_4[i * 4 + 2] = idata[i];
			}
			cudaMemcpy( ctx->gpu_buffer_1, ctx->output_buffer_1, size, cudaMemcpyHostToDevice);
			break;
		case 3:
			for( ; i < size; i++)
			{
				ctx->output_buffer_4[i * 4 + 0] = idata[i * 3 + 0];
				ctx->output_buffer_4[i * 4 + 1] = idata[i * 3 + 1];
				ctx->output_buffer_4[i * 4 + 2] = idata[i * 3 + 2];
			}
			cudaMemcpy(ctx->gpu_buffer_4, ctx->output_buffer_4, size * 4, cudaMemcpyHostToDevice);
			break;
		case 4:
			for( ; i < size; i++)
			{
				ctx->output_buffer_4[i * 4 + 0] = idata[i * 4 + 0];
				ctx->output_buffer_4[i * 4 + 1] = idata[i * 4 + 1];
				ctx->output_buffer_4[i * 4 + 2] = idata[i * 4 + 2];
				ctx->output_buffer_4[i * 4 + 3] = idata[i * 4 + 3];
			}
			cudaMemcpy(ctx->gpu_buffer_4, ctx->output_buffer_4, size * 4, cudaMemcpyHostToDevice);
			break;

		default:
			// this is because we don't know how to copy this input image to gpu buffer
			assert(0);
	}

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

	cudaMalloc( (void **)&ctx->gpu_buffer_4, ctx->size);
	last_error = checkCudaError();
	if(last_error == GPU_OK)
	{
		cudaMalloc( (void **)&ctx->gpu_buffer_1, (ctx->height * ctx->width * sizeof(unsigned char)) );
		last_error = checkCudaError();
	}
	if(last_error == GPU_OK)
	{
		switch (ctx->mem_flag)
		{
			case GPU_MEMORY_HOST:
				ctx->output_buffer_4 = (unsigned char *)malloc(ctx->size);
				if(!(ctx->output_buffer_4))
					last_error = GPU_ERR_MEM;
				ctx->output_buffer_1 = (unsigned char *)malloc((ctx->width * ctx->height * sizeof(unsigned char)));
				if(!(ctx->output_buffer_1))
					last_error = GPU_ERR_MEM;
				break;

			case GPU_MEMORY_PINNED_WRITE_COMBINED:
				cudaHostAlloc( (void **)&ctx->output_buffer_4, ctx->size, cudaHostAllocWriteCombined);
				last_error = checkCudaError();
	
				if(last_error == GPU_OK)
				{
					cudaHostAlloc( (void **)&ctx->output_buffer_1, (ctx->width * ctx->height * sizeof(unsigned char)), cudaHostAllocWriteCombined);
					last_error = checkCudaError();
				}
				break;

			case GPU_MEMORY_PINNED:
				cudaHostAlloc((void **)&ctx->output_buffer_4, ctx->size, cudaHostAllocDefault);
				last_error = checkCudaError();

				if(last_error == GPU_OK)
				{
					cudaHostAlloc( (void **)&ctx->output_buffer_1, (ctx->width * ctx->height * sizeof(unsigned char)), cudaHostAllocDefault);
					last_error = checkCudaError();
				}
				break;

			default:
				// should never happen
				assert(0);
				last_error = GPU_ERR_MEM;
				break;
		}
	}

	/*		Calculating appropriate no. of threaqds for current dimension		*/
	int   imageW = ctx->width;
	int   imageH = ctx->height;
	int    temp1 = imageW/4;
	int    temp2 = imageH/4;	
	int 	   i = 15;

	if( (imageW==480 || imageW==240 || imageW==320 ||  imageW==640) && (imageH==320 || imageH==640 || imageH==240 || imageH==480))
	{
		ctx->threadsX = 20;
		ctx->threadsY = 20;
	}
	else if( imageW==768 && imageH==1024)
	{
		ctx->threadsX = 16;
		ctx->threadsY = 16;
	}
	else
	{
		/*
		while( (temp1%i != 0) && (temp2%i != 0))
		{
			i++;
			if(i>20)break;
		}
		if( i>20 )
		{
			fprintf(stderr,"Invalid dimensions for blob detection");
			exit(EXIT_FAILURE);
		}
		ctx->threadsX = i;
		ctx->threadsY = i;			
		*/
		fprintf(stderr,"Invalid dimension");	
	}

	fprintf(stderr,"%d %d threads \n",ctx->threadsX,ctx->threadsY);
	

	return last_error;
}

///////////// This code will set the context buffer to the input buffer //////////////
gpu_error_t gpu_set_input( gpu_context_t *ctx, unsigned char *idata)
{
	assert( ctx || idata );
	cuda_set_input(ctx, idata);
	return GPU_OK;
}

///////////// This code will set the input buffer to the context buffer //////////////
gpu_error_t gpu_get_output(gpu_context_t *ctx, unsigned char **output)
{
	assert( ctx );
	assert( output != NULL );

	//cudaMemcpy(ctx->output_buffer_1, ctx->gpu_buffer_1, ctx->width * ctx->height , cudaMemcpyDeviceToHost);
	last_error = checkCudaError();
	if ( last_error == GPU_OK )
		*output = ctx->output_buffer_1;
	

	return last_error;
}

///// This code will deallocate all the memory held by context, including the memory on GPU //////
void gpu_context_free( gpu_context_t *ctx)
{
	assert(ctx);
	switch ( ctx->mem_flag )
	{
		case GPU_MEMORY_HOST:
			free(ctx->output_buffer_4);
			free(ctx->output_buffer_1);
			cudaFreeHost(ctx->gpu_buffer_4);
			cudaFreeHost(ctx->gpu_buffer_1);
			break;

		case GPU_MEMORY_PINNED_WRITE_COMBINED:
		case GPU_MEMORY_PINNED:
			cudaFreeHost(ctx->output_buffer_4);
			cudaFreeHost(ctx->gpu_buffer_4);
			cudaFreeHost(ctx->output_buffer_1);
			cudaFreeHost(ctx->gpu_buffer_1);
			break;

		default:
			// should never happen
			assert(0);
			break;
	}
	free(ctx);
}
