/*
// This source file contains the Cuda Code for grayscale of a source Image.
// It is a part of Cuda Image Processing Library .
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
#include "stdio.h"
#include "cuda_runtime.h"
#include "../API/api.h"

/////////////// Grayscale Cuda Fucntion ////////////////////
__global__ void grayscaleKernel(int width, int height, unsigned char *gpu_in_1, unsigned char *gpu_in_4)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x * gridDim.x;
	//int th_value = 40;

	if(offset < width * height)
	{
		float color = 0.3 * (gpu_in_4[offset * 4 + 0]) + 0.6 * (gpu_in_4[offset * 4 + 1]) + 0.1 * (gpu_in_4[offset * 4 + 2]);
		gpu_in_1[offset] = (unsigned char)color;
	}

}

///////////////// CUDA function call wrapper /////////////////
gpu_error_t gpu_grayscale(gpu_context_t *ctx)
{
	assert(ctx);
	
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

	////////////////////////// Kernel Call //////////////////////////////////	
	dim3 block(threadsX,threadsY);
	dim3 grid(temp1,temp2);
	grayscaleKernel<<<grid,block>>>( ctx->width, ctx->height, ctx->gpu_buffer_1, ctx->gpu_buffer_4);
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
	fprintf(stderr,"\nGrayscale:%lf \n",elapsedtime);
	//fclose(file);
	
	return error;
}

