/*
// This source file contains the Cuda Code for subtraction of a source Image.
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

	/////////////////////////////////////////////////////////////////////////////
	
	if( cudaSuccess != cudaMemcpy( ctx->output_buffer_1, ctx->gpu_buffer_1, 240 * 320, cudaMemcpyDeviceToHost));
		error = GPU_ERR_MEM;;
	cudaFree(temp);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//FILE *file;
	//file = fopen("../timing.txt","a+");
	fprintf(stderr,"BgSubtract:%lf \n",elapsedtime);
	//fclose(file);
	
	return error;
}
