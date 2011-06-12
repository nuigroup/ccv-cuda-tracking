/*
	The code here isn't in its working stage. Moreover I am limited with the size of image
	that I can work with, if I use the following technique. Need to work around the size of kernel too.
	FIXME: Refine this appreach before using it.
*/
#include "assert.h"
#include "cuda.h"
#include "api.h"

#define KERNEL_RADIUS 8
#define KERNEL_WIDTH ( 2 * KERNEL_RADIUS + 1 )  // +1 is there because I need to keep in mind the current pixel being processed upon.
__device__ __constant__ float kernel[KERNEL_WIDTH];

#define ROW_TILE_WIDTH 128  
// Just for referenece ROW_TILE_HEIGHT = 1
#define COLUMN_TILE_WIDTH 16
#define COLUMN_TILE_HEIGHT 48
#define KERNEL_RADIUS_ALIGNED 16

#define MUL(a, b) __mul24(a, b)	// It provides additional speedup for multiplications.

/////////////////////////////////// Loop unroll templates ///////////////////////////////////////
/////////////////////////// try and use #pragma unroll instead //////////////////////////////////

template<int i> __device__ float unrollRow(float *data)
{    
	return  data[KERNEL_RADIUS - i] * kernel[i] + unrollRow<i - 1>(data);
}

template<> __device__ float unrollRow<-1>(float *data)
{
    return 0;
}

template<int i> __device__ float unrollColumn(float *data)
{	
	return data[(KERNEL_RADIUS - i) * COLUMN_TILE_WIDTH] * kernel[i] + unrollColumn<i - 1>(data);
}

template<> __device__ float unrollColumn<-1>(float *data)
{
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////// Row convultion /////////////////////////////////////////////
__global__ void row_convolution( gpu_context *ctx, (float *)input, (float *)result )
{
	__shared__ float smem[KERNEL_RADIUS + ROW_TILE_WIDTH + KERNEL_RADIUS];

	// Apron are the extra pixels that are needed to calculate pixel values for input pixels near the border.
	int 	    	 tile_start = MUL( blockIdx.x, ROW_TILE_WIDTH);
	int 		       tile_end = tile_start + ROW_TILE_WIDTH - 1;
	int 		    apron_start	= tile_start - KERNEL_RADIUS;
	int 	apron_start_aligned = tile_start - KERNEL_RADIUS_ALIGNED;
	int 	          apron_end = tile_end + KERNEL_RADIUS;

	//There may be cases when the apron goes beyond the image borders i.e when consider pixels at the borders.
	//In those cases just clamp the pixels to the pixels at the border.
	int 	 tile_end_clamped = max( tile_end, ctx->width - 1);	
	int   apron_start_clamped = min( apron_start, 0);
	int 	apron_end_clamped = max( apron_end, ctx->height - 1);

	// Calculating the x and y offset of the current block
	int		 yo = MUL( blockIdx.y, ctx->width); // FIXME: Verify this width paramemter. It can be faulty.
	int x_input = apron_start_aligned + threadIdx.x;	// x offst in the input buffer

	// We need to have inactive threads at the start
	// (they are made just to align the kernel radius to half warp)
	if( x_input >= apron_start )
	{
		int x_shared = x_input - apron_start; // The position inside the shared memory. This will eventually lead to x_shared = x - 8. Which is observable from the fact 
								  // that shared memory doesnt insclude kernel_radius_aligned.

		smem[x_shared] = ((x_input >= apron_start_clamped) && (x_input <= apron_end_clamped)) ? input[x_input + yo] : 0;
	}
	
	__syncthreads();

	//x offset in the result buffer
	int x_result = tile_start + threadIdx.x;	

	if( x_result < tile_end_clamped)
	{
		int x_shared2 = x_result - apron_start;
		float sum = 0;

#ifdef UNROLL_INNER
	sum = unrollRow<2 * KERNEL_RADIUS>(smem + x_shared2);
#else
	for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS ; k++)	
		sum += smem[x_shared2 + k] * kernel[KERNEL_RADIUS - k];
#endif

	result[yo + x_result] = sum;	// yo is the y offset to the input and result buffer.

	}

}
///////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////// Column Convolution /////////////////////////////////////////////
__global__ void column_convolution( gpu_context *ctx, (float *)input, (float *)result, int shared_stride, int global_stride)
{
	__shared__ float smem[COLUMN_TILE_WIDTH * (KERNEL_RADIUS + COLUMN_TILE_HEIGHT + KERNEL_RADIUS)];

	/////// Calculating the starting and ending indices ////////
	int		tile_start = MUL( blockIdx.y, COLUMN_TILE_HEIGHT);
	int       tile_end = tile_start + COLUMN_TILE_HEIGHT - 1;
	int    apron_start = tile_start - KERNEL_RADIUS;
	int      apron_end = tile_end + KERNEL_RADIUS;

	/////// Clamping the indices to image borders ////////
	int		tile_end_clamped = min( tile_end, ctx->height);
	int  apron_start_clamped = max( apron_start, 0);
	int    apron_end_clamped = min( apron_end, ctx->height - 1);

	int xpos = MUL( blockIdx.x, COLUMN_TILE_WIDTH) + threadIdx.x;

	//// Calculating the corresponding global and shared memory position /////
	int	shared_pos = MUL( threadIdx.y, COLUMN_TILE_WIDTH) + threadIdx.x;
	int global_pos = MUL( apron_start + threadIdx.y, ctx->width) + xpos;

	//// Filling of the shared memory ////
#pragma unroll
	for(int y = apron_start + threadIdx.y; y<= apron_end; y += blockDim.y)
	{
		smem[shared_pos] = ((y >= apron_start_clamped) && (y <= apron_end_clamped)) ? input[global_pos] : 0;
		shared_pos += shared_stride;
		global_pos += global_stride;
	}
	__syncthreads();

	/////// The global and shared positions excluding the apron pixels ///////
	shared_pos = MUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_WIDTH) + threadIdx.x;
	global_pos = MUL(threadIdx.y + tile_start, ctx->width) + xpos;
	
	for(int y = threadIdx.y; y <= tile_end_clamped ; y += blockDim.y )
	{
		float sum = 0;	

#ifdef UNROLL_INNER
		sum = unrollColumn<2 * KERNEL_RADIUS>(kernel + shared_pos);
#else
		for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS ; k++)
			sum += smem[ shared_pos + MUL( k, COLUMN_TILE_WIDTH) ] * kernel[KERNEL_RADIUS - k];
#endif

		result[global_pos] = sum;
		shared_pos += shared_stride;
		global_pos += global_stride;
	}

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////// Wrapper combinig the row and column convolution /////////////////////////////////
gpu_error_t gpu_blur(gpu_context *ctx)
{
	float *input;
	float *result;
	float *h_Kernel;

	h_Kernel = (float *)malloc((KERNEL_WIDTH * sizeof(float)));
	cudaMalloc( (void **)&input, (ctx->width * ctx->height * sizeof(float)));
	cudaMalloc( (void **)&result, (ctx->width * ctx->height * sizeof(float)));

	float kernelSum = 0;
    for(i = 0; i < KERNEL_WIDTH; i++)
    {
        float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
        h_Kernel[i] = expf(- dist * dist / 2);
        kernelSum += h_Kernel[i];
    }
    for(i = 0; i < KERNEL_WIDTH; i++)
        h_Kernel[i] /= kernelSum;

	cudaMemcpyToSymbol( kernel, h_Kernel, (KERNEL_WIDTH * sizeof(float)) );
	for(int i=0; i < ctx->width * ctx->height ;i++)
	{
		input[i] = (float)ctx->output_buffer_1[i];
	}

	int temp1 = (ctx->width % ROW_TILE_WIDTH != 0) ? (ctx->width / ROW_TILE_WIDTH + 1) : (ctx->width / ROW_TILE_WIDTH) ;
	int temp2 = (ctx->width % COLUMN_TILE_WIDTH != 0) ? (ctx->width / COLUMN_TILE_WIDTH + 1) : (ctx->width / COLUMN_TILE_WIDTH);
	int temp3 = (ctx->height % COLUMN_TILE_HEIGHT != 0) ? (ctx->height / COLUMN_TILE_HEIGHT + 1) : (ctx->height / COLUMN_TILE_HEIGHT);
	
	dim3 row_threads(KERNEL_RADIUS_ALIGNED + ROW_TILE_WIDTH + KERNEL_RADIUS); 
	dim3 row_blocks(temp1,ctx->height);  
	dim3 column_threads(COLUMN_TILE_WIDTH, 8); 
	dim3 column_blocks( temp2, temp3); 

	cudaThreadSynchronize();

	row_convolution<<<row_blocks,row_threads>>>( ctx, input, result);
	column_convolution<<<column_blocks,column_threads>>>(ctx, input, result, COLUMN_TILE_WIDTH * column_threads.y, ctx->width * column_threads.y);

	for(int i=0; i < ctx->width * ctx->height ;i++)
	{
		ctx->output_buffer_1[i] = (unsigned char)result[i];
	}

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
