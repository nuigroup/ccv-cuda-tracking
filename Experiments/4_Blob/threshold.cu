#include "cuda.h"

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

void gpu_threshold( unsigned char *in)
{
	int th_value = 20;

	unsigned char *gpu_in;
	cudaMalloc( (void **)&gpu_in, 240 * 320);
	cudaMemcpy( gpu_in, in, 240 * 320, cudaMemcpyHostToDevice);
	
	dim3 blocks(16,20);
	dim3 threads(15,16);

	convert<<<blocks,threads>>>( 240, 320, th_value, gpu_in);

	cudaMemcpy( in, gpu_in, 240 * 320, cudaMemcpyDeviceToHost);

}
