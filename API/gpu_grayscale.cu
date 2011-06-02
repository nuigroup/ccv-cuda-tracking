#ifndef _CUDA_H
#define _CUDA_H
#include "cuda.h"
#endif

//#ifndef _API_H
//#define _API_H
#include "api.h"
//#endif

/////////////// Grayscale Cuda Fucntion ////////////////////
__global__ void convert(int width, int height, uchar4 *gpu_in)
{
	
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;

	if(offset < width * height)
	{	
		float color = 0.3 * (gpu_in[offset].x) + 0.6 * (gpu_in[offset].y) + 0.1 * (gpu_in[offset].z);
		gpu_in[offset].x = color;
		gpu_in[offset].y = color;
		gpu_in[offset].z = color;
		gpu_in[offset].w = 0;
	}	
	
}
///////////////// CUDA function call wrapper /////////////////
gpu_error gpu_grayscale(int width, int height, unsigned char *in)
{
	uchar4 *gpu_in;
	float elapsedtime;
	cudaEvent_t start, stop;
	gpu_error error = No_error;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	////////////////////////// Time consuming Task //////////////////////////////////	
	cudaMalloc((void **)&gpu_in, (width * height * 4 * sizeof(unsigned char)));
	cudaMemcpy(gpu_in, in, (width * height * 4 * sizeof(unsigned char)), cudaMemcpyHostToDevice);
	error = checkCudaError();

	dim3 grid(18,18);
	dim3 block(16,16);
	convert<<<grid,block>>>(width, height, gpu_in);

	error = checkCudaError();
	cudaMemcpy( in, gpu_in, (width * height * 4 * sizeof(unsigned char)), cudaMemcpyDeviceToHost);
	/////////////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaFree(gpu_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return error;
	
}

