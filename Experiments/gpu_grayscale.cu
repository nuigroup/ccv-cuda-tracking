#include "cuda.h"

float elapsedtime;
texture<uchar4, 2, cudaReadModeElementType> texSrc;

////////////////////////////////////CUDA Programming///////////////////////////////////////////////////////////////

__global__ void convert(unsigned char *iin_1)
{

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = tx + ty * blockDim.x * gridDim.x;
    const float  x = (float)tx + 0.5f;
    const float  y = (float)ty + 0.5f;

	if(offset < 320*240)
        return;

	uchar4 temp;
	temp = tex2D(texSrc,x,y);
	float color = 0.3 * temp.x + 0.6 * temp.y + 0.1 * temp.z ;

	iin_1[offset] = color;
    
	/*
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;

	if(offset < 240*320)
	{	
		float color = 0.3 * (gpu_in[offset].x) + 0.6 * (gpu_in[offset].y) + 0.1 * (gpu_in[offset].z);
		gpu_in[offset].x = color;
		gpu_in[offset].y = color;
		gpu_in[offset].z = color;
		gpu_in[offset].w = 0;
	}*/
	
	/*__shared__ unsigned char sh_Tile[16*16*4];
	
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	int ty = threadIdx.y + (blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x*gridDim.x;
	int sh_offset = threadIdx.x + threadIdx.y * 16;

	sh_Tile[sh_offset * 4 + 0] = gpu_in[offset].x;
	sh_Tile[sh_offset * 4 + 1] = gpu_in[offset].y;
	sh_Tile[sh_offset * 4 + 2] = gpu_in[offset].z;
	sh_Tile[sh_offset * 4 + 3] = gpu_in[offset].w;
	
	__syncthreads();

	if(offset < 240*320)
	{	
		float color = 0.3 * (gpu_in[offset].x) + 0.6 * (gpu_in[offset].y) + 0.1 * (gpu_in[offset].z);
		gpu_in[offset].x = color;
		gpu_in[offset].y = color;
		gpu_in[offset].z = color;
		gpu_in[offset].w = 0;
	
		sh_Tile[sh_offset * 4 + 0] = (int) (0.3 * sh_Tile[sh_offset * 4 + 0] + 0.6 * sh_Tile[sh_offset * 4 + 1] + 0.1 * sh_Tile[sh_offset * 4 + 2]); 
	}
	__syncthreads();
	
	gpu_in[offset].x = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].y = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].z = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].w = 0;
	*/

	
	
}
///////////////// CUDA function call wrapper /////////////////
float tograyscale(unsigned char *in, unsigned char * in_1)
{
	//uchar4 *gpu_in;

	unsigned char *iin_1;
	cudaMalloc((void **)&iin_1, (240*320*sizeof(unsigned char)));
	
	cudaArray *src;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    cudaMallocArray(&src, &floatTex, 240, 320);
    cudaMemcpyToArray(src, 0, 0, in, 4 * 240 * 320, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(texSrc, src, floatTex);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	////////////////////////// Time consuming Task //////////////////////////////////	
	//cudaMalloc((void **)&gpu_in, (240*320*4*sizeof(unsigned char)));	
	//cudaMemcpy(gpu_in, in, (240*320*4*sizeof(unsigned char)), cudaMemcpyHostToDevice);

	dim3 grid(18,18);
	dim3 block(16,16);
	convert<<<grid,block>>>(iin_1);

	//cudaMemcpy( in, gpu_in, (240*320*4*sizeof(unsigned char)), cudaMemcpyDeviceToHost);
	/////////////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy( in_1, iin_1, (240*320*sizeof(unsigned char)), cudaMemcpyDeviceToHost);
	
	return elapsedtime;	
}


