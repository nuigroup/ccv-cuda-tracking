#include "cuda.h"
#include "assert.h"

float elapsedTime;

static int flag = 0;
static unsigned char *staticBg = (unsigned char *)malloc(320*240);

texture<unsigned char, 2, cudaReadModeElementType> texSrc;
texture<unsigned char, 2, cudaReadModeElementType> texConstant;

/////////////////////////////////////////// Subtract Image ///////////////////////////////////////////////////////
__global__ void subtract( unsigned char *gpu_in, unsigned char *staticBg, unsigned char *outTemp)
{
	int  		ix = threadIdx.x + __mul24( blockIdx.x, blockDim.x);
	int  		iy = threadIdx.y + __mul24( blockIdx.y, blockDim.y);
	//float		 x = (float)ix + 0.5f; 
	//float 	 y = (float)iy + 0.5f;
	int 	offset = ix + iy * blockDim.x * gridDim.x;

	if(ix >= 240 || iy >= 320)
        return;
        
	//outTemp[offset] = tex2D(texSrc, x, y) - staticBg[offset];
//	outTemp[offset] = gpu_in[offset] - staticBg[offset];
	outTemp[offset] = ( (gpu_in[offset] - staticBg[offset]) < 0 ? 0 : (gpu_in[offset] - staticBg[offset]) );

}

void gpu_subtract( unsigned char *frameIn)
{
	//cudaArray *src;
	//cudaChannelFormatDesc tex = cudaCreateChannelDesc<unsigned char>();
	//cudaMallocArray(&src,&tex,240,320);
	//cudaMemcpyToArray( src, 0, 0, in, 240*320, cudaMemcpyHostToDevice);
	//cudaBindTextureToArray(texSrc, src);

	
	unsigned char *in;
	in = (unsigned char *)malloc( 320 * 240 * sizeof(unsigned char));

	if(flag ==  0)
	{
	for( int i = 0; i < 320 * 240; i++)
	{
		staticBg[i] = frameIn[i];
	}
	}
	flag = 1;
	unsigned char *gpu_in;
	cudaMalloc( (void **)&gpu_in, 240 * 320);
	cudaMemcpy( gpu_in, frameIn, 240 * 320, cudaMemcpyHostToDevice);
	
	unsigned char *temp;
	cudaMalloc( (void **)&temp, 240 * 320);
	cudaMemcpy( temp, staticBg, 240 * 320, cudaMemcpyHostToDevice);

	unsigned char *outTemp;
	cudaMalloc( (void **)&outTemp, 240 * 320);
	
    //////////////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	dim3 threads( 16, 15);
	dim3 blocks( 15, 22);
	subtract<<< blocks, threads>>>( gpu_in, temp, outTemp);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/////////////////////////////////////////////////////////////////////////////
	
	cudaMemcpy( in, outTemp, 240 * 320, cudaMemcpyDeviceToHost);
	for( int i = 0; i < 320 * 240; i++)
	{
		frameIn[i] = in[i];
	}
	//cudaUnbindTexture(texSrc);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

