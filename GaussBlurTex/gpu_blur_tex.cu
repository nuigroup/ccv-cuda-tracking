/*
	The current approach uses only texture memory. Performance is way better than global memory.
	Future release might contain share memory embedded in it. :D
*/
#include "../API/api.h"
#include "cuda.h"

#define MAD(a, b, c) ( __mul24((a), (b)) + (c) )	//Responsible for speeding up the process of multiplicaton
//#define UNROLL_INNER 1 // Using unroll to speed up looping

/////////// Round a / b to nearest higher integer value /////////////
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__constant__ float Kernel[33]; // Using constant memory due to high bandwidth availabilitity

texture<unsigned char, 2, cudaReadModeElementType> texSrc; // Declaring texture memory

///////////////////////////// Loop unrolling templates, needed for best performance /////////////////////////////
////////////////////////////////////// Try and use #pragma unroll instead ///////////////////////////////////////
/*
template<int i> __device__ float convolutionRow(float x, float y){
    return tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * Kernel[i] + convolutionRow<i - 1>(x, y);
}

template<> __device__ float convolutionRow<-1>(float x, float y){
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y){
    return tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * Kernel[i] + convolutionColumn<i - 1>(x, y);
}

template<> __device__ float convolutionColumn<-1>(float x, float y){
    return 0;
}
*/

//////////////////////////////////////////////// Row convolution filter ///////////////////////////////////////////////////
__global__ void convolutionRowsKernel( unsigned char *dst, int imageW, int imageH, int KERNEL_RADIUS, int KERNEL_LENGTH)
{
    const   int ix = MAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = MAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix >= imageW || iy >= imageH)
        return;

    float sum = 0;

   /* #if(UNROLL_INNER)
        sum = convolutionRow<2 * KERNEL_RADIUS>(x, y);
    #else*/
    #pragma unroll
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
        sum += tex2D(texSrc, x + (float)k, y) * Kernel[KERNEL_RADIUS - k];
  //  #endif

    dst[MAD(iy, imageW, ix)] = (unsigned char)sum;
}


void convolutionRowsGPU( unsigned char *dst, cudaArray *src, int imageW, int imageH, int KERNEL_RADIUS, int KERNEL_LENGTH)
{
	dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    cudaBindTextureToArray(texSrc, src);
    convolutionRowsKernel<<<blocks, threads>>>(
        dst,
        imageW,
        imageH,
        KERNEL_RADIUS,
        KERNEL_LENGTH
    );
    cudaUnbindTexture(texSrc);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////// Column convolution filter ////////////////////////////////////////////////
__global__ void convolutionColumnsKernel( unsigned char *dst, int imageW, int imageH, int KERNEL_RADIUS, int KERNEL_LENGTH)
{
    const   int ix = MAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = MAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix >= imageW || iy >= imageH)
        return;

    float sum = 0;

   /* #if(UNROLL_INNER)
        sum = convolutionColumn<2 * KERNEL_RADIUS>(x, y);
    #else*/
    #pragma unroll
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += tex2D(texSrc, x, y + (float)k) * Kernel[KERNEL_RADIUS - k];
        
   // #endif

     dst[MAD(iy, imageW, ix)] = (unsigned char)sum;
}

void convolutionColumnsGPU( unsigned char *dst, cudaArray *src, int imageW, int imageH, int KERNEL_RADIUS, int KERNEL_LENGTH )
{
    dim3 threads(16, 15);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    cudaBindTextureToArray(texSrc, src);
    convolutionColumnsKernel<<<blocks, threads>>>(
        dst,
        imageW,
        imageH,
        KERNEL_RADIUS,
        KERNEL_LENGTH
    );
    cudaUnbindTexture(texSrc);
}

/////////////////////////////////////// Combining the two blurs ////////////////////////////////////////////////
gpu_error_t gpu_blur( gpu_context_t *ctx , int KERNEL_RADIUS)
{	
	gpu_error_t error = GPU_OK;

	int KERNEL_LENGTH = (2 * KERNEL_RADIUS + 1);
	const int imageW = ctx->width;
    const int imageH = ctx->height;
    	
	float *tempKernel;
	unsigned char *in;
	tempKernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));

	in = ctx->output_buffer_1;
	
	cudaArray *src;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&src, &floatTex, imageW, imageH);
    
    unsigned char *d_Output;
    cudaMalloc((void **)&d_Output, imageW * imageH );   
    
	error = checkCudaError();
	
	////////////// calculating kernel //////////////
	float sum = 0;
    for(int i = 0; i < KERNEL_LENGTH; i++)
    {
    	float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
    	tempKernel[i] = expf(- dist * dist / 2);
    	sum += tempKernel[i];
    }
    for(int i = 0; i < KERNEL_LENGTH; i++)
        tempKernel[i] /= sum;            
	cudaMemcpyToSymbol(Kernel, tempKernel, KERNEL_LENGTH * sizeof(float));       
	////////////////////////////////////////////////
	
    cudaMemcpyToArray(src, 0, 0, in, imageW * imageH, cudaMemcpyHostToDevice);
    convolutionRowsGPU( d_Output, src, imageW, imageH, KERNEL_RADIUS, KERNEL_LENGTH);
    cudaThreadSynchronize();

 	if(checkCudaError() == GPU_OK)   
 	{
    	cudaMemcpyToArray(src, 0, 0, d_Output, imageW * imageH, cudaMemcpyDeviceToDevice);
    	convolutionColumnsGPU( d_Output, src, imageW, imageH, KERNEL_RADIUS, KERNEL_LENGTH);
		cudaThreadSynchronize();
	}
	cudaMemcpy(in, d_Output, imageW * imageH, cudaMemcpyDeviceToHost);
	cudaMemcpy( ctx->gpu_buffer_1, d_Output, imageW * imageH, cudaMemcpyDeviceToDevice);	// This is needed so that next filter is able to use gpu_buffer_1
	error = checkCudaError();

	cudaFree(d_Output);
	cudaFreeArray(src);
	return error;
	
}
