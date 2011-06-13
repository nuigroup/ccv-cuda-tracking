#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include "cuda.h"

float elapsedtime;
texture<uchar4, 2, cudaReadModeElementType> texSrc;

////////////////////////////////////CUDA Programming///////////////////////////////////////////////////////////////

__global__ void convert(unsigned char *in_1)
{

	int tx = threadIdx.x + __mul24(blockIdx.x * blockDim.x);
	int ty = threadIdx.y + __mul24(blockIdx.y * blockDim.y);
	int offset = tx + ty * blockDim.x * gridDim.x;
    const float  x = (float)tx + 0.5f;
    const float  y = (float)ty + 0.5f;

	if(tx >= 240 || ty >= 320)
        return;

	uchar4 temp;
	temp = tex2D(texSrc,x,y);
	//in_1[offset] = temp.x;
    
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
		/*float color = 0.3 * (gpu_in[offset].x) + 0.6 * (gpu_in[offset].y) + 0.1 * (gpu_in[offset].z);
		gpu_in[offset].x = color;
		gpu_in[offset].y = color;
		gpu_in[offset].z = color;
		gpu_in[offset].w = 0;*/
	/*
		sh_Tile[sh_offset * 4 + 0] = (int) (0.3 * sh_Tile[sh_offset * 4 + 0] + 0.6 * sh_Tile[sh_offset * 4 + 1] + 0.1 * sh_Tile[sh_offset * 4 + 2]); 
	}
	__syncthreads();
	
	gpu_in[offset].x = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].y = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].z = sh_Tile[sh_offset * 4 + 0];
	gpu_in[offset].w = 0;*/

	
	
}
///////////////// CUDA function call wrapper /////////////////
void tograyscale(unsigned char *in, unsigned char * in_1)
{
	//uchar4 *gpu_in;

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
	convert<<<grid,block>>>(in_1);

	//cudaMemcpy( in, gpu_in, (240*320*4*sizeof(unsigned char)), cudaMemcpyDeviceToHost);
	/////////////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame;
	int key, i;
	unsigned char *pdata, *buffer;
		
	/* load the AVI file */
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	
	if( !capture ) return 1;	
	
	int fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
	
	/* display video */
	cvNamedWindow( "video", 0 );
	cvNamedWindow( "new_video", 0 );

	//cudaHostAlloc( (void **) &buffer, sizeof(unsigned char) * 240 * 320 * 4, cudaHostAllocDefault);
	buffer = new unsigned char[240*320*4]; 
	new_frame = cvCreateImage(cvSize(240,320),IPL_DEPTH_8U,1);
	
	while( key != 'q' ) {
		//////// get a frame //////////
		frame = cvQueryFrame( capture );
		
		/////// always check /////////
		if( !frame ) break;
	
		//printf(" %d ,%d ,%d , %d, ",frame->nSize,frame->width,frame->height,frame->nChannels);

		////////////// Padding the fourth byte ////////////		
		pdata = (unsigned char *)frame->imageData;
		for(i=0;i< 240*320 ;i++)
		{
			buffer[i * 4 + 0] = pdata[i * 3 + 0];
			buffer[i * 4 + 1] = pdata[i * 3 + 1];
			buffer[i * 4 + 2] = pdata[i * 3 + 2];
		}
		////////////// Call to CUDA function //////////////
		
		tograyscale(buffer, (unsigned char *)new_frame->imageData);

		////////////// Creating a new frame /////////////////
		/*for(i=0;i<240*320;i++)
		{
		new_frame->imageData[i * 3 + 0] = buffer[i * 4 + 0]; 
		new_frame->imageData[i * 3 + 1] = buffer[i * 4 + 1];
		new_frame->imageData[i * 3 + 2] = buffer[i * 4 + 2];
		}*/	
		printf("Time taken is %f ",elapsedtime);
		
		//////// display frame /////////
		cvShowImage( "video", frame );
		cvShowImage( "new_video", new_frame );
		
		/// quit if user press 'q' /////
		key = cvWaitKey( 10 );
	}

	
	///////// Free memory ///////////
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
