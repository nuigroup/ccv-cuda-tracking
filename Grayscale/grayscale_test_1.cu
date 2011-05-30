#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include "cuda.h"

float elapsedtime;

////////////////////////////////////CUDA Programming///////////////////////////////////////////////////////////////

__global__ void convert(uchar4 *gpu_in)
{
	
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
	}	
	
}
///////////////// CUDA function call wrapper /////////////////
void tograyscale(unsigned char *in)
{
	uchar4 *gpu_in;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	////////////////////////// Time consuming Task //////////////////////////////////	
	cudaMalloc((void **)&gpu_in, (240*320*4*sizeof(unsigned char)));	
	cudaMemcpy(gpu_in, in, (240*320*4*sizeof(unsigned char)), cudaMemcpyHostToDevice);

	dim3 grid(18,18);
	dim3 block(16,16);
	convert<<<grid,block>>>(gpu_in);

	cudaMemcpy( in, gpu_in, (240*320*4*sizeof(unsigned char)), cudaMemcpyDeviceToHost);
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
	//CvSize size = new CvSize(240,320);
	
	/* load the AVI file */
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if( !capture ) return 1;	
	
	int fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
	
	/* display video */
	cvNamedWindow( "video", 0 );
	cvNamedWindow( "new_video", 0 );

	buffer = new unsigned char[240*320*4];
	
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
		
		tograyscale(buffer);

		////////////// Creating a new frame /////////////////
		new_frame = cvCreateImage(cvSize(240,320),IPL_DEPTH_8U,3);
		for(i=0;i<240*320;i++)
		{
		new_frame->imageData[i *3 + 0] = buffer[i * 4 + 0]; 
		new_frame->imageData[i *3 + 1] = buffer[i * 4 + 1];
		new_frame->imageData[i *3 + 2] = buffer[i * 4 + 2];
		}	
		//printf("Time taken is %f ",elapsedtime);
		
		//////// display frame /////////
		cvShowImage( "video", frame );
		cvShowImage( "new_video", new_frame );
		
		/// quit if user press 'q' /////
		key = cvWaitKey( 1000 / fps );
	}

	
	///////// Free memory ///////////
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
