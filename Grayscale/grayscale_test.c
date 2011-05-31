#include <stdio.h>
#include "gpu_grayscale.h"

#include "cv.h"
#include "highgui.h"

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	int key, i, width, height, fps, depth;
	unsigned char *pdata, *buffer = NULL;

	// load initial avi
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if ( !capture )
		return 1;

	fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	// initialize video output
	cvNamedWindow( "video", 0 );
	cvNamedWindow( "new_video", 0 );

	while ( key != 'q' ) {

		// read camera image
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		if ( buffer == NULL ) {
			// first time, create buffer according to the width/height of the first frame
			width = frame->width;
			height = frame->height;
			buffer = new unsigned char[width * height * 4];
			depth = frame->nChannels;
			assert(depth == 3);

			// create also the destination image
			new_frame = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
		}

		// convert from opencv to cuda color space
		// FIXME use depth here !
		pdata = (unsigned char*)frame->imageData;
		for( i = 0; i < width * height; i++)
		{
			buffer[i * 4 + 0] = pdata[i * 3 + 0];
			buffer[i * 4 + 1] = pdata[i * 3 + 1];
			buffer[i * 4 + 2] = pdata[i * 3 + 2];
		}

		// call the function from the cuda frame buffer
		gpu_grayscale(width, height, (unsigned char*)buffer);

		// convert from cuda to opencv colorspace
		pdata = (unsigned char*)new_frame->imageData;
		for( i = 0; i < width * height; i++)
		{
			pdata[i * 3 + 0] = buffer[i * 4 + 0];
			pdata[i * 3 + 1] = buffer[i * 4 + 1];
			pdata[i * 3 + 2] = buffer[i * 4 + 2];
		}

		// display the source video and the result
		cvShowImage( "video", frame );
		cvShowImage( "new_video", new_frame );

		// check 'q' pressed, and refresh opencv windows
		key = cvWaitKey( 1000 / fps );
	}

	///////// Free memory ///////////
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
