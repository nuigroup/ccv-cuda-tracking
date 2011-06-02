#include <stdio.h>
#include "gpu_grayscale.h"
#include "cv.h"
#include "highgui.h"
#include "api.h"

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	unsigned char *output_buffer = NULL;
	int fps;
	float elapsedTime;	// Used	to measure the time taken by the filter.

	/// load initial avi ///
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if ( !capture )
		return 1;

	fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	/// initialize video output ///
	cvNamedWindow( "video", 0 );
	cvNamedWindow( "new_video", 0 );

	gpu_context *ctx;
	
	if(gpu_context_create(ctx) != No_error)
	{
		fprintf( stderr,"Unable to create GPU context");
		return 0;
	}

	while (!NULL) 
	{
		/// read camera image ///
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		if ( output_buffer == NULL ) {
			
			assert(frame->nChannels == 3);
			if (gpu_context_init( ctx, frame->height, frame->width, frame->nChannels, 2) != No_error)
			{
				fprintf(stderr,"Unable to initialize GPU context");
				break;
			}
			
			output_buffer = (unsigned char *) malloc((frame->width)*(frame->height)*(frame->nChannels));
			new_frame = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, frame->nChannels);
			cvSetData( new_frame, output_buffer, (frame->width * frame->nChannels));
		}

		//////////// Setting up context buffer ///////////////
		if(gpu_set_input( ctx, (unsigned char *)frame->imageData) != No_error)
		{
			fprintf(stderr,"Unable to set context buffer");
			break;
		}

		////////////////////////// CUDA calls /////////////////////////////

		if( gpu_grayscale(ctx->width, ctx->height, (unsigned char*)ctx->buffer) != No_error )
		{
			fprintf(stderr,"Unable to convert to grayscale");
			break;
		}
		
		/////////// Setting up output buffer ////////////////
		if(gpu_get_input( ctx, output_buffer) != No_error)
		{
			fprintf(stderr, "Unable to set context buffer");
			break;
		}
		cvSetData( new_frame, output_buffer, (frame->width * frame->nChannels));		

		// display the source video and the result
		cvShowImage( "video", frame );
		cvShowImage( "new_video", new_frame );

	}
	
	///////// Free memory ///////////
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
