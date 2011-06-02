#include <stdio.h>
#include "gpu_grayscale.h"
#include "cv.h"
#include "highgui.h"
#include "api.h"

#define GPU_ERROR(x) fprintf(stderr, "GPU: " #x "(%s)\n", gpu_error());

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	unsigned char *output_buffer;
	gpu_context_t *ctx = NULL;
	int fps;

	/// load initial avi ///
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if ( !capture )
		return 1;

	fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	/// initialize video output ///
	cvNamedWindow( "video", 0 );
	cvNamedWindow( "new_video", 0 );

	if ( gpu_context_create(&ctx) != GPU_OK )
	{
		GPU_ERROR("Unable to create GPU context");
		return 0;
	}

	while (!NULL)
	{
		/// read camera image ///
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		if ( new_frame == NULL )
		{
			assert(frame->nChannels == 3);
			if ( gpu_context_init( ctx, frame->height, frame->width, frame->nChannels, GPU_MEMORY_HOST) != GPU_OK )
			{
				GPU_ERROR("Unable to initialize GPU context");
				break;
			}
		}

		//////////// Setting up context buffer ///////////////
		if(gpu_set_input( ctx, (unsigned char *)frame->imageData) != GPU_OK)
		{
			GPU_ERROR("Unable to set context buffer");
			break;
		}

		////////////////////////// CUDA calls /////////////////////////////

		if (gpu_grayscale(ctx) != GPU_OK)
		{
			GPU_ERROR("Unable to convert to grayscale");
			break;
		}

		/////////// Setting up output buffer ////////////////
		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer");
			break;
		}

		cvSetData( new_frame, output_buffer, (frame->width * frame->nChannels));

		// display the source video and the result
		cvShowImage( "video", frame );
		cvShowImage( "new_video", new_frame );

	}

	///////// Free memory ///////////
	gpu_context_free( ctx );
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
