#include <stdio.h>
#include "Grayscale/gpu_grayscale.h"
#include "Threshold/gpu_threshold.h"
#include "cv.h"
#include "highgui.h"
#include "api.h"

#define GPU_ERROR(x) fprintf(stderr, "GPU: " #x "(%s)\n", gpu_error());

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	IplImage  *new_frame_1 = NULL;
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
	cvNamedWindow( "new_video_1", 0 );

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

			new_frame = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 4);
			new_frame_1 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
		}

		//////////// Setting up context buffer ///////////////
		if(gpu_set_input( ctx, (unsigned char *)frame->imageData) != GPU_OK)
		{
			GPU_ERROR("Unable to set context buffer");
			break;
		}

		////////////////////////// GPU Grayscale call /////////////////////////////

		if (gpu_grayscale(ctx) != GPU_OK)
		{
			GPU_ERROR("Unable to convert to grayscale");
			break;
		}

		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer");
			break;
		}
		cvSetData( new_frame, output_buffer, (frame->width * 4));
		cvShowImage( "new_video", new_frame );

		/////////// Setting up output buffer ////////////////
		/*
			FIXME:
			If I comment the lines from here on than everythin works fine (grayscale output can be seen)
			Once i uncomment the lines till "cvSetData( new_frame, output_buffer, (frame->width * 4));" (as shown below)
			the grayscale output gets all messed up. Can it be because of thr same output_buffer used ?????

		*/
		ctx->nchannels = 1;	//once i set up the number of channels to 1 output buffer will point to output_buffer_1
		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer");
			break;
		}

		//////// Dont forget to set ctx->nchannels variable for next filter ///////
		//cvSetData( new_frame, output_buffer, (frame->width * 4));
		//cvSetData( new_frame_1, output_buffer, frame->width );
		

		// display the source video and the result
		cvShowImage( "video", frame );
		//cvShowImage( "new_video", new_frame_1 );

		cvWaitKey(10);

	}

	///////// Free memory ///////////
	gpu_context_free( ctx );
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
