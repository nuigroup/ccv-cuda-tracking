#include <stdio.h>
#include "Grayscale/gpu_grayscale.h"
#include "Threshold/gpu_threshold.h"
#include "./GaussBlurTex/gpu_blur_tex.h"
#include "cv.h"
#include "highgui.h"
#include "api.h"

#define GPU_ERROR(x) fprintf(stderr, "GPU: " #x "(%s)\n", gpu_error());

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	IplImage  *new_frame_1 = NULL;
	IplImage  *new_frame_2 = NULL;
	unsigned char *output_buffer;
	gpu_context_t *ctx = NULL;
	int th_value = 20;
	int fps;

	/// load initial avi ///
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if ( !capture )
		return 1;

	fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	/// initialize video output ///
	cvNamedWindow( "video", 0 );	
	cvMoveWindow( "new_video", 30, 30);
	
	cvNamedWindow( "new_video", 0 );
	cvMoveWindow( "new_video", 280, 280);
	
	cvNamedWindow( "new_video_1", 0 );
	cvMoveWindow( "new_video_1", 620, 120);
	
	cvNamedWindow( "new_video_2", 0 );
	cvMoveWindow( "new_video_2", 1020, 120);

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

			new_frame = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_1 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_2 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
		}
		ctx->nchannels = 3;   //VVVIMP to do it in order to make calculations according to color image rather than grayscale one..

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

		///// By changing the number of chanels we can target the respective output and gpu buffers.....
		ctx->nchannels = 1;		
		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer");
			break;
		}
		cvSetData( new_frame, output_buffer, frame->width);
		cvShowImage( "new_video", new_frame );
	
		//////////////////////// GPU blurring Call /////////////////////////////
		if(gpu_blur( ctx, 8) != GPU_OK)
		{
			GPU_ERROR("Unable to blur the image");
		}
		gpu_get_output(ctx, &output_buffer);
		cvSetData( new_frame_2, output_buffer, frame->width);
		cvShowImage( "new_video_2", new_frame_2 );

		/////////////////////////// GPU Threshold Call /////////////////////////////

		if(gpu_threshold( ctx, th_value) != GPU_OK)
		{
			GPU_ERROR("Unable to convert to grayscale");
		}

		gpu_get_output(ctx, &output_buffer);
		cvSetData( new_frame_1, output_buffer, frame->width);
		cvShowImage( "new_video_1", new_frame_1 );

		// display the source video and the result
		cvShowImage( "video", frame );
		
		cvWaitKey(10);

	}

	///////// Free memory ///////////
	gpu_context_free( ctx );
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
