#include "Grayscale/gpu_grayscale.h"
#include "Threshold/gpu_threshold.h"
#include "GaussBlurTex/gpu_blur_tex.h"
#include "BgSub/gpu_sub.h"
#include "Amplify/gpu_amplify.h"
#include "Blobs/gpu_blob.h"
#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include "API/api.h"

int main( int argc, char** argv )
{
	IplImage  *frame, *new_frame = NULL;
	IplImage  *new_frame_1 = NULL;
	IplImage  *new_frame_2 = NULL;
	IplImage  *new_frame_3 = NULL;
	IplImage  *new_frame_4 = NULL;
	IplImage  *new_frame_5 = NULL;
	unsigned char *output_buffer, *staticBg;
	gpu_context_t *ctx = NULL;
	bgMode mode = STATIC;
	bool flag = true; // It is used to capture only first frame as background. 
	int th_value = 25;
	int fps;
	int counter = 0;
		
	/// load initial avi ///
	CvCapture *capture = cvCaptureFromAVI( "out.avi" );
	if ( !capture )
		return 1;

	fps = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	/// initialize video output ///
	cvNamedWindow( "video", 0 );	
	cvMoveWindow( "video", 0, 50);
	
	cvNamedWindow( "Grayscale", 0 );
	cvMoveWindow( "Grayscale", 340, 50);
	
	cvNamedWindow( "BgFilter", 0 );
	cvMoveWindow( "BgFilter", 680, 50);
	
	cvNamedWindow( "Blurring", 0 );
	cvMoveWindow( "Blurring", 1020, 50);

	cvNamedWindow( "Amplify", 0 );
	cvMoveWindow( "Amplify", 0, 400);

	cvNamedWindow( "Threshold", 0 );
	cvMoveWindow( "Threshold", 340, 400);

	cvNamedWindow( "Blobs", 0 );
	cvMoveWindow( "Blobs", 680, 400);

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

			staticBg = (unsigned char *)malloc( frame->width * frame->height * sizeof(unsigned char));
			new_frame = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_1 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_2 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_3 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_4 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
			new_frame_5 = cvCreateImageHeader(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);
		}
		
		/////////////////////// Setting up context buffer /////////////////////////
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
			GPU_ERROR("Unable to get output buffer after grayscale");
			break;
		}
		cvSetData( new_frame, output_buffer, frame->width);
		cvShowImage( "Grayscale", new_frame );

		////////////// Setting necessary variables for Bg subtraction //////////
		if( (mode == STATIC) && (flag == true))
		{
			for(int i = 0; i < ctx->width * ctx->height; i++)
				staticBg[i] = new_frame->imageData[i];
		}
				
		flag = false;

		/* Background subtraction */
		
		if(gpu_sub( ctx, staticBg) != GPU_OK) 
		{
			GPU_ERROR("unable to remove background ");
		}

		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer after Bg");
			break;
		}
		cvSetData( new_frame_1, output_buffer, frame->width);
		cvShowImage( "BgFilter", new_frame_1 );
		
	    //////////////////////// GPU blurring Call /////////////////////////////
		if(gpu_blur( ctx, 3) != GPU_OK)
		{
			GPU_ERROR("Unable to blur the image");
		}

		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer after Blur");
			break;
		}
		cvSetData( new_frame_2, output_buffer, frame->width);
		cvShowImage( "Blurring", new_frame_2 );

		////////////////////////// GPU Amplify call ////////////////////////////

		if(gpu_amplify( ctx, 1.5f) != GPU_OK)
		{
			GPU_ERROR("Unable to threshold");
		}

		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer after amp");
			break;
		}
		cvSetData( new_frame_3, output_buffer, frame->width);
		cvShowImage( "Amplify", new_frame_3 );

		//////////////////////// GPU Threshold Call ////////////////////////////

		if(gpu_threshold( ctx, th_value) != GPU_OK)
		{
			GPU_ERROR("Unable to threshold");
		}

		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer threshold");
			break;
		}
		cvSetData( new_frame_4, output_buffer, frame->width);
		cvShowImage( "Threshold", new_frame_4 );

		/////////////////////// GPU Blob Detection /////////////////////////////

		if( gpu_DetectBlob(ctx) != GPU_OK)
		{
			GPU_ERROR("Unable to detect blobs");
			break;
		}
		
		if (gpu_get_output(ctx, &output_buffer) != GPU_OK)
		{
			GPU_ERROR("Unable to get output buffer after blobs");
			break;
		}
		cvSetData( new_frame_5, output_buffer, frame->width);
		cvShowImage( "Blobs", new_frame_5 );


		/////////////////////////////////////////// //////////////////////////

		

		// display the source video and the result
		cvShowImage( "video", frame );

		counter ++;
		
		cvWaitKey(10);

	}

	///////// Free memory ///////////
	gpu_context_free( ctx );
	cvReleaseCapture( &capture );
	cvDestroyWindow( "video" );
	cvDestroyWindow( "new_video" );

	return 0;
}
