#include "cuda.h"
#include "stdio.h"

__device__ unsigned char deviceFlag[1];

__device__ int check[1];

/* Description:
*    Takes a binary input image and labels all non-background elements
*    with a unique value
*  Parameters:
*     deviceFlag: flag to indicate whether or not the image has changed
*          input: the input image
*         output: the output image
*    numElements: the number of pixels in the input/output image
*/
__global__ void uniqueLabel(unsigned char *input,
							unsigned int *output,
							unsigned char *outputU,
							unsigned int numElements)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < numElements)
	{
		if (input[idx] != 0)
		{
			output[idx] = (unsigned int) idx+1;
			outputU[idx] = (unsigned char) idx+1;
		} else {
			output[idx] = -1;
			outputU[idx] = -1;
		}
	}

	// Initialize the flag to 0 (image has not changed)
	if (idx == 0)
	{
		deviceFlag[0] = 0;
	}
}

/* Description:
*    Takes a input image and assigns a unique label to each blob
*    Note: the binary image must be preprocessed by the 'uniqueLabel'
*    kernel
*  Parameters:
*     deviceFlag: flag to indicate whether or not the image has changed
*         output: the input/output labeled image
*        numRows: number of rows in the image
*        numRows: number of columns in the image
*    numElements: the number of pixels in the input/output image
*/
__global__ void minLabel( unsigned int *output,
						 unsigned char *outputU,
						 unsigned int numRows,
						 unsigned int numCols,
						 unsigned int numElements)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int i, j;
	unsigned int minLabel;
	unsigned int nPixel, sPixel, ePixel, wPixel;
	unsigned int nwPixel, nePixel, swPixel, sePixel;

	i = idx % numRows; // Current Row
	j = idx / numRows; // Current Column

	unsigned int isNotAtTop = (i != 0);
	unsigned int isNotAtBottom = (i != numRows-1);
	unsigned int isNotAtRight = (j != 0);
	unsigned int isNotAtLeft = (j != numCols-1);
	unsigned int writeFlag;

	// Perform the labeling
	if (idx < numElements)
	{
		if (output[idx] != -1)
		{
			for (int a = 0; a < 7; a++)
			{
				writeFlag = 0;
				minLabel = output[idx];

				if (isNotAtTop)
				{
					// North Pixel
					nPixel = output[j+((i-1)*numRows)];


					if ( (nPixel < minLabel) && (nPixel != 0) )
					{
						minLabel = nPixel;
						writeFlag = 1;
					}

					if (isNotAtRight)
					{
						// North-West Pixel
						nwPixel = output[ j-1+((i-1) * numRows)];

						if ( (nwPixel < minLabel) && (nwPixel != 0) )
						{
							minLabel = nwPixel;
							writeFlag = 1;
						}
					}

					if (isNotAtLeft)
					{
						// North-East Pixel
						nePixel = output[ j+1+((i-1) * numRows)];

						if ( (nePixel < minLabel) && (nePixel != 0) )
						{
							minLabel = nePixel;
							writeFlag = 1;
						}
					}
				}

				if (isNotAtBottom)
				{

					sPixel = output[ j+((i+1)*numRows)];

					if ( (sPixel < minLabel) && (sPixel != 0) )
					{
						minLabel = sPixel;
						writeFlag = 1;
					}

					if (isNotAtRight)
					{
						// South-West Pixel
						swPixel = output[ j-1+((i+1)*numRows)];

						if ( (swPixel < minLabel) && (swPixel != 0) )
						{
							minLabel = swPixel;
							writeFlag = 1;
						}
					}

					if (isNotAtLeft)
					{
						// South-East Pixel
						sePixel = output[j+1+((i+1)*numRows)];

						if ( (sePixel < minLabel) && (sePixel != 0) )
						{
							minLabel = sePixel;
							writeFlag = 1;
						}
					}
				}
				
				if (isNotAtRight)
				{
					// West Pixel
					wPixel = output[j-1+(i*numRows)];

					if ( (wPixel < minLabel) && (wPixel != 0) )
					{
						minLabel = wPixel;
						writeFlag = 1;
					}
				}

				if (isNotAtLeft)
				{
					// East Pixel
					ePixel = output[i+1+(i*numRows)];

					if ( (ePixel < minLabel) && (ePixel != 0) )
					{
						minLabel = ePixel;
						writeFlag = 1;
					}
				}

				output[idx] = (unsigned int) minLabel;
				outputU[idx] = (unsigned char) minLabel;

				if (writeFlag)
				{
					deviceFlag[0] = 1;
				}
			}
		}
	}
}


float gpu_BlobTest( unsigned char *in, unsigned char *labels)
{

	unsigned char *gpu_in;
	cudaMalloc( (void **)&gpu_in, 240 * 320 * sizeof(unsigned char));
	cudaMemcpy( gpu_in, in, 240*320*sizeof(unsigned char),cudaMemcpyHostToDevice);

	unsigned int *gpu_labels, *labels_uint;
	cudaMalloc( (void **)&gpu_labels, 240 * 320 * sizeof(unsigned int));
	labels_uint = (uint *)malloc(240*320*sizeof(uint));

	unsigned char *gpu_labels_uchar;
	cudaMalloc( (void **)&gpu_labels_uchar, 240 * 320 * sizeof(unsigned char));

	dim3 block(240);
	dim3 grid(320);

	unsigned char localFlag;
	float a=234;
	int counter = 0;

	uniqueLabel<<<grid,block>>>( gpu_in, gpu_labels, gpu_labels_uchar, 240*320);
	//minLabel<<<grid,block>>>( gpu_labels, gpu_labels_uchar, 320, 240, 240*320);

	while(1)
	{
	  	 // Make each pixel take the minimum label of itself
 		 //   and its neighbors
		 minLabel<<<grid,block>>>( gpu_labels, gpu_labels_uchar, 320, 240, (240*320));
  		  cudaThreadSynchronize();
                
 		 // Load the flag into CPU memory
  		cudaMemcpy(&localFlag, deviceFlag, 1, cudaMemcpyDeviceToHost);

  		counter++;
                
 		 if (localFlag == 0)
		  {
      		  // The image has not changed
		        //   labeling has completed
		        a=345;
		        break;
		  } else {
        		// The image has changed
        		//   labeling is not complete
		        localFlag = 0;
		        a=123;
 		 }
                
  	// Reset the flag and sent it back to global GPU memory
 	 cudaMemcpy(deviceFlag, &localFlag,  1, cudaMemcpyHostToDevice);
	}
	cudaMemcpy( labels, gpu_labels_uchar, 240*320*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy( labels_uint, gpu_labels, 240*320*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	FILE *file;
	file = fopen("file.txt","a+"); // apend file (add text to a file or create a file if it does not exist.
	for(int i=0;i<240*320;i++)
	{
		if((i>239) && (i%240==0))
			fprintf(file,"\n");
		fprintf(file,"%d ", labels_uint[i]); 
	}
	fprintf(file,"\n");
	fclose(file); //done!

	
	return a+counter;

}

