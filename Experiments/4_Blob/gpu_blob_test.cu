#include "cuda.h"
#include "cuda_runtime.h"

texture<uchar, 2, cudaReadModeElementType> segmentedTex;

__global__ int debugArr[10];

inline __device__ int findRoot(int* buf, int x) 
{
	int nextX;
    do {
	  nextX = x;
      x = buf[nextX];
    } while (x < nextX);
    return x;    
}

inline
__device__ void unionF(int* buf, unsigned char *buf_uchar, unsigned char seg1, unsigned char seg2, int reg1, int reg2, int* changed)
{
	if(seg1 == seg2) {			
		int newReg1 = findRoot(buf, reg1);		
		int newReg2 = findRoot(buf, reg2);	
	
		if(newReg1 > newReg2) {			
			atomicMin(buf+newReg1, newReg2);		
			buf_uchar[newReg1] = min( buf_uchar[newReg1], newReg2);		
			changed[0] = 1;			
		} else if(newReg2 > newReg1) {		
			atomicMin(buf+newReg2, newReg1);	
			buf_uchar[newReg2] = min( buf_uchar[newReg2], newReg2);
			changed[0] = 1;
		}			
	} 	
}

////////////////////////////////////////////////// Shared Labelling /////////////////////////////////////////////

__global__ void sharedLabellingKernel( unsigned char *gpu_in, int *gpu_labels, unsigned char *gpu_labels_uchar, const int pitch, const int dataWidth, const int segOff)
{
	int 	  x = blockIdx.x * blockDim.x + threadIdx.x;
    int       y = blockIdx.y * blockDim.y + threadIdx.y;
    int  offset = x + y * blockDim.x * gridDim.x;
    int shPitch = blockDim.x + 2;	// This is the pitch for shared memory of labels.

    int    localIndex = threadIdx.x + 1 + (threadIdx.y + 1) * shPitch; // This is the local address inside shared memory that has 1 pixel width of apron.
    int      newLabel = localIndex;
    int      oldLabel = 0;
    int         index = x + y * pitch; // This is the address of the pixel in global memory
    int segLocalIndex = localIndex + segOff;

	// This is the new syntax for declaring shared memory //
    extern __shared__ int sMem[];

    //shared flag that is used to check for the final solution on the processed tile 
	//if there are any two connected elements with different labels the flag is set to 1
	__shared__ int sChanged[1];

	////// Initializing the shared memory. Setting the boundary values to 0 i.e background /////
	if(threadIdx.x == blockDim.x-1) 
	{	
		sMem[localIndex+1] = 0;
		sMem[segLocalIndex+1] = 0;
	}
	if(threadIdx.x == 0) 
	{	
		sMem[localIndex-1] = 0;
		sMem[segLocalIndex-1] = 0;
	}
	if(threadIdx.y == blockDim.y-1) {			
		sMem[localIndex+shPitch] = 0;
		sMem[segLocalIndex+shPitch] = 0;

		if(threadIdx.x == 0) {			
			sMem[localIndex+shPitch-1] = 0;
			sMem[segLocalIndex+shPitch-1] = 0;
		}
		if(threadIdx.x == blockDim.x-1) {			
			sMem[localIndex+shPitch+1] = 0;
			sMem[segLocalIndex+shPitch+1] = 0;
		}	
	}
	if(threadIdx.y == 0) {			
		sMem[localIndex-shPitch] = 0;
		sMem[segLocalIndex-shPitch] = 0;
		if(threadIdx.x == 0) {			
			sMem[localIndex-shPitch-1] = 0;
			sMem[segLocalIndex-shPitch-1] = 0;
		}
		if(threadIdx.x == blockDim.x-1) {			
			sMem[localIndex-shPitch+1] = 0;
			sMem[segLocalIndex-shPitch+1] = 0;
		}	
	}

	/// VVV IMP: I think that the variables declared inside a __global__ function call are register variables, and not normal variables.
	/// The register variables are faster than shared memory. But dont overuse it due to threads throughput.
	unsigned char pixel;
	unsigned char nPixel[8];	// The neighbouring pixels.

	// Current pixel retrieved for processing //
	//pixel = tex2D( texSrc, x, y);  // This is with usage of textures.
	pixel = gpu_in[ offset ];	// This is with global memory call.
	
	sMem[segLocalIndex] = (int)pixel;// This step will load the segmentation shared memory with all the required pixels
	__syncthreads();

	//store data about segments into registers so that we don't have to access shared memory
	//(the data are never modified)
	nPixel[0] = sMem[segLocalIndex-shPitch-1];
	nPixel[1] = sMem[segLocalIndex-shPitch];
	nPixel[2] = sMem[segLocalIndex-shPitch+1];
	nPixel[3] = sMem[segLocalIndex-1];
	nPixel[4] = sMem[segLocalIndex+1];
	nPixel[5] = sMem[segLocalIndex+shPitch-1];
	nPixel[6] = sMem[segLocalIndex+shPitch];
	nPixel[7] = sMem[segLocalIndex+shPitch+1];

	while(!NULL)
	{
		//in first pass the newLabel is equal to the local address of the element
		sMem[localIndex] = newLabel;

		//reset the check flag for each block
		if((threadIdx.x | threadIdx.y) == 0) sChanged[0] = 0;
		oldLabel = newLabel;
		__syncthreads();

		//if the element is not a background, compare the element's label with its neighbors
		if(pixel != 0) 
		{	
			if( pixel == nPixel[0])
				newLabel = min( newLabel, sMem[localIndex-shPitch-1]);
			if( pixel == nPixel[1])
				newLabel = min( newLabel, sMem[localIndex-shPitch]);
			if( pixel == nPixel[2])
				newLabel = min( newLabel, sMem[localIndex-shPitch+1]);
			if( pixel == nPixel[3])
				newLabel = min( newLabel, sMem[localIndex-1]);
			if( pixel == nPixel[4])
				newLabel = min( newLabel, sMem[localIndex+1]);
			if( pixel == nPixel[5])
				newLabel = min( newLabel, sMem[localIndex+shPitch-1]);
			if( pixel == nPixel[6])
				newLabel = min( newLabel, sMem[localIndex+shPitch]);
			if( pixel == nPixel[7])
				newLabel = min( newLabel, sMem[localIndex+shPitch+1]);
				
		}
		__syncthreads();

		if( oldLabel > newLabel)
		{
			//if there is a neigboring element with a smaller label, update the equivalence tree of the processed element
			//(the tree is always flattened in this stage so there is no need to use findRoot to find the root)
	
			// VVVVIMP: This step is like merging of two trees together.
				
            //Be carefull when removing this function. Atomic is used to prevent multiple threads from accessing same memory.
			//It is like a particualar thread has acquired a lock on the address.
			
			
			atomicMin(sMem+oldLabel, newLabel); 
			//sMem[localIndex] = newLabel;
			//sMem[oldLabel] = min( sMem[oldLabel], newLabel);

			
			//set the flag to 1 bcoz it is necessary to perform another iteration of the CCL solver
			sChanged[0] = 1;
		}
		__syncthreads();

		if(sChanged[0] == 0) break;

		//flatten the equivalence tree
		newLabel = findRoot(sMem,newLabel);			
		__syncthreads();
	
	}	

	if(pixel == 0) newLabel = -1;	 // This is the labelling of the background pixel.
	else
	{	// The following loop translates each local label to a unique global label.
		//transfer the label into global coordinates 
		y = newLabel / (blockDim.x+2);
		x = newLabel - y*(blockDim.x+2);
		x = blockIdx.x*blockDim.x + x-1;
		y = blockIdx.y*blockDim.y + y-1;
		newLabel = x+y*dataWidth;	
	}	

	gpu_labels[index] = newLabel;
	gpu_labels_uchar[index] = (unsigned char)newLabel;
	
}	

void sharedLabelling( int *gpu_labels, unsigned char *gpu_labels_uchar, unsigned char *gpu_in, int imageW, int imageH, int threadsX, int threadsY)
{
	dim3 block( threadsX, threadsY, 1);
	dim3 grid( imageW/block.x, imageH/block.y, 1);

	int labelsSize = sizeof(int)*(threadsX+2)*(threadsY+2);
	int segSize = sizeof(int)*(threadsX+2)*(threadsY+2);

	sharedLabellingKernel<<< grid, block, labelsSize+segSize>>>(gpu_in, gpu_labels, gpu_labels_uchar, 240, 240,labelsSize/sizeof(int));
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////// merging borders ///////////////////////////////////////////////////////////////////////////

__global__ void mergeBordersKernel( int *gpu_labels, unsigned char *gpu_labels_uchar, int tileDim)
{

	int tileX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int tileY = (blockIdx.y * blockDim.y) + threadIdx.y;

	

	int offset;
	unsigned char seg;
	int nextTileDim = tileDim * blockDim.x;

	__shared__ sChanged[1];

	// Horizontal border
	if( threadIdx.y < blockDim.y-1)
	{
		int  y = (tileY + 1) * tileDim - 1;
		offset = threadIdx.x*tileDim + threadIdx.z;
		int  x = tileX * tileDim + threadIdx.z;

		seg = tex2D(segmentedTex, x+0.5f, y+0.5f); 
		if(seg != 0)
		{	int idx = x + y *pitch;
			if(offset > 0)unionF( gpu_labels, gpu_labels_uchar, seg, tex2D( segmentedTex, x-1, y+1), idx, idx+pitch-1, sChanged);
			unionF( gpu_labels, gpu_labels_uchar, seg, tex2D( segmentedTex, x, y+1), idx, idx+pitch, sChanged);
			if(offset < nextTileDim)unionF( gpu_labels, gpu_labels_uchar, seg, tex2D( segmentedTex, x+1, y+1), idx, idx+pitch+1, sChanged);			
		}
	}


}

void mergeBorders( unsigned char *gpu_in, int *gpu_labels, unsigned char *gpu_labels_uchar, int imageW, int imageH, int threadsX, int threadsY)
{

	int xTiles = 4;
	int yTiles = 4;
	int tileSize = 20;
	int threadsPerBlock = 20;

	size_t offset;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(&offset, &segmentedTex, gpu_in, &channelDesc, imageW, imageH, 240);

	while( tileSize < imageW || tileSize < imageH)
	{
		dim3 block( xTiles, yTiles, threadsPerBlocks);	
		dim3 grid( imageW/(xTiles * tileSize), imageH/(yTiles * tileSize), 1);
		mergeBordersKernel<<<grid,block>>>( gpu_labels, gpu_labels_uchar, tileSize);
	}

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float gpu_DetectBlob( unsigned char *in, unsigned char *labels)
{
	int   imageW = 240;
	int   imageH = 320;
	int threadsX = 20;
	int threadsY = 20;
	float elapsedtime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *gpu_labels;
    cudaMalloc( (void **)&gpu_labels, imageW * imageH * sizeof(int));

    //// This buffer is temporary and only used for debuggin purpose ////
    unsigned char *gpu_labels_uchar;
    cudaMalloc( (void **)&gpu_labels_uchar, imageW * imageH * sizeof(unsigned char));
	//////
	
	unsigned char *gpu_in;
	cudaMalloc( (void **)&gpu_in, 240 * 320);
	cudaMemcpy( gpu_in, in, 240 * 320, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start,0);

	///////// Shared Labelling ///////////
	
	sharedLabelling( gpu_labels, gpu_labels_uchar, gpu_in, imageW, imageH, threadsX, threadsY);
	
	//////////////////////////////////////

	///////// Merging borders ///////////

	mergeBorders( gpu_in, gpu_labels, gpu_labels_uchar, imageW, imageH, threadsX, threadsY);

	/////////////////////////////////////

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//cudaMemcpy( in, gpu_labels, 240*320, cudaMemcpyDeviceToHost);

	cudaMemcpy( labels, gpu_labels_uchar, 240*320, cudaMemcpyDeviceToHost);
    //cudaUnbindTexture(texSrc);
    

    return elapsedtime;
    

}
