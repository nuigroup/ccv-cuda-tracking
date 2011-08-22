/*
// This source file contains the Cuda Code for Blob Detection of a source Image.
// It is a part of Cuda Image Processing Library .
// Copyright (C) 2011 Remaldeep Singh

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/



/*	
	In the shared memory the labelSharedMemory contains the index value of the pixel as root. Labelling is done by storing value of indices at the pixels.
	The index value stored is the one that is minimum from its neighbouring 8 pixels.
							
	-------------------------------------------------------------------------
	|							Shared Memory								|
	|	-------------------------		-----------------------------		|
	|	|						|		|							|		|
	|	|						|		|							|		|
	|	|		Label			|		|		  Segment 			|		|
	|	|	Shared Memory		|		|	   Shared Memory    	|		|
	|	|						|		|							|		|
	|	|						|		|							|		|
	|	|						|		|							|		|
	|	-------------------------		-----------------------------		|
	-------------------------------------------------------------------------

	The best way to do labelling is using disjoint set datasctructure(Union Find DS).
	See Wikipidea
	Dont Use __mul24 for devices with compute capability >= 2.0.
	Hence define the apropriate flags accordingly.
	The gpu_labels_uchar is only for debugging. Remove it when you are done.
*/


#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "../API/api.h"

inline __device__ int findRoot(int* buf, int x) 
{
	int nextX;
    do {
	  nextX = x;
      x = buf[nextX];
    } while (x < nextX);
    return x;    
}

inline __device__ void unionTrees(int* buf, unsigned char *buf_uchar, unsigned char seg1, unsigned char seg2, int reg1, int reg2, int* changed)
{
	if(seg1 == seg2) 
	{			
		int newReg1 = findRoot(buf, reg1);		
		int newReg2 = findRoot(buf, reg2);	
	
		if(newReg1 > newReg2) {			
			atomicMin(buf+newReg1, newReg2);		
			//gpuMin(buf+newReg1,newReg2);
			buf_uchar[newReg1] = min( buf_uchar[newReg1], newReg2);		
			changed[0] = 1;			
		} else if(newReg2 > newReg1) {		
			atomicMin(buf+newReg2, newReg1);	
			//gpuMin1(buf+newReg2,newReg2);
			buf_uchar[newReg2] = min( buf_uchar[newReg2], newReg2);
			changed[0] = 1;
		}			
	} 	
}

texture<unsigned char, 2, cudaReadModeElementType> texSrc;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*													  Local labelling of Blobs 															   */
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void localLabelling( unsigned char *gpu_in, int *gpu_labels, unsigned char *gpu_labels_uchar, const int pitch, const int segOff, const int dataWidth)
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
	
	sMem[segLocalIndex] = (int)pixel;	// This step will load the segmentation shared memory with all the required pixels
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
			//VVVVIMP: This step is like merging of two trees together.				
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

/////////////////////////////////////////////// Merge Borders ////////////////////////////////////////////////////////////////////////

__global__ void mergeEquivalenceTreesOnBordersKernel( int* dLabelsInOut, unsigned char *gpu_labels_uchar, const int pitch, const int tileDim);

void mergeBorders( int *gpu_labels, unsigned char *gpu_labels_uchar, int threadsX, int threadsY, int imageW, int imageH)
{

/*
	int xTiles = 4;
	int yTiles = 4;
	int threadsPerBlock = threadsX;	// This denotes the no. of pixels in borders to be merged at a time.... If the size of border is large we can also increment these threads
	int tileSize = threadsX;

	dim3 block(xTiles,yTiles,threadsPerBlock);
	dim3 grid(imageW/(block.x*block.z), imageH/(block.y*block.z));

	//merge<<<grid,block>>>( gpu_labels, gpu_labels_uchar, tileSize, imageW);
	//mergeEquivalenceTreesOnBordersKernel<<<grid,block>>>(gpu_labels, gpu_labels_uchar, imageW, tileSize);
*/

	int tileSize = threadsX;
	size_t offset;
			
	while(tileSize < imageW || tileSize < imageH) 
	{
		//compute the number of tiles that are going to be merged in a singe thread block
		int xTiles = 4;
		int yTiles = 4;
		if(xTiles*tileSize > imageW) xTiles = imageW / tileSize;
		if(yTiles*tileSize > imageH) yTiles = imageH / tileSize;
		//the number of threads that is going to be used to merge neigboring tiles
		int threadsPerBlock = threadsX;
		if(tileSize < threadsPerBlock) threadsPerBlock = tileSize;
		dim3 block(xTiles,yTiles,threadsPerBlock);
		dim3 grid(imageW/(block.x*tileSize), imageH/(block.x*tileSize), 1);
//fprintf(stderr,"I was here\n");
		//call KERNEL 2
		mergeEquivalenceTreesOnBordersKernel<<<grid, block>>>( gpu_labels, gpu_labels_uchar, imageW, tileSize);
//fprintf(stderr,"I was here %d %d %d %dx %dy \n", tileSize, grid.x, grid.y, xTiles, yTiles);
		if(yTiles > xTiles) tileSize = yTiles * tileSize;
		else tileSize = xTiles * tileSize;
				
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mergeEquivalenceTreesOnBordersKernel( int* dLabelsInOut, unsigned char *gpu_labels_uchar, const int pitch, const int tileDim)
{
	//local tileX and Y are stored directly in blockIdx.x and blockIdx.x
	//all threads for each block are stored in the z-dir of each block (threadIdx.z)
	int tileX = threadIdx.x + blockIdx.x * blockDim.x;	
	int tileY = threadIdx.y + blockIdx.y * blockDim.y;
	//the number of times each thread has to be used to process one border of the tile
	int threadIterations = tileDim / blockDim.z;
	//dimensions of the tile on the next level of the merging scheme
	int nextTileDim = tileDim * blockDim.x;
	
	unsigned char seg;
	int offset;
	
	//shared variable that is set to 1 if an equivalence tree was changed
	__shared__ int sChanged[1];
	while(1) {		
		//reset the check variable
		if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			sChanged[0] = 0;			
		 }		
		__syncthreads();
		//first process horizontal borders that are between merged tiles (so exclude tiles from the last row)
		if(threadIdx.y < blockDim.y-1) {
			//the horizontal border corresponds to the last row of the tile
			unsigned int y = (tileY+1)*tileDim-1;	
			//offset of the element from the left most boundary of the tile
			offset = threadIdx.x*tileDim + threadIdx.z;
			unsigned int x = tileX*tileDim + threadIdx.z;
#pragma	unroll			
			for(int i=0;i<threadIterations;++i) {					
				//load the segment data for the element
				seg = tex2D(texSrc, x, y); 
				if(seg != 0) {		
					//address of the element in the global space
					int idx = x+y*pitch;				
					//perform the unionTrees operation on neigboring elements from other tiles that are to be merged with the processed tile
					if(offset>0) unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x-1, y+1), idx, idx-1+pitch, sChanged);
					unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x, y+1), idx, idx+pitch, sChanged);
					if(offset<nextTileDim-1) unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x+1, y+1), idx, idx+1+pitch, sChanged);
					
				}
				//set the processed element to the next in line on the same boundary (in case the threads are used for multiple elements on the boundary)
				x += blockDim.z;
				offset += blockDim.z;
			}
		}
		// vertical right borders
		if(threadIdx.x < blockDim.x-1) 
		{
			unsigned int x = (tileX+1)*tileDim-1;		
			offset = threadIdx.y*tileDim + threadIdx.z;
			unsigned int y = tileY*tileDim+threadIdx.z;
#pragma unroll
			for(int i=0;i<threadIterations;++i) 
			{			
				seg = tex2D(texSrc, x, y); 
				if(seg != 0) 
				{
					int idx = x+y*pitch;
					//perform the unionTrees operation on neigboring elements from other tiles that are to be merged with the processed tile
					if(offset>0) unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x+1, y-1), idx, idx+1-pitch, sChanged);
					unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x+1, y), idx, idx+1, sChanged);
					if(offset<nextTileDim-1) unionTrees(dLabelsInOut, gpu_labels_uchar, seg, tex2D(texSrc, x+1, y+1), idx, idx+1+pitch, sChanged);			
				}	
				y += blockDim.z;
				offset += blockDim.z;
			}		
		}		
		__syncthreads();
		
		if(sChanged[0] == 0) 		
			break;	
		
		__syncthreads();
	}	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*													Flattening of all the elements															*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void flattenEquivalenceTreesKernel(int* gpu_labels_out, int* gpu_labels_in, unsigned char *gpu_labels_uchar, unsigned int pitch, const int dataWidth, int *gpu_nRegions)												
{
	unsigned int     x = (blockIdx.x*blockDim.x)+threadIdx.x;
    unsigned int     y = (blockIdx.y*blockDim.y)+threadIdx.y;  
    unsigned int index = x+y*pitch;
    unsigned int label = gpu_labels_in[index];

	unsigned int newLabel;

	if((label != -1) && (label != index))
	{
		newLabel = findRoot( gpu_labels_in, label);

		if(newLabel < label)
		{
			gpu_labels_out[index] = newLabel;
			gpu_labels_uchar[index] = (unsigned char)newLabel;
		}
	}
	__syncthreads();

	if(gpu_labels_out[index] == index)
	{
		atomicAdd( gpu_nRegions, 1);
	}
	
}

void flattenTrees( int *gpu_labels, unsigned char *gpu_labels_uchar, int threadsX, int threadsY, int imageW, int imageH, int *gpu_nRegions)
{	
	dim3 block(threadsX, threadsY, 1);
    dim3 grid(imageW / block.x, imageH / block.y, 1);

	cudaMemset( gpu_nRegions, 0, sizeof(int));
    flattenEquivalenceTreesKernel<<<grid,block>>>( gpu_labels, gpu_labels, gpu_labels_uchar, imageW, imageW, gpu_nRegions);
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* 													 Calculating centroid 															  */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: Try and make it volatile for masively random access in case of blob centroid.

__global__ void calcCentroidKernel( int *gpu_labels, int *gpu_nRegions, int *gpu_regionOff, int *gpu_regionSize, int *gpu_centroid, int *i, int pitch)
{
	int     x = (blockIdx.x*blockDim.x)+threadIdx.x;
    int     y = (blockIdx.y*blockDim.y)+threadIdx.y;  
    int index = x+y*pitch;
    int     j = 0;

	//// Implement this critical section with atomics. The followning code wont work... :(
	if( gpu_labels[index] == index)
	{
		atomicAdd( i, 1);						
		atomicAdd( gpu_regionOff + (*i), index);		// Change i to *i
	}

	// I dont think there is a need to __syncthreads but try it, if it doesnt work out.

	// Finding the index where offset is stored.	
	for( j=0; j < *gpu_nRegions; j++)
	{
		if( gpu_labels[index] == gpu_regionOff[j])
			break;
	}
	
	// Storing the value of centroid at 2*j position.
	if( gpu_labels[index] != -1)
	{
		atomicAdd( gpu_centroid+(2*j), x);
		atomicAdd( gpu_centroid+(2*j)+1, y);
		atomicAdd( gpu_regionSize+j, 1);
	}	

}

__global__ void calcCentroidSharedKernel(int *gpu_labels,int *gpu_nRegions,int *gpu_regionOff,int *gpu_regionsSize,int *gpu_centroid,int *i,int pitch)
{

	extern __shared__ int sMem[];

	int     			x = (blockIdx.x*blockDim.x)+threadIdx.x;
    int     			y = (blockIdx.y*blockDim.y)+threadIdx.y;  
    int 			index = x+y*pitch;
    int 				j = 0; 
    int 	shSize_Offset = 0;
    int shCentroid_Offset = *gpu_nRegions;

	if((index >= 0) && (index < (3*(*gpu_nRegions))))
		sMem[index] = 0;
    
    if( gpu_labels[index] == index)
    {
		atomicAdd( i, 1);
		atomicAdd( gpu_regionOff + (*i), index);
    }
	__syncthreads();
	
    for( j=0; j < *gpu_nRegions; j++)
	{
			if( gpu_labels[index] == gpu_regionOff[j])
			{
						atomicAdd( sMem+shCentroid_Offset+(2*j), x);
						atomicAdd( sMem+shCentroid_Offset+(2*j)+1, y);
						atomicAdd( sMem+j, 1);
			}
	}
	__syncthreads();

	if((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		atomicAdd(gpu_centroid+(2*j), sMem[shCentroid_Offset+(2*j)]);
		atomicAdd(gpu_centroid+(2*j)+1, sMem[shCentroid_Offset+(2*j)+1]);
		atomicAdd(gpu_regionsSize+j, sMem[j]);
	}

}

		// nRegions was added later delete it..............

void calcCentroid( int *gpu_labels, unsigned char *gpu_labels_uchar, int threadsX, int threadsY, int imageW, int imageH, int nRegions, int *gpu_nRegions, int *gpu_regionOff, int *gpu_regionSize, int *gpu_centroid)
{
	int *i;
	cudaMalloc( (void **)&i, sizeof(int));
	cudaMemset( i, -1, sizeof(int));
	int shSize = (nRegions)*4*sizeof(int);
	dim3 block( 15, 16, 1);
	dim3 grid( imageW / block.x, imageH / block.y, 1);

	calcCentroidKernel<<<grid,block>>>( gpu_labels, gpu_nRegions, gpu_regionOff, gpu_regionSize, gpu_centroid, i, imageW);
	

/*	if(shSize != 0)
	{
		calcCentroidSharedKernel<<<grid,block,shSize>>>(gpu_labels,gpu_nRegions,gpu_regionOff,gpu_regionSize,gpu_centroid,i,imageW);
	}
*/
}	

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*                 									Main Wrapper about the function   								  				  */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
gpu_error_t gpu_DetectBlob( gpu_context_t *ctx)
{

	gpu_error_t err = GPU_OK;

	int *regionSize, *centroid;
//	int min_blobSize = 0;
//	int max_blobSize = 1000;

	int   imageW = ctx->width;
	int   imageH = ctx->height;
	int threadsX = ctx->threadsX;
	int threadsY = ctx->threadsY;

	fprintf(stderr,"%d %d dimensions \n",ctx->width,ctx->height);

	float elapsedtime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *gpu_nRegions, *nRegions;
	cudaMalloc( (void **)&gpu_nRegions, sizeof(int));
	nRegions = (int *)malloc(sizeof(int));
	
    int *gpu_labels;
    cudaMalloc( (void **)&gpu_labels, imageW * imageH * sizeof(int));

	err = checkCudaError();
	if( err != GPU_OK)
		return err;

    int *labels_int;
    labels_int = (int *)malloc(imageW*imageH*sizeof(int));

    //// This buffer is temporary and only used for debuggin purpose and is reponsible for diplaying the last detected blob image ////
    unsigned char *gpu_labels_uchar;
    cudaMalloc( (void **)&gpu_labels_uchar, imageW * imageH * sizeof(unsigned char));

   	err = checkCudaError();
	if( err != GPU_OK)
		return err;
	//////
   
    cudaEventRecord(start,0);

    /******************************************* Local Shared Labelling ****************************************************/
    dim3 threads(threadsX,threadsY);
    dim3 blocks( imageW/threadsX, imageH/threadsY);

    int labelSize = (threads.x + 2) * (threads.y + 2) * sizeof(int); //This is the size for storage of labels to the corresponding pixels
    int   segSize = (threads.x + 2) * (threads.y + 2) * sizeof(int); //This is the size of storage for segments.
    
	localLabelling<<< blocks, threads, (labelSize + segSize)>>>( ctx->gpu_buffer_1, gpu_labels, gpu_labels_uchar, imageW, labelSize/sizeof(int), imageW);

	err = checkCudaError();
	if( err != GPU_OK)
		return err;
	/***********************************************************************************************************************/

	
	/******************************************* Merging Blobs Together ****************************************************/

	cudaArray *src;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&src, &floatTex, imageW, imageH);
    cudaMemcpyToArray(src, 0, 0, ctx->gpu_buffer_1, imageW * imageH, cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(texSrc, src);  

	mergeBorders( gpu_labels, gpu_labels_uchar, threadsX, threadsY, imageW, imageH);

	err = checkCudaError();
	if( err != GPU_OK)
		return err;
 
    cudaUnbindTexture(texSrc);


	/****************************************** Updating all the labels (i.e flattening) ********************************************/

	flattenTrees( gpu_labels, gpu_labels_uchar, threadsX, threadsY, imageW, imageH, gpu_nRegions);
	cudaMemcpy( nRegions, gpu_nRegions, sizeof(int), cudaMemcpyDeviceToHost);
	err = checkCudaError();
	if( err != GPU_OK)
		return err;


	int *gpu_regionOff, *gpu_regionSize, *gpu_centroid;
	cudaMalloc( (void **)&gpu_regionOff, (*nRegions)*sizeof(int));
	cudaMalloc( (void **)&gpu_regionSize, (*nRegions)*sizeof(int));
	cudaMalloc( (void **)&gpu_centroid, (*nRegions)*2*sizeof(int));
	cudaMemset( gpu_regionOff, 0, (*nRegions)*sizeof(int));
	cudaMemset( gpu_regionSize, 0, (*nRegions)*sizeof(int));
	cudaMemset( gpu_centroid, -1, (*nRegions)*2*sizeof(int));
	regionSize = (int *)malloc((*nRegions)*sizeof(int));
    centroid = (int *)malloc((*nRegions)*2*sizeof(int));

	/********************************************************** Calculating Centroid *******************************************************/

	calcCentroid( gpu_labels, gpu_labels_uchar, threadsX, threadsY, imageW, imageH, *nRegions, gpu_nRegions, gpu_regionOff, gpu_regionSize, gpu_centroid);
	err = checkCudaError();
	if( err != GPU_OK)
		return err;


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fprintf(stderr,"Blobs: %lf\n\n",elapsedtime);

	
	cudaMemcpy( regionSize, gpu_regionSize, (*nRegions)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( centroid, gpu_centroid, (*nRegions)*2*sizeof(int), cudaMemcpyDeviceToHost);	
	// In Order to find a centroid just divide centroid[i] and centroid[i+1] with regionSize[i] to get X and Y respectively. 
	
	cudaMemcpy( ctx->output_buffer_1, gpu_labels_uchar, imageW*imageH, cudaMemcpyDeviceToHost);
	cudaMemcpy( labels_int, gpu_labels, imageW*imageH*sizeof(int), cudaMemcpyDeviceToHost);
	err = checkCudaError();
	if( err != GPU_OK)
		return err;
    cudaMemcpy( labels_int, gpu_labels, imageW*imageH*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(gpu_labels);
	cudaFree(gpu_labels_uchar);

	// Rest of the code is just for debugging. Remove it after measuring performance. //
/*
	FILE *file;
	file = fopen("debug_1.txt","a+"); // apend file (add text to a file or create a file if it does not exist.
	for(int i=0;i<imageW*imageH;i++)
	{
		if((i>imageW-1) && (i%imageW==0))
			fprintf(file,"\n");
		fprintf(file,"%d ", labels_int[i]); 
	}
	fprintf(file,"\n");
	fclose(file); //done!
*/

	FILE *file;
	file = fopen("regions.txt","a+"); // apend file (add text to a file or create a file if it does not exist.
	fprintf(file,"%d %f ", *nRegions, elapsedtime);
	
	for(int i=0;i<(*nRegions);i++)
	{
		fprintf(file," %d %d ", centroid[2*i],centroid[2*i]+1);
		fprintf(file,"%d ", regionSize[i]);
	}

	fprintf(file,"\n");
	fclose(file); //done!

	return err;
}
