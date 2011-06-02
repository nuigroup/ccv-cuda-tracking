#ifndef _API_H
#define _API_H

typedef struct 
{
	int height;
	int width;
	int nchannels;
	int size ;
	unsigned char *output_buffer;	
	unsigned char *gpu_buffer;
	// 0 for pageable memory, 1 for pinned memory with write combined (not recommended for host reading), 2 for pinned memory (can be read by host frequently)
	int mem_flag;	  
}gpu_context;

typedef enum{ No_error, Memory_Allocation_error, Cuda_error}gpu_error_t;

__global__ void cuda_set_input(gpu_context *ctx, unsigned char *idata);

__global__ void cuda_get_output(gpu_context *ctx)
		
gpu_error_t checkCudaError();

gpu_error_t gpu_context_create( gpu_context *ctx );

gpu_error_t gpu_context_init( gpu_context *ctx, int host_height, int host_width, int host_nchannels, int host_flag);

gpu_error_t gpu_set_input( gpu_context *ctx, unsigned char *data);

gpu_error_t gpu_get_input( gpu_context *ctx, unsigned char *odata);

void gpu_context_free( gpu_context *ctx);

#endif
