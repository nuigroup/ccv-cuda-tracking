#ifndef _API_H
#define _API_H

#define GPU_ERROR(x) fprintf(stderr, "GPU: " #x "(%s)\n", gpu_error());
#define CHECK_CUDA_ERROR() checkCudaError();
	
typedef enum {
	GPU_MEMORY_HOST = 0,
	GPU_MEMORY_PINNED_WRITE_COMBINED,
	GPU_MEMORY_PINNED,
} gpu_context_memory_t;

typedef enum {
	GPU_OK = 0,
	GPU_ERR_MEM,
	GPU_ERR_CUDA
} gpu_error_t;

typedef enum {
	STATIC = 0,
	DYNAMIC
} bgMode;

typedef struct
{
	int height;
	int width;
	int nchannels;
	int size ;
	unsigned char *output_buffer_4;		// The only usage of output_buffer_4 is that it provides gpu_buffer_4 in initial stage.
	unsigned char *gpu_buffer_4;		// The only usage of gpu_buffer_4 is to calculate grayscale. Grayscale image is then recieved from output_buffer_1.	
	unsigned char *output_buffer_1;
	unsigned char *gpu_buffer_1;
	gpu_context_memory_t mem_flag;
} gpu_context_t;

const char *gpu_error();

gpu_error_t checkCudaError();

gpu_error_t gpu_context_create(gpu_context_t **ctx);

gpu_error_t gpu_context_init(gpu_context_t *ctx, int host_height, int host_width, int host_nchannels, gpu_context_memory_t host_flag);

gpu_error_t gpu_set_input(gpu_context_t *ctx, unsigned char *data);

gpu_error_t gpu_get_output(gpu_context_t *ctx, unsigned char **data);

void gpu_context_free(gpu_context_t *ctx);

#endif
