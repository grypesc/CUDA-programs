#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <windows.h> // for high-resolution performance counter

#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (cudaSuccess != e) { \
			printf("cuda failure \"%s\" at %s:%d\n", \
				cudaGetErrorString(e), \
			     __FILE__, __LINE__); \
			exit(1); \
		} \
	} while (0)
#endif

#define WIDTH		(1 * 1024)	// total width is 1024*1024
#define	TILE_WIDTH	8		// block will be (TILE_WIDTH,TILEWIDTH)
#define	GRID_WIDTH	(WIDTH / TILE_WIDTH)	// grid will be (GRID_WDITH,GRID_WDITH)


void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}


__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
    // c[y][x] = sum_k a[y][k] * b[k][x]
    // c[y * WIDTH + x] = sum_k a[y*WIDTH + k] * b[k*WIDTH + x]
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int gy = by * TILE_WIDTH + ty; // global y index
    int gx = bx * TILE_WIDTH + tx; // global x index
    float sum = 0.0F;
    for (register int m = 0; m < width / TILE_WIDTH; ++m) {
		// read into the shared memory blocks
        s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
        s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width + gx];
        __syncthreads();
		// use the shared memory blocks to get the partial sum
        for (register int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    g_C[gy * width + gx] = sum;
}


int main(void) {
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	// malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	// generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);
	// CUDA: allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;
	CUDA_CHECK( cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float)) );
	CUDA_CHECK( cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float)) );
	// CUDA: copy from host to device
	CUDA_CHECK( cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
	// start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul <<< dimGrid, dimBlock>>>(pCdev, pAdev, pBdev, WIDTH);
	// end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	CUDA_CHECK( cudaPeekAtLastError() );
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));
	// CUDA: copy from device to host
	CUDA_CHECK( cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
	// print sample cases
	int i, j;
	i = 0; j = 0; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	// CUDA: free the memory
	CUDA_CHECK( cudaFree(pAdev) );
	CUDA_CHECK( cudaFree(pBdev) );
	CUDA_CHECK( cudaFree(pCdev) );
	// free the memory
	free(pA);
	free(pB);
	free(pC);
}

