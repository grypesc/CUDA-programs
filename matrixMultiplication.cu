#include <cstdio>

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


// kernel program for the device (GPU): compiled by NVCC
__global__ void mulKernel(int* c, const int* a, const int* b, const int WIDTH) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * WIDTH + x; // [y][x] = y * WIDTH + x;
	int sum = 0;
	for (int k = 0; k < WIDTH; ++k) {
		sum += a[y * WIDTH + k] * b[k * WIDTH + x];
	}
	c[i] = sum;
}


// main program for the CPU: compiled by MS-VC++
int main(void) {
	// host-side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };
	// make a, b matrices
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			a[y][x] = y + x;
			b[y][x] = y + x;
		}
	}
	// device-side data
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	// allocate device memory
	CUDA_CHECK( cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int)) );
	// copy from host to device
	CUDA_CHECK( cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice) );
	// launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(WIDTH, WIDTH, 1); // x, y, z
	mulKernel <<< 1, dimBlock>>>(dev_c, dev_a, dev_b, WIDTH);
	CUDA_CHECK( cudaPeekAtLastError() );
	// copy from device to host
	CUDA_CHECK( cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost) );
	// free device memory
	CUDA_CHECK( cudaFree(dev_c) );
	CUDA_CHECK( cudaFree(dev_a) );
	CUDA_CHECK( cudaFree(dev_b) );
	// print the result
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			printf("%5d", c[y][x]);
		}
		printf("\n");
	}
	// done
	return 0;
}
