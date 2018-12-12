#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define GRIDSIZE	(32 * 1024)
#define BLOCKSIZE	1024
#define TOTALSIZE	(GRIDSIZE * BLOCKSIZE)

void genData(unsigned* ptr, unsigned size) {
	while (size--) {
		*ptr++ = (unsigned)(rand() % 10000);
	}
}

__global__ void kernel(unsigned* pData, unsigned* pAnswer, unsigned target) {
	// each thread loads multiple element from global to shared memory
	register unsigned tid = threadIdx.x;
	if (tid == 0) {
		*pAnswer = TOTALSIZE;
	}
	__syncthreads();
	register unsigned first = tid * GRIDSIZE;
	register unsigned last = (tid + 1) * GRIDSIZE;
	if (pData[first] <= target && target <= pData[last - 1]) {
		while (first < last) {
			register unsigned mid = (first + last) / 2;
			if (target == pData[mid]) {
				atomicMin(pAnswer, mid);
				last = first;
			} else if (target < pData[mid]) {
				last = mid - 1;
			} else {
				first = mid + 1;
			}
		}
	}
}


int main(void) {
	unsigned* pData = NULL;
	unsigned answer;
	// prepare timer 
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// malloc memories on the host-side
	pData = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));
	// generate source data
	genData(pData, TOTALSIZE);
	printf("search on %d data\n", TOTALSIZE);
	std::sort(pData, pData + TOTALSIZE);
	// CUDA: allocate device memory
	unsigned* pDataDev;
	unsigned* pAnswerDev;
	cudaMalloc((void**)&pDataDev, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnswerDev, 4 * sizeof(unsigned));
	// CUDA: copy from host to device
	cudaMemcpy(pDataDev, pData, TOTALSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
	// start timer
	cudaEventRecord(start, 0);
	// CUDA: launch the kernel
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	kernel<<<dimGrid, dimBlock>>>(pDataDev, pAnswerDev, 5000U);
	// end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elapsed time = %f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// CUDA: copy from device to host
	cudaMemcpy(&answer, pAnswerDev, sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("index = %u, value = %u\n", answer, pData[answer]);
	// CUDA: free the memory
	cudaFree(pDataDev);
	cudaFree(pAnswerDev);
	// free the memory
	free(pData);
}
