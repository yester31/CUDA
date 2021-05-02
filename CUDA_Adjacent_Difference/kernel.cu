// 2020-5-24 by YH 

#include "../util_cu/util_cuda.cuh"
#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <io.h> // for open(), write(), close() in WIN32
#include <fcntl.h> // for open(), write()
#include <sys/stat.h> 
#include <windows.h> // for high-resolution performance counter

#define GRIDSIZE (8 * 1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE) // 32M byte needed!

// CUDA global mem
__global__ void adj_diff_navie(float* result, float* input) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0) {
		float x_i = input[i];
		float x_i_minus_1 = input[i - 1];
		result[i] = x_i - x_i_minus_1;
	}
}

//random data generation
void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;// 0.0 ~ 1.0
	}
}

int main(void) {

	float* pSource = NULL;
	float* pResult = NULL;
	int i;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	//malloc memories on the host-side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));

	//generate source data
	genData(pSource, TOTALSIZE);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch

	// CUDA : allocate device memory
	float* pSourceDev = NULL;
	float* pResultDev = NULL;
	cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float));
	// CUDA : copy from host to device
	cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);
	// CUDA : launch the kernel: result[i] = source[i] - source[i-1];
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	adj_diff_navie <<<dimGrid, dimBlock >>> (pResultDev, pSourceDev);
	// CUDA : copy from device to host
	cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
	
	//end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));//end the stop watch

	printf("elapsed time = %f usec\n", (double)(cntEnd - cntStart)*1000000.0 / (double)(freq));

	// print sample cases
	i = 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE - 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE / 2;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);

	// free the memory
	free(pSource);
	free(pResult);
}