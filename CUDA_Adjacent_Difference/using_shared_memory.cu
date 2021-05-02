// 2020-5-24 by YH
// 2020-5-28 re_version by YH
// 2020-5-30 re_re_version by YH
// 2020-5-31 re_re_re_version by YH


// cpu가 계산 속도가 더 빠름 원인 찾지 못함... 
// -> 데이터 전송 제외하고 단순하게 *계산 함수 비교시* gpu가 대략 500백 정도 빠름 ... 

#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <io.h> // for open(), write(), close() in WIN32
#include <fcntl.h> // for open(), write()
#include <sys/stat.h> 
#include <windows.h> // for high-resolution performance counter
#include "../util_cu/util_cuda.cuh"

#include <chrono>
using namespace std;
using namespace chrono;

#define GRIDSIZE (8 * 1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE) // 32M byte needed!

// CUDA shared mem
__global__ void adj_diff_shared(float* result, float* input) { 
	__shared__ float s_data[BLOCKSIZE];
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[tx] = input[i];
	__syncthreads();
	if (tx > 0) {
		result[i] = s_data[tx] - s_data[tx - 1];
	}else if (i > 0) {
		result[i] = s_data[tx] - input[i - 1];
	}
}

// with dynamic allocation 공유메모리 동적 할당 
__global__ void adj_diff_shared2(float* result, float* input) {
	extern __shared__ float s_data[];//no size declaration!
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[tx] = input[i];
	__syncthreads();
	if (tx > 0) {
		result[i] = s_data[tx] - s_data[tx - 1];
	}
	else if (i > 0) {
		result[i] = s_data[tx] - input[i - 1];
	}
}

// overuse the syncthreads()
__global__ void adj_diff_shared3(float* result, float* input) {
	__shared__ float s_data[BLOCKSIZE];
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	register float answer;
	s_data[tx] = input[i];
	__syncthreads();
	if (tx > 0) {
		answer = s_data[tx] - s_data[tx - 1];
	}
	else if (i > 0) {
		answer = s_data[tx] - input[i - 1];
	}
	__syncthreads();
	result[i] = answer;
}


//perform the action : result[i] = source[i] - source[i-1];
void getDiff(float* dst, const float* src, unsigned int size) {
	for (register int i = 1; i < size; ++i) {
		dst[i] = src[i] - src[i - 1];
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
	float* pResult_cpu = NULL;

	int i;
	long long cntStart, cntEnd, cntStart2, cntEnd2, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	//malloc memories on the host-side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult_cpu = (float*)malloc(TOTALSIZE * sizeof(float));

	//generate source data
	genData(pSource, TOTALSIZE);


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
	//adj_diff_shared << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);
	//adj_diff_shared2 << <dimGrid, dimBlock, BLOCKSIZE * sizeof(float) >> > (pResultDev, pSourceDev);											//커널에서 사용하는 shared memory 크기 전달 
	//start the timer
	uint64_t total_time = 0;
	uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch

	adj_diff_shared3 << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);
	// CUDA : copy from device to host
	//end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));//end the stop watch
	total_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;

	printf("(gpu) elapsed time     = %6.3f [msec] \n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));
	printf("(gpu) dur_time(chrono) = %6.3f [msec] \n", total_time / 1000.f);

	cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);


	uint64_t total_time2 = 0;
	uint64_t start_time2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart2)); // start the stop watch

	//perform the action : result[i] = source[i] - source[i-1];
	pResult_cpu[0] = 0.0F; // exceptional case
	getDiff(pResult_cpu, pSource, TOTALSIZE);
	
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd2));//end the stop watch
	total_time2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time2;

	printf("(cpu) elapsed time     = %6.3f [msec] \n", (double)(cntEnd2 - cntStart2) * 1000.0 / (double)(freq));
	printf("(cpu) dur_time(chrono) = %6.3f [msec] \n\n", total_time2 / 1000.f);
	printf("msec = microsecond, 10E-6s\n\n");


	// print sample cases
	i = 1;
	printf("(gpu) i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	printf("(cpu) i=%2d: %f = %f - %f \n", i, pResult_cpu[i], pSource[i], pSource[i - 1]);

	i = TOTALSIZE - 1;
	printf("(gpu) i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	printf("(cpu) i=%2d: %f = %f - %f\n", i, pResult_cpu[i], pSource[i], pSource[i - 1]);

	i = TOTALSIZE / 2;
	printf("(gpu) i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	printf("(cpu) i=%2d: %f = %f - %f\n", i, pResult_cpu[i], pSource[i], pSource[i - 1]);




	// free the memory
	free(pSource);
	free(pResult);
	free(pResult_cpu);

	cudaFree(pSourceDev);
	cudaFree(pResultDev);

}
