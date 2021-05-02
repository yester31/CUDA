// 2020-5-31 by YH

#include "../util_cu/util_cuda.cuh"
#include <cstdio>
#include <stdlib.h> // for rand(), malloc(), free()
#include <windows.h> // for high-resolution performance counter
#include <chrono>

using namespace std;
using namespace chrono;

const int WIDTH = 1024;							// total width is 1024* 1024
const int TILE_WIDTH = 32;						// block will be (TILE_WIDTH, TILE_WIDTH)
const int GRID_WIDTH = (WIDTH / TILE_WIDTH);	// grid will be (GRID_WIDTH, GRID_WIDTH)

// CUDA shared mem
__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
	//c[y][x] = sum_ka[y][k] * b[k][x]
	//c[y * WIDTH + x] = sum_ka[y * WIDTH + k] * b[k * WIDTH + x]
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int by = blockIdx.y; int bx = blockIdx.x;
	int ty = threadIdx.y; int tx = threadIdx.x;
	int gy = by * TILE_WIDTH + ty; // global y index
	int gx = bx * TILE_WIDTH + tx; // global x index
	float sum = 0.0F;
	for (register int m = 0; m < width / TILE_WIDTH; ++m) {
		//read into the shared memory blocks
		s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
		s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty)* width + gx];
		__syncthreads();
		//use the shared memory blocks to get the partial sum
		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[gy * width + gx] = sum;

}


//random data generation
void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;// 0.0 ~ 1.0
	}
}

int main(void) {

	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;

	//malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));

	//generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);

	//start the timer
	uint64_t total_time = 0;

	// CUDA : allocate device memory
	float* pADev = NULL;
	float* pBDev = NULL;
	float* pCDev = NULL;

	cudaMalloc((void**)&pADev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pBDev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pCDev, WIDTH * WIDTH * sizeof(float));

	// CUDA : copy from host to device
	cudaMemcpy(pADev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pBDev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	// CUDA : launch the kernel: result[i] = source[i] - source[i-1];
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	//adj_diff_shared << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);
	//adj_diff_shared2 << <dimGrid, dimBlock, BLOCKSIZE * sizeof(float) >> > (pResultDev, pSourceDev);	
	
	uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	//커널에서 사용하는 shared memory 크기 전달 
	matmul << <dimGrid, dimBlock >> > (pCDev, pADev, pBDev, WIDTH);

	//end the timer
	total_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
	printf("dur_time(chrono) = %6.3f [msec] \n", total_time / 1000.f);

	// CUDA : copy from device to host
	cudaMemcpy(pC, pCDev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

	// print sample cases
	int i, j;
	i = 0; j = 0;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH/2; j = WIDTH/2;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH-1; j = WIDTH-1;
	printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);


	// free the memory
	free(pA);
	free(pB);
	free(pC);

	cudaFree(pADev);
	cudaFree(pBDev);
	cudaFree(pCDev);

	return 0;
}
