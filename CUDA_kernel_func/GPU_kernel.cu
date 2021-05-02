// 2020-5-14 by YH 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#if defined(NDEBUG)     //release mode
#define CUDA_CHECK(x) (x)   
#else                   // debug mode
//error check 
#define CUDA_CHECK(x)   do{\
    (x); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("cuda failure %s at %s:%d \n", \
        cudaGetErrorString(e), \
            __FILE__, __LINE__); \
        exit(0); \
    } \
}while(0)
#endif

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x; // each thread knows its own index
	c[i] = a[i] + b[i];
}

int main(void) {

	const int size = 5;
	const int a[size] = { 1, 2, 3, 4, 5 };
	const int b[size] = { 10, 20, 30, 40, 50 };
	int c[size] = { 0 };

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	CUDA_CHECK(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, size * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

	addKernel << <1, size >> > (dev_c, dev_a, dev_b);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dev_c));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
}