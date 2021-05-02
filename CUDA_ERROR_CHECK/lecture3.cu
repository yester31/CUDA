// 2020-5-13 by YH 

#include <cstdio>
#include "cuda_runtime.h"

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

int main()
{
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = { 1, 2, 3, 4, 5 }; //source data
    int b[SIZE] = { 0,0,0,0,0 }; //final destination  

    //device-side data
    int* dev_a = 0;
    int* dev_b = 0;

    // allocate device memory 
	CUDA_CHECK(cudaMalloc((void**)&dev_a, SIZE * sizeof(int)) );
    CUDA_CHECK(cudaMalloc((void**)&dev_b, SIZE * sizeof(int)) );
  
    //copy from host to device (error)
    CUDA_CHECK(cudaMemcpy(dev_a, a, SIZE * sizeof(int),cudaMemcpyDeviceToHost) );

    return 0;
}
