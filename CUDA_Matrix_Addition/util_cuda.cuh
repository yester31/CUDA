#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

// ERROR CHECK

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