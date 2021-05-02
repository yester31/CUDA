// 2020-5-13 by YH 

#include <cstdio>
#include "cuda_runtime.h"

int main()
{
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = { 1, 2, 3, 4, 5 }; //source data
    int b[SIZE] = { 0,0,0,0,0 }; //final destination  

    //print source
    printf("Before {%d,%d,%d,%d,%d}\n", b[0], b[1], b[2], b[3], b[4]);

    //device-side data
    int* dev_a = 0;
    int* dev_b = 0;

    // allocate device memory
    cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE * sizeof(int));

    //copy from host to device
    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //copy from device to device
    cudaMemcpy(dev_b, dev_a, SIZE * sizeof(int), cudaMemcpyDeviceToDevice);

    //copy from device to host
    cudaMemcpy(b, dev_b, SIZE * sizeof(int), cudaMemcpyHostToHost);

    //free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    //print source
    printf("After {%d,%d,%d,%d,%d}\n", b[0], b[1], b[2], b[3], b[4]);

    return 0;
}
