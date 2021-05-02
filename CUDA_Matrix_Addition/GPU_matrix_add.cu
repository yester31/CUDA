// 2020-5-16 by YH 

#include "util_cuda.cuh"

//kernel program for the device (GPU): compiled by NVCC
__global__ void addKernel(int*c, const int* a, const int * b) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * (blockDim.x) + x; //[y][x] = y * WIDTH + x
	c[i] = a[i] + b[i];
}


int main(void) {
	//host-side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };

	// make a, b matirces
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			a[y][x] = y * 10 + x;
			b[y][x] = (y * 10 + x) * 100;
		}
	}

	//device-side data
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	// allocate device memory
	CUDA_CHECK(cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int)));

	//copy from host to device 
	CUDA_CHECK(cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));//dev_a=a;
	CUDA_CHECK(cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));//dev_b=b;

	//launch a kernel on the GPU with one thread for each element.
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(WIDTH, WIDTH, 1);//x,y,z
	addKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b);
	CUDA_CHECK(cudaPeekAtLastError());

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(c,  dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost));//c=dev_c;

	//free device memory
	CUDA_CHECK(cudaFree(dev_c));
	CUDA_CHECK(cudaFree(dev_a));
	CUDA_CHECK(cudaFree(dev_b));

	//print the result 
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			printf("%5d", c[y][x]);
		}printf("\n");
	}

	return 0;
}