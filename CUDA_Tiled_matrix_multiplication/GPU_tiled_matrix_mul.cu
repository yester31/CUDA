// 2020-5-17 by YH 

#include "../util_cu/util_cuda.cuh"
 
//kernel program for the device (GPU): compiled by NVCC
__global__ void mulKernel(int*c, const int* a, const int * b, const int WIDTH) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0.0f;
	for (int k = 0; k < WIDTH; ++k) {
		float lhs = a[y * WIDTH + k];
		float rhs = b[k * WIDTH + x];
		sum += lhs * rhs;
	}
	c[y * WIDTH + x] = sum;
}


int main(void) {
	//host-side data
	const int WIDTH = 8;
	const int TILE_WIDTH = 4;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };

	// make a, b matirces
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			a[y][x] = y + x;
			b[y][x] = y + x;
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
	dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);//x,y,z
	mulKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, WIDTH);
	CUDA_CHECK(cudaPeekAtLastError());

	//copy from device to host
	CUDA_CHECK(cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost));//c=dev_c;

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