
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <stdio.h>

#define N 1024
texture<float, 1, cudaReadModeElementType> tex;

__global__ void kernel_tex1Dfetch()
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = tex1Dfetch(tex, i);
}

int main() {
	float* buffer;
	cudaMalloc(&buffer, N * sizeof(float));
	cudaBindTexture(0, tex, buffer, N * sizeof(float));

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(N, 1, 1);
	kernel_tex1Dfetch << <dimGrid, dimBlock >> > ();
	cudaUnbindTexture(tex);
	cudaFree(buffer);



}

