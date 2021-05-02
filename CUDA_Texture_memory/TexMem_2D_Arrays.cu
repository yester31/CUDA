#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <stdio.h>

#define W 256
#define H 256

texture<float, 2, cudaReadModeElementType> tex;

__global__ void kernel()
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float v = tex2D (tex, x, y);
}

int main() {
	float* buffer;
	cudaMalloc(&buffer, W * H * sizeof(float));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(0, tex, buffer, desc, W, H, W * sizeof(float));

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(W, H, 1);
	kernel << <dimGrid, dimBlock >> > ();
	cudaUnbindTexture(tex);
	cudaFree(buffer);



}

