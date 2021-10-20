// 2020-5-19 by YH 

#include "../util_cu/util_cuda.cuh"

#define BLOCK_SIZE 10

__global__ void kernelFunc(int* g_in) {
	__shared__ int s_data[BLOCK_SIZE];
	s_data[threadIdx.x] = g_in[threadIdx.x];
	//synchronize the local threads writing to the local memory cache
	__syncthreads();// �տ� �ִ� ��� ������ �۾��� �Ϸ� Ȯ��.  
	// all data available for all threads in the block

	// ... action ...
	// every thread can use the shard data
}
// __syncthreads(); �� �ʿ� �Ҷ� �� ����ϴ°��� ���� (�ӵ� ���鿡�� ������)

// Example: using global variables

//compute result[i] = input[i] - input[i-1]
__global__ void adj_diff_navie(int* g_result, int* g_input) {
	//compute this thread's global index
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > 0) {
		//each thread loads two elements from global memory
		int x_i = g_input[i];
		int x_i_minus_1 = g_input[i - 1];
		g_result[i] = x_i - x_i_minus_1;
	}	
}

__global__ void adj_diff(int* g_result, int* g_input) {
	int tx = threadIdx.x; // shorthands for threadsIdx.x
	//allocate a __shared__ array, one element per thread
	__shared__ int s_data[BLOCK_SIZE];
	//each thread reads one elements to s_data
	unsigned int i = blockIdx.x * blockDim.x + tx;
	s_data[tx] = g_input[i];
	//avoid race condition : ensure all loads complete before continuing
	__syncthreads();
	//now action

	if (tx > 0) {
		g_result[i] = s_data[tx] - s_data[tx-1];
	}
	else if(i > 0){//tx�� 0�̰� �۷ι� �ε����� 0�� �ƴ� �� ���
		//handle thread block boundary (for tx == 0 case)
		g_result[i] = s_data[tx] - s_data[i - 1];
	}
}

// ������ �Ҷ� shard �޸� ũ�⸦ �𸦶� (shared memory �����Ҵ�)
__global__ void adj(int* result, int* input) {
	// use extern to indicate a __shared__ array will be
	// allocated dynamically at kernel launch time
	extern __shared__ int s_data[];
	// ...
}
// pass the size of the per-block array, in bytes, as the third
// argument to the triple chevrons
adj <<<num_blocks, block_size, block_size * sizeof(int)>>>(r,i);// block ������ ��ŭ int �������� �迭�� ������.
