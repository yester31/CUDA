// 2020-5-28 by YH

#include "../util_cu/util_cuda.cuh"

__device__ int my_global_variable;
__constant__ int my_constant_variable = 13;

__global__ void foo() {
	__shared__ int my_shared_variable;

	int* ptr_to_global = &my_global_variable;
	const int* ptr_to_constant = &my_constant_variable;
	int* ptr_to_shared = &my_shared_variable;

	*ptr_to_global = *ptr_to_shared;

	__shared__ int* ptr; // shared 영역에 ptr 포인터 변수가 저장된다.(shared 영역을 가리킨다는 의미가 아님) 
}

// gpu에서 포인터의 포인터는 자제...복잡하고 잘 작동안함...
// 