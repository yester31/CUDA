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

	__shared__ int* ptr; // shared ������ ptr ������ ������ ����ȴ�.(shared ������ ����Ų�ٴ� �ǹ̰� �ƴ�) 
}

// gpu���� �������� �����ʹ� ����...�����ϰ� �� �۵�����...
// 