#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <io.h> // for open(), write(), close() in WIN32
#include <fcntl.h> // for open(), write()
#include <sys/stat.h> 
#include <windows.h> // for high-resolution performance counter

#define GRIDSIZE (8 * 1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE) // 32M byte needed!

//random data generation
void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;// 0.0 ~ 1.0
	}
}

//perform the action : result[i] = source[i] - source[i-1];
void getDiff(float* dst, const float* src, unsigned int size) {
	for (register int i = 1; i < size; ++i) {
		dst[i] = src[i] - src[i-1];
	}
}

int main(void){

	float* pSource = NULL;
	float* pResult = NULL;
	int i;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	//malloc memories on the host-side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));

	//generate source data
	genData(pSource, TOTALSIZE);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));

	//perform the action : result[i] = source[i] - source[i-1];
	pResult[0] = 0.0F; // exceptional case
	getDiff(pResult, pSource, TOTALSIZE);
	
	//end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));

	printf("elapsed time = %f usec\n", (double)(cntEnd - cntStart)*1000000.0 / (double)(freq));

	// print sample cases
	i = 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE - 1;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	i = TOTALSIZE / 2;
	printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	
	// free the memory
	free(pSource);
	free(pResult);
}