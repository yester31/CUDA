//Windows QueryPerformanceCounter ���� ��Ȯ�� �ð� ���� �Լ� �̴�.

#include <windows.h>
#include <winbase.h>
#include <cstdio>

void main() {
	LARGE_INTEGER start, end, f;

	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&start);
	// action
	printf("hi");

	QueryPerformanceCounter(&end);

	__int64 ms_interval = (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000);
	__int64 micro_interval = (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000000);

	printf("millisecond : %d, microsecond : %d\n", (int)ms_interval, (int)micro_interval);
}