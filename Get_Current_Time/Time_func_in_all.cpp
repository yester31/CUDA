#include <chrono>
#include <stdio.h>
#include <windows.h>

using namespace chrono;
using namespace std;

int main(void) {
	system_clock::time_point start = system_clock::now();

	Sleep(2000);

	system_clock::time_point end = system_clock::now();
	nanoseconds du = end - start;
	printf("%lld nano-seconds\n", du);
	return 0;
}