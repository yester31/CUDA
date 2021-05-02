#include <cstdio>

void add(const int x, const int y,
	const int WIDTH, int*c, const int*a,const int*b) {
	int i = y * (WIDTH)+x;
	c[i] = a[i] + b[i];
}

int main(void) {

	//host-side data
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = {0};

	// make a, b matirces
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			a[y][x] = y * 10 + x;
			b[y][x] = (y * 10 + x) * 100;
		}
	}

	// calculate_method_1
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			c[y][x] = a[y][x] + b[y][x];
		}
	}

	// calculate_method_2
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			add(x, y, WIDTH, (int*)c, (int*)a, (int*)b);
		}
	}

	//print the result 
	for (int y = 0; y < WIDTH; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			printf("%5d", c[y][x]);
		}printf("\n");
	}

	//done
	return 0;
}