#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 128

__global__ void kernelAdd(float* a, float* b, float* c, int n)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < n) {
		c[x] = b[x] + a[x];
	}
}

int main()
{
	std::cout << "Starting..." << std::endl;

	float *a, *b, *c,*d;
	int n = 67107850;
	
	a = (float*) malloc(sizeof(float) * n);
	b = (float*) malloc(sizeof(float) * n);
	c = (float*) malloc(sizeof(float) * n);
	d = (float*) malloc(sizeof(float) * n);
	
	float *d_a, *d_b, *d_c;
	
	cudaMalloc(&d_a, (size_t) sizeof(float) * n);
	cudaMalloc(&d_b, (size_t) sizeof(float) * n);
	cudaMalloc(&d_c, (size_t) sizeof(float) * n);

	for (int i = 0; i < n; i++) {
		a[i] = i;
		b[i] = n - i;
	}
	
	dim3 grid;
	dim3 block;
	grid.x = (n / BLOCK_SIZE) + 1;
	block.x = BLOCK_SIZE;
	
	cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

	cudaError_t error;	
	error = cudaPeekAtLastError();
	if (error) {
		std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
	}

	//_____________________________________________________________
	
	kernelAdd<<<grid, block>>>(d_a, d_b, d_c, n);
	//_____________________________________________________________
	
	cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
	
	error = cudaPeekAtLastError();
	if (error) {
		std::cout << "CUDA error: " << cudaGetErrorString(error)  << std::endl;
	}

	for (int i=0; i < n;i++) {
		d[i] = a[i] + b[i];
		if (d[i] != c[i]) {
			std::cout << "Incorrect result" << std::endl;
			return 0;
		}	
	}
	std::cout << "Sucessfully finished" << std::endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);
	free(d);
}