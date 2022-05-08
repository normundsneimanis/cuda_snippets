#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <iostream>

using namespace std;

void generateRandomCudaHost(float* localMemRandom, unsigned int arraySize, unsigned int memsize);

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void add(int* a, int* b, int* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < n)
		c[index] = a[index] + b[index];
}

int main()
{
	float* localMemRandom;
	unsigned int arraySize = 1024 * 1024 * 100;
	size_t memsize = sizeof(float) * arraySize;
	localMemRandom = (float*)malloc(memsize);

	// Generate random and copy to local memory.
	generateRandomCudaHost(localMemRandom, arraySize, memsize);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Reduce elements to average

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "---------------------" << endl;
	cout << "random generation time for size " << (memsize / 1024 / 1024) << "MB  on GPU: " << elapsedTime << " milliseconds" << endl;
	cout << "---------------------" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();
	checkCUDAError("cuda device reset");

	return 0;
}


void generateRandomCudaHost(float* localMemRandom, unsigned int arraySize, unsigned int memsize) {

	checkCUDAError("memory allocation");

	curandGenerator_t randomGenerator;
	curandCreateGeneratorHost(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("random generator creation");

	curandSetPseudoRandomGeneratorSeed(randomGenerator, time(NULL));
	checkCUDAError("seeding random generator");

	curandGenerateUniform(randomGenerator, localMemRandom, arraySize);
	checkCUDAError("random generation process");

	curandDestroyGenerator(randomGenerator);
	checkCUDAError("freeing generator memory");
}