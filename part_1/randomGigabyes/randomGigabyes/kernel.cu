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

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void generateRandomCuda(float* localMemRandom, unsigned int arraySize, unsigned int memsize);
void generateRandomCudaHost(float* localMemRandom, unsigned int arraySize, unsigned int memsize);
// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

/*
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
*/

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

int main()
{
	float* localMemRandom;
	unsigned int arraySize = 1024 * 1024 * 500;
	size_t memsize = sizeof(float) * arraySize;
	localMemRandom = (float*)malloc(memsize);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i = 0; i < 20; i++) {
		printf("%f ", localMemRandom[i]);
	}
	printf("\n");

	// Generate random and copy to local memory.
	generateRandomCuda(localMemRandom, arraySize, memsize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "---------------------" << endl;
	cout << "random generation time for size " << (int) (memsize / 1024 / 1024)  << "MB on GPU: " << elapsedTime << " milliseconds" << endl;
	cout << "---------------------" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	for (int i = 0; i < 20; i++) {
		printf("%f ", localMemRandom[i]);
	}
	printf("\n");

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

void generateRandomCuda(float* localMemRandom, unsigned int arraySize, unsigned int memsize) {
	float* deviceMemRandom;
	cudaMalloc((void**)&deviceMemRandom, memsize);

	checkCUDAError("memory allocation");

	curandGenerator_t randomGenerator;
	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("random generator creation");

	curandSetPseudoRandomGeneratorSeed(randomGenerator, time(NULL));
	checkCUDAError("seeding random generator");

	curandGenerateUniform(randomGenerator, deviceMemRandom, arraySize);
	checkCUDAError("random generation process");

	// Copy result back to host
	cudaMemcpy(localMemRandom, deviceMemRandom, memsize, cudaMemcpyDeviceToHost);
	checkCUDAError("copy from device to host");

	curandDestroyGenerator(randomGenerator);
	checkCUDAError("freeing generator memory");
}

/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/