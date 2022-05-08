
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

void showCUDADeviceProperties(const cudaDeviceProp& dp) {
	cout << "---------------------" << endl;
	cout << "CUDA device name	: " << dp.name << endl;
	cout << "Compute capability	: " << dp.major << "." << dp.minor << endl;
	cout << "Total global memory	: " << (dp.totalGlobalMem / 1000000) << " MB" << endl;
	cout << "Shared memory per block	: " << dp.sharedMemPerBlock << endl;
	cout << "Warp size		: " << dp.warpSize << endl;
	cout << "Max threads per block	: " << dp.maxThreadsPerBlock << endl;
	cout << "Total constant memory	: " << dp.totalConstMem << endl;
	cout << "Multiprocessor count	: " << dp.multiProcessorCount << endl;
	cout << "Max 1D texture size	: " << dp.maxTexture1D << endl;
	cout << "Max threads dim		: " << dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1] << " " << dp.maxThreadsDim[2] << endl;
	cout << "Max grid size		: " << dp.maxGridSize[0] << " " << dp.maxGridSize[1] << " " << dp.maxGridSize[2] << endl;
	cout << "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: " << dp.canMapHostMemory << endl;
	cout << "---------------------" << endl;
}

int main()
{
	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, 0);

	showCUDADeviceProperties(dp);

    return 0;
}
