/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include <iostream>
#include <ctime>

#include "bitmap_image.h"
#include "median_filter_CUDA.h"
#include "median_filter_CPU.h"

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
	cout << "---------------------" << endl;
}

int main( int argc, char* argv[]) {
	cout << "*** Salt-and-pepper noise filtering on GPU ***" << endl 
		 << "***    Solution by Arnis Lektauers 2014    ***" << endl << endl;
		 
	if (argc < 3) {
		cerr << "Usage: input.bmp output.bmp [--noise | --cpu]" << endl;
		cerr << "--noise - generate output image with salt-and-pepper noise" << endl;
		cerr << "--cpu - use CPU based filtering" << endl;
        return 1;
    }
	
	bool computeOnGPU = true;
	bool generateNoiseImage = false;

	if (argc > 3) {
		if (std::string(argv[3]) == "--noise") {
			generateNoiseImage = true;
		} else if (std::string(argv[3]) == "--cpu") {
			computeOnGPU = false;
		}
	}

	clock_t begin = clock();
	clock_t end;
	float elapsedTime;
	
	if (computeOnGPU) {
		cudaFree(0); // force lazy context establishment in the CUDA runtime
	
		end = clock();
		elapsedTime = float(end - begin) / CLOCKS_PER_SEC;
		cout << "CUDA initialized in " << elapsedTime << " seconds" << endl << endl;

		begin = clock();
	}

	BitmapImage inputImage;
	if (!inputImage.loadFromFile(argv[1])) {
		return 1;
	};
	cout << "Loaded image '" << argv[1] << "' (size: " << inputImage.getWidth() << "x" << inputImage.getHeight() << ")" << endl << endl;
	
	BitmapImage outputImage(inputImage, generateNoiseImage);

	if (generateNoiseImage) {
		outputImage.generateSaltAndPepperNoise();
	} else {					
		if (computeOnGPU) {
			cudaDeviceProp dp;
			cudaGetDeviceProperties(&dp, 0);

			showCUDADeviceProperties(dp);
			
			cudaError_t cudaStatus = medianFilterGrayscaleCUDA(inputImage, outputImage, dp.maxThreadsPerBlock);
			if (cudaStatus != cudaSuccess) {
				cerr << "CUDA median filter failed!";
				return 1;
			}

			// cudaDeviceReset must be called before exiting in order for profiling and
			// tracing tools such as Nsight and Visual Profiler to show complete traces.
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				cerr << "cudaDeviceReset failed!";
				return 1;
			}
		} else {
			medianFilterGrayscaleCPU(inputImage, outputImage);
		}
	}

	outputImage.saveToFile(argv[2]);

	end = clock();
	elapsedTime = float(end - begin) / CLOCKS_PER_SEC;
	cout << "Total processing time: " << elapsedTime << " seconds" << endl;
	
	return 0;
}