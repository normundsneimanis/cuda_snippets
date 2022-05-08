/*
 * Harris corners detector algorithm on GPU 
 * Arnis Lektauers 2019 Riga Technical University
 */
#include <iostream>
#include <ctime>

#include "bitmap_image.h"
#include "harris_detector_common.h"
#include "harris_detector_CUDA.h"
#include "harris_detector_CPU.h"

using namespace std;

void showCUDADeviceProperties(int deviceNum, const cudaDeviceProp& dp) {
	std::cout << "---------------------------" << std::endl;
	cout << "CUDA device : " << deviceNum << endl;
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
	std::cout << "---------------------------" << std::endl;
}

void makeResultComparisonReport(const BitmapImage& inputImage, const BYTE* cornersMapGPU, const BYTE* cornersMapCPU, const std::string &outputFileName) {
	BitmapImage outputImage(inputImage);
	outputImage.copy32bpp(inputImage);
		
	for (int y = 0; y < outputImage.getHeight(); y++) {
		for (int x = 0; x < outputImage.getWidth(); x++) {
			const int p = x + y * outputImage.getWidth();

			RGBA color = RGBA::Black;
			if (cornersMapCPU[p] && cornersMapGPU[p]) {
				color = RGBA::Green;
			} else if (cornersMapGPU[p]) {
				color = RGBA::Red;
			} else if (cornersMapCPU[p]) {
				color = RGBA::Blue;
			}

			if (color != RGBA::Black) {
				drawRectangleMarker(outputImage.getRawData(), x, y, color, 5, outputImage.getWidth(), outputImage.getHeight());
			}
		}
	}
	
	outputImage.saveToFile(outputFileName);
}

void makeOutputImage(const BitmapImage& inputImage, const BYTE* cornersMap, const std::string &outputFileName, const RGBA& color) {
	BitmapImage outputImage(inputImage);
	outputImage.copy32bpp(inputImage);

	for (int y = 0; y < outputImage.getHeight(); y++) {
		for (int x = 0; x < outputImage.getWidth(); x++) {
			if (cornersMap[x + y * outputImage.getWidth()]) {
				drawRectangleMarker(outputImage.getRawData(), x, y, color, 5, outputImage.getWidth(), outputImage.getHeight());
			}
		}
	}
	
	outputImage.saveToFile(outputFileName);
}

int main( int argc, char* argv[]) {
	cout << "*** Harris corners detector algorithm on GPU ***" << endl 
		 << "***    Solution by Arnis Lektauers 2016    ***" << endl << endl;
		 
	if (argc < 3) {
		cerr << "Usage: input.bmp threshold [--cpu | --gpu]" << endl;
		cerr << "threshold - threshold value" << endl;
		cerr << "--cpu - use only CPU based algorithm" << endl;
		cerr << "--gpu - use only GPU based algorithm" << endl;
        return 1;
    }
	
	bool computeOnGPU = true;
	bool computeOnCPU = true;
	
	float threshold = (float)atof(argv[2]);
	
	if (argc > 3) {
		if (std::string(argv[3]) == "--cpu") {
			computeOnGPU = false;
		} else if (std::string(argv[3]) == "--gpu") {
			computeOnCPU = false;
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
	
	BYTE* cornersMapGPU = NULL; 
	BYTE* cornersMapCPU = NULL; 
				
	if (computeOnGPU) {
		cornersMapGPU = new BYTE[inputImage.getWidth() * inputImage.getHeight()];

		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, 0);

		showCUDADeviceProperties(1, dp);
			
		cudaError_t cudaStatus = harrisDetectorCUDA(inputImage, cornersMapGPU, dp.maxThreadsPerBlock, threshold);
		if (cudaStatus != cudaSuccess) {
			cerr << "CUDA Harris detector failed!";
			return 1;
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			cerr << "cudaDeviceReset failed!";
			return 1;
		}
	} 
	
	if (computeOnCPU) {
		cornersMapCPU = new BYTE[inputImage.getWidth() * inputImage.getHeight()];

		harrisDetectorCPU(inputImage, cornersMapCPU, threshold);
	}
	
	if (cornersMapGPU && cornersMapCPU) {
		makeResultComparisonReport(inputImage, cornersMapGPU, cornersMapCPU, "outputDiff.bmp");
	}
	
	if (cornersMapGPU) {
		makeOutputImage(inputImage, cornersMapGPU, "outputGPU.bmp", RGBA::Red);

		delete [] cornersMapGPU;
	}
	if (cornersMapCPU) {
		makeOutputImage(inputImage, cornersMapCPU, "outputCPU.bmp", RGBA::Blue);

		delete [] cornersMapCPU;
	}

	end = clock();
	elapsedTime = float(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "Total processing time: " << elapsedTime << " milliseconds" << endl;
	
	return 0;
}